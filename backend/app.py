import atexit
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import wraps
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
import time
from urllib.parse import urlparse

from dotenv import load_dotenv
from flask import Flask, abort, g, jsonify, request, send_file, send_from_directory, session
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from backend.analysis import Analyzer
from backend.mailer import ready as mail_ready, send_password_reset_email
from backend.pdf_report import report_pdf
from backend.store import Store, now
from backend.verification import DEFAULT_SOURCES, verify

ROOT = Path(__file__).resolve().parents[1]


def create_app(overrides=None):
    load_dotenv(ROOT / 'backend' / '.env')
    app = Flask(__name__, static_folder=None)
    config = dict(APP_ENV=os.getenv('APP_ENV', 'development'), SECRET_KEY=os.getenv('SECRET_KEY'),
                  DATABASE_BACKEND=os.getenv('DATABASE_BACKEND', 'sqlite'), MONGODB_URI=os.getenv('MONGODB_URI'),
                  MONGODB_DATABASE=os.getenv('MONGODB_DATABASE', 'veritas'),
                  SQLITE_PATH=os.getenv('SQLITE_PATH', str(ROOT / 'data' / 'development.sqlite')),
                  IMAGE_MODEL_PATH=str(ROOT / os.getenv('IMAGE_MODEL_PATH', 'models/image-v2/model.joblib')),
                  AI_IMAGE_MODEL_PATH=str(ROOT / os.getenv('AI_IMAGE_MODEL_PATH', 'models/ai-image')),
                  TEXT_MODEL_PATH=str(ROOT / os.getenv('TEXT_MODEL_PATH', 'models/text')),
                  GOOGLE_FACT_CHECK_API_KEY=os.getenv('GOOGLE_FACT_CHECK_API_KEY'),
                  GEMINI_API_KEY=os.getenv('GEMINI_API_KEY'),
                  GEMINI_MODEL=os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite'),
                  WHISPER_MODEL=str(ROOT / os.environ['WHISPER_MODEL']) if os.getenv('WHISPER_MODEL') else None,
                  EVIDENCE_PROVIDER=os.getenv('EVIDENCE_PROVIDER', 'public'),
                  EVIDENCE_MODEL_PATH=str(ROOT / os.getenv('EVIDENCE_MODEL_PATH', 'models/evidence/index.joblib')),
                  PUBLIC_BASE_URL=os.getenv('PUBLIC_BASE_URL', 'http://127.0.0.1:5000'),
                  PASSWORD_RESET_TTL_SECONDS=int(os.getenv('PASSWORD_RESET_TTL_SECONDS', '3600')),
                  MAIL_HOST=os.getenv('MAIL_HOST'), MAIL_PORT=int(os.getenv('MAIL_PORT', '587')),
                  MAIL_USERNAME=os.getenv('MAIL_USERNAME'), MAIL_PASSWORD=os.getenv('MAIL_PASSWORD'),
                  MAIL_FROM=os.getenv('MAIL_FROM'), MAIL_USE_TLS=os.getenv('MAIL_USE_TLS', 'true').lower() == 'true',
                  MAIL_USE_SSL=os.getenv('MAIL_USE_SSL', 'false').lower() == 'true',
                  MAIL_TIMEOUT_SECONDS=float(os.getenv('MAIL_TIMEOUT_SECONDS', '12')),
                  MODEL_TARGET_ACCURACY=float(os.getenv('MODEL_TARGET_ACCURACY', '.90')),
                  TEMPORAL_MODEL_PATH=str(ROOT / os.getenv('TEMPORAL_MODEL_PATH', 'models/video-temporal/model.joblib')),
                  AI_TEXT_MODEL_PATH=str(ROOT / os.getenv('AI_TEXT_MODEL_PATH', 'models/ai-text/model.joblib')),
                  MAX_CONTENT_LENGTH=50 * 1024 * 1024, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Strict',
                  SESSION_COOKIE_SECURE=os.getenv('COOKIE_SECURE', 'false').lower() == 'true',
                  PERMANENT_SESSION_LIFETIME=timedelta(hours=8), TESTING=False, RECOVER_INTERRUPTED_JOBS=False)
    config.update(overrides or {})
    if not .5 < config['MODEL_TARGET_ACCURACY'] < 1:
        raise RuntimeError('MODEL_TARGET_ACCURACY must be between .5 and 1.')
    if config['EVIDENCE_PROVIDER'] not in ('local', 'public', 'rss', 'gdelt', 'google', 'auto', 'none'):
        raise RuntimeError('EVIDENCE_PROVIDER must be local, public, rss, gdelt, google, auto, or none.')
    if not config['SECRET_KEY']:
        if config['APP_ENV'] == 'production':
            raise RuntimeError('Set a long random SECRET_KEY in production.')
        keyfile = ROOT / 'data' / 'session.key'
        keyfile.parent.mkdir(parents=True, exist_ok=True)
        if not keyfile.exists():
            keyfile.write_text(secrets.token_hex(32))
        config['SECRET_KEY'] = keyfile.read_text().strip()
    if config['APP_ENV'] == 'production' and (config['DATABASE_BACKEND'] != 'mongo' or not config['SESSION_COOKIE_SECURE']):
        raise RuntimeError('Production requires MongoDB and COOKIE_SECURE=true behind HTTPS.')
    if not 300 <= config['PASSWORD_RESET_TTL_SECONDS'] <= 86400:
        raise RuntimeError('PASSWORD_RESET_TTL_SECONDS must be between 300 and 86400.')
    if not 1 <= config['MAIL_PORT'] <= 65535 or config['MAIL_TIMEOUT_SECONDS'] <= 0:
        raise RuntimeError('MAIL_PORT and MAIL_TIMEOUT_SECONDS must be positive valid values.')
    public_url = urlparse(config['PUBLIC_BASE_URL'])
    if (public_url.scheme not in ('http', 'https') or not public_url.hostname or public_url.username
            or public_url.password or public_url.query or public_url.fragment):
        raise RuntimeError('PUBLIC_BASE_URL must be an absolute HTTP(S) URL without credentials, query, or fragment.')
    if config['APP_ENV'] == 'production' and (public_url.scheme != 'https' or not mail_ready(config)):
        raise RuntimeError('Production requires HTTPS PUBLIC_BASE_URL and configured SMTP password-reset email.')
    if config['MAIL_USE_TLS'] and config['MAIL_USE_SSL']:
        raise RuntimeError('Choose either MAIL_USE_TLS or MAIL_USE_SSL, not both.')
    app.config.update(config)
    store = Store(app.config)
    analyzer = Analyzer(app.config)
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='veritas-analysis')
    slots = threading.BoundedSemaphore(4)
    mail_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='veritas-mail')
    mail_slots = threading.BoundedSemaphore(8)
    app.extensions.update(store=store, analyzer=analyzer, analysis_executor=executor,
                          mail_executor=mail_executor, mail_slots=mail_slots)
    atexit.register(executor.shutdown, wait=False)
    atexit.register(mail_executor.shutdown, wait=False)
    rate_lock = threading.Lock()
    rates = defaultdict(deque)

    if not store.find('sources', limit=1):
        for name, url, description in DEFAULT_SOURCES:
            store.save('sources', dict(name=name, url=url, domain=urlparse(url).hostname.removeprefix('www.'),
                                       description=description, enabled=True))
    # The local executor cannot resume work after a process restart.
    for state in (('queued', 'analyzing') if app.config['RECOVER_INTERRUPTED_JOBS'] else ()):
        for report in store.find('reports', {'status': state}, limit=10000):
            report.update(status='failed', error='Analysis was interrupted by a server restart. Please submit again.')
            store.save('reports', report)

    def public_user(user):
        return {k: user[k] for k in ('id', 'name', 'email', 'role', 'created_at', 'disabled')}

    def json_object():
        # All JSON endpoints accept objects. A valid JSON array or scalar should
        # produce a useful client error instead of an AttributeError and a 500.
        if request.content_length and request.content_length > 64 * 1024:
            abort(413, description='JSON requests must be smaller than 64 KB.')
        body = request.get_json()
        if not isinstance(body, dict):
            abort(400, description='Send a JSON object with the required fields.')
        return body

    def text_field(body, key, default=''):
        value = body.get(key, default)
        if not isinstance(value, str):
            abort(400, description=f'{key} must be text.')
        return value

    def unique_record(collection, document, message):
        try:
            return store.save(collection, document)
        except Exception as exc:
            if exc.__class__.__name__ in ('DuplicateKeyError', 'IntegrityError'):
                abort(409, description=message)
            raise

    def release_analysis_resources(filepath):
        try:
            if filepath:
                Path(filepath).unlink(missing_ok=True)
        except OSError:
            app.logger.exception('Could not remove a temporary analysis upload')
        finally:
            # A locked/deleted temporary file must never consume queue capacity.
            slots.release()

    def protect(admin=False):
        def decorate(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                if not g.user:
                    abort(401, description='Please sign in to continue.')
                if admin and g.user['role'] != 'admin':
                    abort(403, description='Administrator access is required.')
                return fn(*args, **kwargs)
            return wrapper
        return decorate

    def throttle(key, limit, period=60):
        if app.config['TESTING']:
            return
        stamp = time.monotonic()
        with rate_lock:
            # Bound IP entries as well as timestamps.
            if len(rates) > 5000:
                rates.clear()
            bucket = rates[key]
            while bucket and bucket[0] < stamp - period:
                bucket.popleft()
            if len(bucket) >= limit:
                abort(429, description='Too many requests. Please wait a minute and try again.')
            bucket.append(stamp)

    @app.before_request
    def before():
        g.user = None
        # Health must report database outages consistently even when the browser
        # has a signed-in session; resolving that user can itself fail offline.
        if session.get('user_id') and request.path != '/api/health':
            user = store.one('users', {'id': session['user_id']})
            if (user and not user.get('disabled')
                    and session.get('session_version', 0) == user.get('session_version', 0)):
                g.user = user
            else:
                session.clear()
        if request.path.startswith('/api/') and request.method in ('POST', 'PATCH', 'DELETE', 'PUT'):
            token = request.headers.get('X-CSRF-Token', '')
            if not token or not hmac.compare_digest(token.encode('utf8'), session.get('csrf', '').encode('utf8')):
                abort(403, description='Your session has expired. Refresh the page and try again.')

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; font-src 'self'; media-src 'self' blob:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        if request.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        app.logger.exception('Unhandled request failure')
        return jsonify(error='The request could not be completed. Please try again.'), 500

    @app.get('/api/health')
    def health():
        try:
            store.ping()
            return jsonify(status='ok', database=store.kind, database_connected=True,
                           password_reset_email=mail_ready(app.config), models=analyzer.readiness())
        except Exception:
            return jsonify(status='degraded', database=store.kind, database_connected=False,
                           password_reset_email=mail_ready(app.config)), 503

    @app.get('/api/session')
    def get_session():
        session.setdefault('csrf', secrets.token_urlsafe(32))
        return jsonify(user=public_user(g.user) if g.user else None, csrf_token=session['csrf'])

    def auth_response(user):
        session.clear()
        session.update(user_id=user['id'], session_version=user.get('session_version', 0), csrf=secrets.token_urlsafe(32))
        session.permanent = True
        return jsonify(user=public_user(user), csrf_token=session['csrf'])

    @app.post('/api/auth/register')
    def register():
        throttle(('auth', request.remote_addr), 10)
        body = json_object()
        name, email, password = text_field(body, 'name').strip(), text_field(body, 'email').strip().lower(), text_field(body, 'password')
        if not 2 <= len(name) <= 80 or not re.fullmatch(r'[^\s@]{1,64}@[^\s@.]+\.[^\s@]{2,}', email) or len(email) > 254:
            abort(400, description='Enter your name and a valid email address.')
        if not 12 <= len(password) <= 128:
            abort(400, description='Use a password between 12 and 128 characters.')
        if store.one('users', {'email': email}):
            abort(409, description='This email is already registered. Please sign in.')
        user = unique_record('users', dict(name=name, email=email, password_hash=generate_password_hash(password), role='user', disabled=False, session_version=0),
                             'This email is already registered. Please sign in.')
        return auth_response(user)

    @app.post('/api/auth/login')
    def login():
        throttle(('auth', request.remote_addr), 10)
        body = json_object()
        email = text_field(body, 'email').strip().lower()
        password = text_field(body, 'password')
        user = store.one('users', {'email': email})
        # Constant-cost dummy check when an email does not exist.
        valid = check_password_hash(user['password_hash'] if user else DUMMY_HASH, password[:129])
        if not user or not valid or user.get('disabled'):
            abort(401, description='Email or password is incorrect, or the account is disabled.')
        return auth_response(user)

    @app.post('/api/auth/forgot-password')
    def forgot_password():
        throttle(('password-reset-ip', request.remote_addr), 10, period=3600)
        body = json_object()
        email = text_field(body, 'email').strip().lower()
        if not re.fullmatch(r'[^\s@]{1,64}@[^\s@.]+\.[^\s@]{2,}', email) or len(email) > 254:
            abort(400, description='Enter a valid email address.')
        if not mail_ready(app.config):
            abort(503, description='Password reset email is not configured. Contact the site administrator.')
        throttle(('password-reset-email', email), 5, period=3600)
        user = store.one('users', {'email': email})
        if user and not user.get('disabled'):
            # Replace earlier links for this account and store only a digest of
            # the new opaque token. A TTL index reclaims expired Atlas records.
            for old in store.find('password_resets', {'user_id': user['id']}, limit=100):
                store.delete('password_resets', old['id'])
            token = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(token.encode('utf8')).hexdigest()
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=app.config['PASSWORD_RESET_TTL_SECONDS'])
            store.save('password_resets', {'id': token_hash, 'user_id': user['id'], 'expires_at': expires_at})
            reset_url = app.config['PUBLIC_BASE_URL'].rstrip('/') + '/#reset/' + token

            mail_slots = app.extensions['mail_slots']
            if not mail_slots.acquire(blocking=False):
                store.delete('password_resets', token_hash)
                app.logger.warning('Password reset email queue is full for account %s', user['id'])
            else:
                def deliver_reset():
                    try:
                        send_password_reset_email(app.config, email, user.get('name', ''), reset_url)
                    except Exception:
                        store.delete('password_resets', token_hash)
                        app.logger.exception('Password reset email delivery failed for account %s', user['id'])
                    finally:
                        mail_slots.release()

                try:
                    app.extensions['mail_executor'].submit(deliver_reset)
                except RuntimeError:
                    mail_slots.release()
                    store.delete('password_resets', token_hash)
                    app.logger.exception('Password reset email could not be queued for account %s', user['id'])
        return jsonify(message='If an account uses that email, a reset link will be sent shortly.'), 200

    @app.post('/api/auth/reset-password')
    def reset_password():
        throttle(('password-reset-ip', request.remote_addr), 20, period=3600)
        body = json_object()
        token = text_field(body, 'token')
        password = text_field(body, 'password')
        if len(token) > 200 or not 12 <= len(password) <= 128:
            abort(400, description='The reset link or new password is invalid.')
        token_hash = hashlib.sha256(token.encode('utf8')).hexdigest()
        reset = store.take('password_resets', token_hash)
        if not reset:
            abort(400, description='The reset link is invalid or expired. Request a new link.')
        expires_at = reset.get('expires_at')
        if isinstance(expires_at, str):
            try:
                expires_at = datetime.fromisoformat(expires_at)
            except ValueError:
                expires_at = None
        if expires_at and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if not isinstance(expires_at, datetime) or expires_at <= datetime.now(timezone.utc):
            abort(400, description='The reset link is invalid or expired. Request a new link.')
        user = store.one('users', {'id': reset.get('user_id')})
        if not user or user.get('disabled'):
            abort(400, description='The reset link is invalid or expired. Request a new link.')
        user['password_hash'] = generate_password_hash(password)
        user['session_version'] = int(user.get('session_version', 0)) + 1
        store.save('users', user)
        return jsonify(ok=True)

    @app.post('/api/auth/logout')
    def logout():
        session.clear()
        session['csrf'] = secrets.token_urlsafe(32)
        return jsonify(ok=True, csrf_token=session['csrf'])

    @app.patch('/api/profile')
    @protect()
    def update_profile():
        throttle(('profile', g.user['id']), 10)
        body = json_object()
        allowed = {'name', 'email', 'current_password', 'password', 'confirm_password'}
        if set(body) - allowed or not set(body).intersection({'name', 'email', 'password'}):
            abort(400, description='Update your name, email address, or password.')

        user = dict(g.user)
        if 'name' in body:
            name = text_field(body, 'name').strip()
            if not 2 <= len(name) <= 80:
                abort(400, description='Name must be between 2 and 80 characters.')
            user['name'] = name

        email_changed = False
        if 'email' in body:
            email = text_field(body, 'email').strip().lower()
            if not re.fullmatch(r'[^\s@]{1,64}@[^\s@.]+\.[^\s@]{2,}', email) or len(email) > 254:
                abort(400, description='Enter a valid email address.')
            email_changed = email != user['email']
            user['email'] = email

        password_changed = 'password' in body
        if password_changed:
            password = text_field(body, 'password')
            confirmation = text_field(body, 'confirm_password')
            if not 12 <= len(password) <= 128:
                abort(400, description='Use a password between 12 and 128 characters.')
            if password != confirmation:
                abort(400, description='The new passwords do not match.')

        if email_changed or password_changed:
            current_password = text_field(body, 'current_password')
            if not check_password_hash(g.user['password_hash'], current_password[:129]):
                abort(401, description='Enter your current password to change your email or password.')
        if password_changed:
            user['password_hash'] = generate_password_hash(password)

        updated = unique_record('users', user, 'This email is already registered.')
        return jsonify(user=public_user(updated))

    @app.get('/api/sources')
    def sources():
        return jsonify(items=store.find('sources', {'enabled': True}))

    @app.get('/api/reports')
    @protect()
    def reports():
        items = store.find('reports', {'owner_id': g.user['id']}, limit=200)
        return jsonify(items=[{k: v for k, v in item.items() if k not in ('content',)} for item in items])

    def owned_report(report_id):
        report = store.one('reports', {'id': report_id, 'owner_id': g.user['id']})
        if not report:
            abort(404, description='Report not found.')
        return report

    def ensure_report_summary(report):
        """Upgrade older completed reports when they are opened or downloaded."""
        if report.get('status') != 'complete':
            return report
        result = report.get('result') or {}
        if (result.get('report_summary') or {}).get('schema_version', 0) >= 4:
            return report

        from backend.report_summary import filter_relevant_evidence, summarize_report

        kind = report.get('type', 'text')
        claim = str(report.get('content') or result.get('transcript') or '').strip()
        transcript = str(result.get('transcript') or '').strip()
        if transcript and transcript not in claim:
            claim = '\n'.join(part for part in (claim, transcript) if part)
        verification = result.get('verification') or {'evidence': [], 'provider_status': 'unavailable'}
        for item in verification.get('evidence') or []:
            if isinstance(item, dict):
                item.setdefault('retrieved_at', verification.get('checked_at'))
                item.setdefault('source_type', item.get('kind', 'source'))
        verification = filter_relevant_evidence(claim, verification)

        # Older records may not have saved evidence, or may contain only
        # unrelated matches. Retry retrieval once before settling on Unverified.
        if claim and not verification.get('evidence'):
            try:
                fresh = verify(
                    claim,
                    report.get('language', 'en'),
                    app.config.get('GOOGLE_FACT_CHECK_API_KEY'),
                    store.find('sources', {'enabled': True}),
                    provider=app.config.get('EVIDENCE_PROVIDER', 'local'),
                    evidence_path=app.config.get('EVIDENCE_MODEL_PATH'),
                )
                fresh = filter_relevant_evidence(claim, fresh)
                if fresh.get('evidence') or not verification.get('evidence'):
                    verification = fresh
            except Exception:
                app.logger.warning('Evidence refresh failed while upgrading a saved report.')

        result['verification'] = verification
        result['report_summary'] = summarize_report(
            claim, kind, result, verification,
            api_key=app.config.get('GEMINI_API_KEY'),
            model=app.config.get('GEMINI_MODEL', 'gemini-3.5-flash-lite'),
        )
        report['result'] = result
        store.save('reports', report)
        return report

    @app.get('/api/reports/<report_id>')
    @protect()
    def get_report(report_id):
        return jsonify(ensure_report_summary(owned_report(report_id)))

    @app.post('/api/reports/<report_id>/ask')
    @protect()
    def ask_report(report_id):
        throttle(('report-question', g.user['id']), 20, period=3600)
        report = owned_report(report_id)
        if report.get('status') != 'complete':
            abort(409, description='Wait for the report to finish before asking about its sources.')
        question = text_field(json_object(), 'question').strip()
        if not 3 <= len(question) <= 1000:
            abort(400, description='Enter a question between 3 and 1,000 characters.')
        from backend.rag import answer_from_report
        return jsonify(answer_from_report(question, report,
            api_key=app.config.get('GEMINI_API_KEY'), model=app.config.get('GEMINI_MODEL', 'gemini-3.5-flash-lite')))

    @app.delete('/api/reports/<report_id>')
    @protect()
    def delete_report(report_id):
        report = owned_report(report_id)
        if report['status'] in ('queued', 'analyzing'):
            abort(409, description='Wait for analysis to finish before deleting this report.')
        store.delete('reports', report_id)
        return jsonify(ok=True)

    @app.get('/api/reports/<report_id>/pdf')
    @protect()
    def pdf(report_id):
        from io import BytesIO
        report = ensure_report_summary(owned_report(report_id))
        if report['status'] != 'complete':
            abort(409, description='The report is not ready yet.')
        return send_file(BytesIO(report_pdf(report)), mimetype='application/pdf', as_attachment=True,
                         download_name=f"veritas-{report['id'][:10]}.pdf")

    @app.post('/api/analyze')
    @protect()
    def analyze():
        throttle(('analysis', g.user['id']), 10)
        kind = request.form.get('type', 'text')
        text = request.form.get('text', '').strip()
        language = request.form.get('language', 'en')
        if kind not in ('text', 'image', 'video') or language not in ('en', 'hi', 'te'):
            abort(400, description='Select a supported content type and language.')
        if len(text) > 12000 or (kind == 'text' and len(text) < 20):
            abort(400, description='Enter between 20 and 12,000 characters of text.')
        media = request.files.get('file')
        if kind != 'text' and (not media or not media.filename):
            abort(400, description='Choose an image or video to analyze.')
        filename = secure_filename(media.filename) if media else ''
        allowed = {'.png', '.jpg', '.jpeg', '.webp'} if kind == 'image' else {'.mp4', '.webm', '.mov'}
        if kind != 'text' and Path(filename).suffix.lower() not in allowed:
            abort(400, description='Use JPG, PNG, WebP for images, or MP4, MOV, WebM for videos.')
        if not slots.acquire(blocking=False):
            abort(429, description='All analysis slots are busy. Please try again shortly.')
        filepath = None
        report = None
        try:
            if media and kind != 'text':
                # Random, private temporary names. Never serve user uploads as active web assets.
                with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix.lower(), delete=False) as handle:
                    filepath = handle.name
                media.save(filepath)
                if os.path.getsize(filepath) == 0:
                    abort(400, description='The uploaded file is empty.')
                if kind == 'image':
                    from PIL import Image
                    try:
                        with Image.open(filepath) as image:
                            if image.format not in ('JPEG', 'PNG', 'WEBP') or image.width * image.height > 30_000_000:
                                raise ValueError()
                            image.verify()
                    except Exception:
                        abort(400, description='The image is invalid or exceeds 30 megapixels.')
            report = store.save('reports', dict(owner_id=g.user['id'], type=kind, language=language,
                title=text[:110] if text else filename, content=text, filename=filename, status='queued', result=None))
            enabled_sources = store.find('sources', {'enabled': True})
            def run():
                try:
                    report['status'] = 'analyzing'
                    store.save('reports', report)
                    report['result'] = analyzer.analyze(kind, text, filepath, language, enabled_sources)
                    report.update(status='complete', completed_at=now())
                except Exception as exc:
                    logging.exception('Analysis failed for report %s', report['id'])
                    report.update(status='failed', error=str(exc) if isinstance(exc, ValueError) else 'Analysis could not be completed. Check the installed model and media format.')
                finally:
                    try:
                        store.save('reports', report)
                    finally:
                        release_analysis_resources(filepath)
            executor.submit(run)
            return jsonify(id=report['id'], status='queued'), 202
        except Exception:
            # A report may already have been inserted when queue submission or
            # source loading fails. Do not leave it queued forever in history.
            if report is not None:
                try:
                    report.update(status='failed', error='Analysis could not be scheduled. Please submit again.')
                    store.save('reports', report)
                except Exception:
                    app.logger.exception('Could not persist failed scheduling for report %s', report['id'])
            release_analysis_resources(filepath)
            raise

    @app.get('/api/admin/overview')
    @protect(admin=True)
    def admin_overview():
        cards = {}
        for label, path in [('image', Path(app.config['IMAGE_MODEL_PATH']).with_name('model-card.json')),
                            ('text', Path(app.config['TEXT_MODEL_PATH']) / 'model-card.json'),
                            ('video', Path(app.config['TEMPORAL_MODEL_PATH']).with_name('model-card.json')),
                            ('ai_writing', Path(app.config['AI_TEXT_MODEL_PATH']).with_name('model-card.json'))]:
            if path.is_file():
                cards[label] = json.loads(path.read_text(encoding='utf8'))
        return jsonify(counts={name: store.count(name) for name in ('users', 'reports', 'sources', 'datasets')},
                       models=analyzer.readiness(), model_cards=cards, database=store.kind,
                       password_reset_email=mail_ready(app.config),
                       reports=store.find('reports', limit=100), datasets=store.find('datasets'),
                       sources=store.find('sources'), users=[public_user(u) for u in store.find('users')])

    @app.patch('/api/admin/users/<user_id>')
    @protect(admin=True)
    def manage_user(user_id):
        if user_id == g.user['id']:
            abort(400, description='You cannot disable your own administrator account.')
        user = store.one('users', {'id': user_id})
        if not user:
            abort(404)
        disabled = json_object().get('disabled')
        if not isinstance(disabled, bool):
            abort(400, description='disabled must be true or false.')
        user['disabled'] = disabled
        store.save('users', user)
        return jsonify(user=public_user(user))

    @app.post('/api/admin/sources')
    @protect(admin=True)
    def add_source():
        body = json_object()
        url, name = text_field(body, 'url').strip(), text_field(body, 'name').strip()
        description = text_field(body, 'description', 'Trusted source')[:200]
        try:
            parsed = urlparse(url)
            valid_url = (parsed.scheme == 'https' and parsed.hostname and not parsed.username and not parsed.password
                         and parsed.port in (None, 443) and not any(char.isspace() for char in url))
        except ValueError:
            valid_url = False
        if not valid_url or not 2 <= len(name) <= 100 or len(url) > 500:
            abort(400, description='Enter a source name and a valid HTTPS URL without credentials.')
        domain = parsed.hostname.lower().removeprefix('www.')
        if store.one('sources', {'domain': domain}):
            abort(409, description='This source domain is already registered. Enable or disable the existing source.')
        return jsonify(unique_record('sources', dict(name=name, url=url, domain=domain,
            description=description, enabled=True), 'This source domain is already registered.')), 201

    @app.patch('/api/admin/sources/<source_id>')
    @protect(admin=True)
    def toggle_source(source_id):
        source = store.one('sources', {'id': source_id})
        enabled = json_object().get('enabled')
        if not source:
            abort(404)
        if not isinstance(enabled, bool):
            abort(400)
        source['enabled'] = enabled
        return jsonify(store.save('sources', source))

    @app.post('/api/admin/datasets')
    @protect(admin=True)
    def add_dataset():
        body = json_object()
        if not all(isinstance(body.get(k), str) and 1 <= len(body[k]) <= 200 for k in ('name', 'version', 'modality')):
            abort(400, description='Dataset name, version, and modality are required.')
        counts = body.get('counts', {})
        if not isinstance(counts, dict) or len(counts) > 30 or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts.values()):
            abort(400, description='Dataset counts must be nonnegative integers.')
        if body['modality'] not in ('text', 'image', 'video'):
            abort(400, description='Choose text, image, or video.')
        if store.one('datasets', {'name': body['name'], 'version': body['version']}):
            abort(409, description='This dataset version is already registered. Use a new version for updated data.')
        return jsonify(unique_record('datasets', {k: body.get(k, {}) for k in ('name', 'version', 'modality', 'counts')},
                       'This dataset version is already registered. Use a new version for updated data.')), 201

    @app.cli.command('create-admin')
    def create_admin():
        """Create an administrator interactively; no shared default password."""
        import click
        name = click.prompt('Name').strip()
        email = click.prompt('Email').strip().lower()
        password = click.prompt('Password (12-128 characters)', hide_input=True, confirmation_prompt=True)
        if (not 2 <= len(name) <= 80 or not 12 <= len(password) <= 128 or len(email) > 254
                or not re.fullmatch(r'[^\s@]{1,64}@[^\s@.]+\.[^\s@]{2,}', email)):
            raise click.ClickException('Use a name with 2-80 characters, a valid email, and a password with 12-128 characters.')
        if store.one('users', {'email': email}):
            raise click.ClickException('Email already exists. Use a separate administrator address.')
        store.save('users', dict(name=name, email=email, password_hash=generate_password_hash(password), role='admin', disabled=False))
        click.echo('Administrator created.')

    @app.get('/')
    @app.get('/<path:path>')
    def frontend(path='index.html'):
        if path.startswith('api/'):
            abort(404, description='API route not found.')
        directory = ROOT / 'frontend' / 'dist'
        if not directory.exists():
            return jsonify(message='Frontend not built. Run npm run build or use the Vite development server.'), 503
        candidate = (directory / path).resolve()
        if directory.resolve() not in candidate.parents:
            abort(404)
        return send_from_directory(directory, path if candidate.is_file() else 'index.html')

    return app


DUMMY_HASH = generate_password_hash('constant-cost-dummy-password')


def create_server_app():
    """Recover jobs only on server startup, never during admin/dataset CLI commands."""
    return create_app({'RECOVER_INTERRUPTED_JOBS': True})

if __name__ == '__main__':
    create_server_app().run(host='127.0.0.1', port=int(os.getenv('PORT', '5000')), debug=False)
