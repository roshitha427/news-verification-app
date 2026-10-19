"""Loopback-only browser QA with an isolated disposable SQLite database."""
from backend.app import create_app, ROOT
from werkzeug.security import generate_password_hash
from waitress import serve

if __name__=='__main__':
    import argparse
    import os
    import tempfile
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence-provider',choices=['none','public'],default='none')
    args=parser.parse_args()
    # Each QA server owns fresh disposable state; old QA metadata can predate
    # uniqueness constraints and must not block today's browser checks.
    (ROOT/'artifacts').mkdir(exist_ok=True)
    handle, database_path = tempfile.mkstemp(prefix='qa-', suffix='.sqlite', dir=ROOT/'artifacts')
    os.close(handle)
    app=create_app({'APP_ENV':'development','DATABASE_BACKEND':'sqlite',
        'SQLITE_PATH':database_path,'SECRET_KEY':'local-browser-qa-only-session-key',
        'SESSION_COOKIE_SECURE':False,'GOOGLE_FACT_CHECK_API_KEY':None,'WHISPER_MODEL':None,'EVIDENCE_PROVIDER':args.evidence_provider})
    store=app.extensions['store']
    if not store.one('sources',{'domain':'factly.in'}):
        store.save('sources',dict(name='Factly',url='https://factly.in',domain='factly.in',description='India · Fact checking',enabled=True))
    email='qa-admin@example.invalid'
    if not store.one('users',{'email':email}):
        store.save('users',{'name':'QA Administrator','email':email,'role':'admin','disabled':False,
                           'password_hash':generate_password_hash('local-browser-qa-only-password')})
    print('Isolated browser QA ready at http://127.0.0.1:5001',flush=True)
    serve(app,host='127.0.0.1',port=5001,threads=6)
