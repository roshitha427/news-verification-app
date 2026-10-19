import pytest
from backend.app import create_app


@pytest.fixture
def app(tmp_path):
    app = create_app({'TESTING': True, 'DATABASE_BACKEND': 'sqlite', 'SECRET_KEY': 'test-only-key',
        'SQLITE_PATH': str(tmp_path / 'test.sqlite'), 'IMAGE_MODEL_PATH': str(tmp_path / 'missing.joblib'),
        'TEXT_MODEL_PATH': str(tmp_path / 'text'), 'AI_IMAGE_MODEL_PATH': str(tmp_path / 'ai-image'),
        'TEMPORAL_MODEL_PATH': str(tmp_path / 'temporal.joblib'),
        'AI_TEXT_MODEL_PATH': str(tmp_path / 'ai-text.joblib'),
        'GOOGLE_FACT_CHECK_API_KEY': None, 'GEMINI_API_KEY': None, 'WHISPER_MODEL': None, 'EVIDENCE_PROVIDER':'none',
        'MAIL_HOST':None, 'MAIL_FROM':None, 'MAIL_USERNAME':None, 'MAIL_PASSWORD':None})
    yield app
    app.extensions['analysis_executor'].shutdown(wait=True)
    app.extensions['mail_executor'].shutdown(wait=True)


@pytest.fixture
def client(app):
    return app.test_client()


def token(client):
    return client.get('/api/session').json['csrf_token']


def register(client, email='reader@example.com'):
    result = client.post('/api/auth/register', json={'name':'Reader', 'email':email, 'password':'long-test-password-123'},
                         headers={'X-CSRF-Token':token(client)})
    assert result.status_code == 200
    return result.json
