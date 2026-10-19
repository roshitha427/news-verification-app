"""Check configured local application readiness without printing credentials."""
import json
from pathlib import Path


def main():
    from backend.app import ROOT, create_app

    print('AI-Powered Fake News Detection and Verification System readiness check')
    print('Frontend build: ' + ('present' if (ROOT / 'frontend/dist/index.html').is_file() else 'MISSING; run npm run build'))
    app = None
    try:
        app = create_app()
        store = app.extensions['store']
        store.ping()
        print('Database: connected (' + ('MongoDB Atlas' if store.kind == 'mongo' else 'local SQLite') + ')')
        print('Password reset email: ' + ('configured' if app.config.get('MAIL_HOST') and app.config.get('MAIL_FROM') else 'not configured'))
        ready = app.extensions['analyzer'].readiness()
        for name, enabled in ready.items():
            print(f'{name}: ' + ('configured' if enabled else 'not configured'))
        print('Evidence mode: ' + app.config['EVIDENCE_PROVIDER'])
        if app.config['EVIDENCE_PROVIDER'] == 'local' and not ready['evidence_search']:
            print('Evidence library needs source records; no external verification calls are enabled.')
        for kind, path in [('text', Path(app.config['TEXT_MODEL_PATH']) / 'model-card.json'),
                           ('image', Path(app.config['IMAGE_MODEL_PATH']).with_name('model-card.json')),
                           ('ai-writing', Path(app.config['AI_TEXT_MODEL_PATH']).with_name('model-card.json')),
                           ('video', Path(app.config['TEMPORAL_MODEL_PATH']).with_name('model-card.json'))]:
            if path.is_file():
                card = json.loads(path.read_text(encoding='utf8'))
                evaluation = card.get('evaluation')
                if evaluation:
                    print(f"{kind} historical test accuracy: {evaluation['accuracy'] * 100:.2f}% on {evaluation['test_count']} examples; not a real-world guarantee.")
                for benchmark, external in card.get('external_evaluations', {}).items():
                    print(f"{kind} external {benchmark}: {external['accuracy'] * 100:.2f}% on {external['test_count']} claims.")
                for benchmark, external in card.get('domain_evaluation', {}).items():
                    if benchmark != 'test':
                        print(f"{kind} external {benchmark}: {external['accuracy'] * 100:.2f}% on {external['test_count']} examples.")
        if ready['ai_generation']:
            print('AI-image model: experimental; independent local accuracy not measured.')
        print(f"Requested accuracy target: {app.config['MODEL_TARGET_ACCURACY'] * 100:.0f}%. No overall accuracy guarantee has been established.")
        return 0 if ready['text'] and ready['image'] and (ROOT / 'frontend/dist/index.html').is_file() else 1
    except Exception:
        print('FAILED: configured database or application startup could not be checked. Verify Atlas credentials, network access and configuration. No secret values are displayed.')
        return 1
    finally:
        if app:
            app.extensions['analysis_executor'].shutdown(wait=True)
            store = app.extensions['store']
            if store.kind == 'mongo':
                store.client.close()
            else:
                store.connection.close()


if __name__ == '__main__':
    raise SystemExit(main())
