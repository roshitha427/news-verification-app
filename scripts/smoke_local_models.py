"""Exercise real local video/frame, AI-image, speech, text and PDF integration.

Uses an isolated SQLite database and the generated English QA recording, not Atlas.
This synthetic plumbing check does not measure detection accuracy.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json
from pathlib import Path
import subprocess
import tempfile
import time

from backend.app import ROOT, create_app
from backend.speech import ffmpeg_executable


def main():
    import torch
    torch.set_num_threads(3)
    recording = ROOT / 'data' / 'speech-check.wav'
    photo = ROOT / 'data' / 'fake.jpg'
    if not recording.is_file() or not photo.is_file() or not ffmpeg_executable():
        raise SystemExit('Need the local speech-check.wav, fake.jpg and FFmpeg QA fixtures.')
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='models-smoke-', dir=artifacts) as scratch:
        scratch = Path(scratch)
        video = scratch / 'spoken-clip.mp4'
        subprocess.run([ffmpeg_executable(), '-nostdin', '-y', '-loglevel', 'error', '-loop', '1',
            '-framerate', '5', '-i', str(photo), '-i', str(recording), '-shortest', '-t', '12',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(video)],
            check=True, timeout=40, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        app = create_app({'DATABASE_BACKEND': 'sqlite', 'SQLITE_PATH': str(scratch / 'qa.sqlite'),
            'SECRET_KEY': 'isolated-model-integration-check', 'EVIDENCE_PROVIDER': 'local',
            'EVIDENCE_MODEL_PATH': str(scratch / 'no-production-corpus.joblib'), 'TESTING': True})
        client = app.test_client()
        try:
            token = client.get('/api/session').json['csrf_token']
            registration = client.post('/api/auth/register', json={'name': 'Model QA',
                'email': 'local-model-qa@example.invalid', 'password': 'temporary-local-model-testing'},
                headers={'X-CSRF-Token': token})
            assert registration.status_code == 200
            token = registration.json['csrf_token']
            with video.open('rb') as media:
                submission = client.post('/api/analyze', data={'type': 'video', 'text': 'The speaker discusses a reported public event.', 'language': 'en',
                    'file': (media, 'spoken-clip.mp4')}, headers={'X-CSRF-Token': token})
            assert submission.status_code == 202, 'Video submission failed'
            report_id = submission.json['id']
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                report = client.get('/api/reports/' + report_id).json
                if report['status'] in ('complete', 'failed'):
                    break
                time.sleep(.4)
            assert report['status'] == 'complete', 'Model integration failed or timed out'
            result = report['result']
            assert result['fake_percentage'] is not None and result['ai_percentage'] is not None
            assert result['ai_assessment']['scope'] == 'sampled_video_frames'
            assert len(result['frames']) >= 2 and all('ai_percentage' in frame for frame in result['frames'])
            assert 'council' in result.get('transcript', '').lower()
            # The fixture transcript is intentionally too short for the
            # article classifier's claim-domain guard; it must abstain rather
            # than return a misleading fake-news score.
            assert result['speech_text_assessment']['fake_percentage'] is None
            assert result['speech_text_assessment']['model']['status'] == 'claim_out_of_domain'
            assert result['verification']['provider'] == 'Local evidence model'
            assert result['verification']['provider_status'] == 'not_configured'
            pdf = client.get('/api/reports/' + report_id + '/pdf')
            assert pdf.status_code == 200 and pdf.data.startswith(b'%PDF')
            (artifacts / 'local-video-models.pdf').write_bytes(pdf.data)
            (artifacts / 'local-video-models.json').write_text(json.dumps(report, indent=2), encoding='utf8')
            print(json.dumps({'video': 'passed', 'shared_frame_model': 'passed', 'ai_frame_scores': 'passed',
                'speech_transcript': 'passed', 'spoken_text_classifier': 'safely_abstained', 'pdf': 'passed',
                'external_verification_calls': 0, 'sampled_frames': len(result['frames']), 'accuracy_evaluation': False}))
        finally:
            app.extensions['analysis_executor'].shutdown(wait=True)
            app.extensions['store'].connection.close()


if __name__ == '__main__':
    main()
