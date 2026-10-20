"""Integration checks exercise the actual locally trained artifacts, when present."""
from io import BytesIO
from pathlib import Path
import json
import time
import numpy as np
from PIL import Image
import pytest
from backend.app import ROOT
from ml.vision import VisualModel
from tests.conftest import register
from tests.test_api import wait_report


VISUAL = ROOT / 'models' / 'image-v2' / 'model.joblib'
TEXT = ROOT / 'models' / 'text'


@pytest.mark.skipif(not VISUAL.is_file(), reason='Fine-tuned visual artifact not yet generated')
def test_trained_visual_model_is_used_by_api(client, app):
    model = VisualModel(VISUAL)
    image = Image.open(ROOT / 'data' / 'fake.jpg').convert('RGB')
    expected, card = model.predict([image])
    app.extensions['analyzer'].visual = model
    auth = register(client)
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    response = client.post('/api/analyze', data={'type':'image', 'text':'This image claims to show a false scene.', 'file':(buffer,'sample.png')},
        headers={'X-CSRF-Token':auth['csrf_token']})
    report = wait_report(client,response.json['id'])
    assert report['status']=='complete'
    result=report['result']
    assert result['visual_model_assessment']['fake_percentage']==round(expected[0]*100,1)
    assert result['models'][0]['version']==card['version']
    assert abs(result['visual_model_assessment']['fake_percentage']+result['visual_model_assessment']['real_percentage']-100)<.11
    assert result['fake_percentage'] is None
    assert result['ai_percentage'] is None


@pytest.mark.skipif(not (TEXT/'model-card.json').is_file(),reason='BERT artifact not yet generated')
def test_trained_bert_model_is_used_by_api(client,app):
    analyzer=app.extensions['analyzer']
    analyzer.config['TEXT_MODEL_PATH']=str(TEXT)
    content='The city council met on Monday to discuss a proposal to expand the public transport network. Officials said the budget would be reviewed at the next scheduled meeting.'
    probability,card=analyzer.text(content)
    assert probability is not None
    auth=register(client)
    response=client.post('/api/analyze',data={'type':'text','text':content,'language':'en'},headers={'X-CSRF-Token':auth['csrf_token']})
    report=wait_report(client,response.json['id'])
    assert report['status']=='complete'
    assert report['result']['fake_percentage']==round(probability*100,1)
    assert 'BERT' in report['result']['models'][0]['name']


@pytest.mark.skipif(not (TEXT/'model-card.json').is_file(),reason='BERT artifact not yet generated')
@pytest.mark.parametrize('content', [
    'Who is the current Prime Minister of Andhra Pradesh?',
    'India gained independence on 15 August 1947.',
])
def test_short_text_gets_experimental_scores_separate_from_truth(app, content):
    analyzer=app.extensions['analyzer']
    analyzer.config['TEXT_MODEL_PATH']=str(TEXT)
    probability,card=analyzer.text(content)
    assert probability is not None and 0 <= probability <= 1
    assert card['status']=='ready'
    assert card['experimental_input'] is True
    assessment = analyzer._text_model_assessment(probability, card)
    assert assessment['status'] == 'assessed'
    assert 'experimental' in assessment['reason']


@pytest.mark.skipif(not (TEXT/'model-card.json').is_file(),reason='BERT artifact not yet generated')
@pytest.mark.parametrize('content',[
    'ఈ వార్త నిజమా కాదా అని తెలుసుకోవడానికి ఆధారాలు పరిశీలించాలి.',
    'इस खबर की सच्चाई जानने के लिए विश्वसनीय प्रमाण देखना जरूरी है।',
])
def test_english_bert_abstains_for_unvalidated_scripts(client,app,content):
    app.extensions['analyzer'].config['TEXT_MODEL_PATH']=str(TEXT)
    auth=register(client)
    response=client.post('/api/analyze',data={'type':'text','text':content,'language':'en'},headers={'X-CSRF-Token':auth['csrf_token']})
    report=wait_report(client,response.json['id'])
    assert report['status']=='complete'
    assert report['result']['fake_percentage'] is None
    assert report['result']['models'][0]['status']=='unsupported_language'


@pytest.mark.skipif(not VISUAL.is_file(),reason='Fine-tuned visual artifact not yet generated')
def test_video_reuses_the_visual_classifier_and_aggregates_frames(client,app,tmp_path):
    import cv2
    app.extensions['analyzer'].visual=VisualModel(VISUAL)
    image=cv2.imread(str(ROOT/'data'/'fake.jpg'))
    image=cv2.resize(image,(128,128))
    path=tmp_path/'model-clip.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),5,(128,128))
    for _ in range(10):writer.write(image)
    writer.release()
    auth=register(client)
    with path.open('rb') as handle:
        response=client.post('/api/analyze',data={'type':'video','text':'This clip makes a claim about a news event.','file':(handle,'clip.mp4')},headers={'X-CSRF-Token':auth['csrf_token']})
    report=wait_report(client,response.json['id'])
    assert report['status']=='complete'
    result=report['result']
    assert len(result['frames'])==2
    assert abs(result['visual_model_assessment']['fake_percentage']-np.mean([f['visual_fake_percentage'] for f in result['frames']]))<.11
    assert result['fake_percentage'] is None
    model_card = json.loads(VISUAL.with_name('model-card.json').read_text(encoding='utf8'))
    assert result['models'][0]['version'] == model_card['version']
