"""AI-origin estimates and manipulation diagnostics remain separate and fail independently."""
from types import SimpleNamespace

from PIL import Image
import pytest


def install_visual(analyzer):
    analyzer.visual = SimpleNamespace(ready=lambda: True, predict=lambda images: (
        [.23] * len(images), {'name': 'Visual test fixture', 'version': 'test', 'scope': 'Visual patterns'}))


@pytest.mark.parametrize('score, expected_confidence', [(.82, 82.), (.05, 95.), (.5, 50.)])
def test_generation_is_independent_of_fake_real_and_confidence(app, tmp_path, score, expected_confidence):
    analyzer = app.extensions['analyzer']
    install_visual(analyzer)
    analyzer.generation = SimpleNamespace(ready=lambda: True, predict=lambda images: (
        [score] * len(images), {'name': 'Generation fixture', 'version': 'test',
        'purpose': 'ai_generation', 'source_url': 'https://example.org/model', 'limitations': ['Uncalibrated test score']}))
    image = tmp_path / 'example.png'
    Image.new('RGB', (120, 120), '#4d6972').save(image)
    result = analyzer.analyze('image', '', image, 'en', [])
    assert result['fake_percentage'] == round(score * 100, 1)
    assert result['real_percentage'] == round((1 - score) * 100, 1)
    assert result['confidence'] == expected_confidence
    assert 'uncalibrated' in result['confidence_note']
    assert result['visual_model_assessment']['fake_percentage'] == 23.
    assert result['ai_percentage'] == round(score * 100, 1)
    assert result['ai_assessment'] == {'status': 'experimental', 'calibrated': False,
        'locally_validated': False,
        'scope': 'image', 'sample_count': 1, 'source_url': 'https://example.org/model'}
    assert result['verification']['status'] == 'not_applicable'
    assert result['media_origin']['verdict'] == 'inconclusive'
    assert result['report_summary']['headline'] == 'Origin inconclusive'
    assert 'classifies this as likely AI-generated' not in result['report_summary']['summary']


@pytest.mark.parametrize('score, validated, verdict', [
    (99.9, False, 'inconclusive'), (0.1, False, 'inconclusive'),
    (None, False, 'classification_unavailable'), (50., True, 'inconclusive'),
    (85., True, 'ai_generated'), (15., True, 'likely_real'),
])
def test_origin_policy_requires_validation_and_preserves_raw_score(score, validated, verdict):
    from backend.generation import media_origin_assessment
    origin = media_origin_assessment(score, {'locally_validated': validated})
    assert origin['verdict'] == verdict
    assert origin['score'] == score


def test_generation_failure_preserves_other_analysis(app, tmp_path):
    analyzer = app.extensions['analyzer']
    install_visual(analyzer)
    def broken(images):
        raise RuntimeError('Unreadable local checkpoint')
    analyzer.generation = SimpleNamespace(ready=lambda: True, predict=broken)
    image = tmp_path / 'example.png'
    Image.new('RGB', (80, 80), 'white').save(image)
    result = analyzer.analyze('image', '', image, 'en', [])
    assert result['fake_percentage'] is None
    assert result['visual_model_assessment']['fake_percentage'] == 23.
    assert result['ai_percentage'] is None
    assert result['ai_assessment']['status'] == 'unavailable'
    assert result['sha256']


def test_text_never_borrows_an_image_or_fake_score(app):
    analyzer = app.extensions['analyzer']
    analyzer.text = lambda text: (.91, {'name': 'Text fixture', 'version': 'test', 'status': 'ready'})
    def must_not_run(images):
        pytest.fail('Image detector cannot be applied to writing')
    analyzer.generation = SimpleNamespace(ready=lambda: True, predict=must_not_run)
    result = analyzer.analyze('text', 'An English news claim.', None, 'en', [])
    assert result['fake_percentage'] == 91.
    assert result['ai_percentage'] is None
    assert any('AI-writing' in limitation for limitation in result['limitations'])
