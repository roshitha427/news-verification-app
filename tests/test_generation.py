"""Generation detection must remain separate from fake/real and fail independently."""
from types import SimpleNamespace

from PIL import Image
import pytest


def install_visual(analyzer):
    analyzer.visual = SimpleNamespace(ready=lambda: True, predict=lambda images: (
        [.23] * len(images), {'name': 'Visual test fixture', 'version': 'test', 'scope': 'Visual patterns'}))


def test_generation_is_independent_of_fake_real_and_confidence(app, tmp_path):
    analyzer = app.extensions['analyzer']
    install_visual(analyzer)
    analyzer.generation = SimpleNamespace(ready=lambda: True, predict=lambda images: (
        [.82] * len(images), {'name': 'Generation fixture', 'version': 'test',
        'purpose': 'ai_generation', 'source_url': 'https://example.org/model', 'limitations': ['Uncalibrated test score']}))
    image = tmp_path / 'example.png'
    Image.new('RGB', (120, 120), '#4d6972').save(image)
    result = analyzer.analyze('image', '', image, 'en', [])
    assert (result['fake_percentage'], result['real_percentage'], result['confidence']) == (23., 77., 77.)
    assert result['ai_percentage'] == 82.
    assert result['ai_assessment'] == {'status': 'experimental', 'calibrated': False,
        'scope': 'image', 'sample_count': 1, 'source_url': 'https://example.org/model'}
    assert result['verification']['status'] == 'unverified'


def test_generation_failure_preserves_other_analysis(app, tmp_path):
    analyzer = app.extensions['analyzer']
    install_visual(analyzer)
    def broken(images):
        raise RuntimeError('Unreadable local checkpoint')
    analyzer.generation = SimpleNamespace(ready=lambda: True, predict=broken)
    image = tmp_path / 'example.png'
    Image.new('RGB', (80, 80), 'white').save(image)
    result = analyzer.analyze('image', '', image, 'en', [])
    assert result['fake_percentage'] == 23.
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
