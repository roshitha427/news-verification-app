from backend.analysis import Analyzer


def test_text_model_output_exposes_classifier_signal_separately_from_truth():
    result = Analyzer._text_model_assessment(0.82, {
        'name': 'News text classifier', 'version': 'v1', 'status': 'ready',
        'evaluation': {'accuracy': 0.88, 'test_count': 120},
    })
    assert result['status'] == 'assessed'
    assert result['verdict'] == 'fake'
    assert result['fake_percentage'] == 82.0
    assert result['real_percentage'] == 18.0
    assert result['confidence'] == 82.0
    assert 'not a fact-check' in result['reason']


def test_text_model_output_explains_when_no_classifier_is_installed():
    result = Analyzer._text_model_assessment(None, {
        'name': 'Transformer text classifier', 'version': 'not installed',
        'status': 'not_configured',
    })
    assert result['verdict'] == 'not_assessed'
    assert 'No trained text classifier is installed' in result['reason']
