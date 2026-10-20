from backend.analysis import Analyzer
from tests.conftest import register


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


def test_opening_old_short_text_adds_scores_without_changing_evidence_verdict(client, app, monkeypatch):
    auth = register(client)
    report = app.extensions['store'].save('reports', {
        'owner_id': auth['user']['id'], 'type': 'text', 'status': 'complete',
        'content': 'A short historical test claim.', 'title': 'Short test claim',
        'result': {'text_model_assessment': {'status': 'claim_out_of_domain'},
                   'models': [{'name': 'BERT', 'status': 'claim_out_of_domain'}],
                   'verification': {'evidence': []},
                   'report_summary': {'schema_version': 5, 'verdict': 'false',
                                      'summary': 'The retained source assessment.'}},
    })
    monkeypatch.setattr(app.extensions['analyzer'], 'text', lambda text: (.27, {
        'name': 'BERT', 'status': 'ready', 'version': 'fixture', 'experimental_input': True}))
    response = client.get('/api/reports/' + report['id'])
    assert response.status_code == 200
    result = response.json['result']
    assert result['fake_percentage'] == 27.
    assert result['confidence'] == 73.
    assert result['text_metrics']['word_count'] == 5
    assert result['report_summary']['verdict'] == 'false'
    assert result['report_summary']['summary'] == 'The retained source assessment.'
    assert result['text_model_assessment']['experimental_input']
