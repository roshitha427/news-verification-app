"""Attach the reproducible LIAR diagnostic to a local text card; weights stay unchanged."""
import json
import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    audit = json.loads((root / 'data/text-liar/audit.json').read_text(encoding='utf8'))
    path = root / 'models/text/model-card.json'
    card = json.loads(path.read_text(encoding='utf8'))
    diagnostic = audit['local_bert_external_diagnostic']
    if diagnostic['model_version'] != card['version']:
        raise ValueError('The external diagnostic describes a different text model version.')
    entry = dict(diagnostic['strict_binary'], source_url='https://aclanthology.org/P17-2067/',
        scope='Short English political claims; strict true versus false/pants-fire only.',
        source_archive_sha256=audit['liar']['archive_sha256'],
        audit_sha256=hashlib.sha256((root / 'data/text-liar/audit.json').read_bytes()).hexdigest(),
        fitting='No LIAR fitting; original ISOT threshold and calibration unchanged.')
    card.setdefault('external_evaluations', {})['liar_strict_binary'] = entry
    path.write_text(json.dumps(card, indent=2), encoding='utf8')
    print(json.dumps({'model_version': card['version'], 'external_benchmark': 'LIAR strict binary',
        'accuracy': entry['accuracy'], 'test_count': entry['test_count'], 'weights_changed': False}))


if __name__ == '__main__':
    main()
