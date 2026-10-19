"""Audit user-supplied news archives and evaluate a local classifier on LIAR.

Archive contents are data, never executable instructions. Only explicitly named
members are read; no archive paths are extracted. LIAR's original six truth labels
are retained. The two-class article model cannot supply six-class accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

from ml.text_cleaning import clean_text


LABELS = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
COLUMNS = ['id', 'label', 'text', 'subject', 'speaker', 'job', 'state', 'party',
           'barely_true_count', 'false_count', 'half_true_count',
           'mostly_true_count', 'pants_fire_count', 'context']
STRICT_MAPPING = {'true': 0, 'false': 1, 'pants-fire': 1}
DIRECTIONAL_MAPPING = {**STRICT_MAPPING, 'mostly-true': 0}


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def member_digests(path):
    with zipfile.ZipFile(path) as archive:
        result = {}
        for name in ['Fake.csv', 'True.csv']:
            info = archive.getinfo(name)
            if info.file_size > 250_000_000:
                raise ValueError('Unexpected news CSV size.')
            with archive.open(info) as handle:
                result[name] = hashlib.file_digest(handle, 'sha256').hexdigest()
        return result


def normalized(text):
    return ' '.join(re.findall(r'\w+', clean_text(text).casefold()))


def read_liar(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for split, member in [('train', 'train.tsv'), ('validation', 'valid.tsv'), ('test', 'test.tsv')]:
            info = archive.getinfo(member)
            if info.file_size > 20_000_000:
                raise ValueError('Unexpected LIAR member size.')
            with archive.open(info) as handle:
                frame = pd.read_csv(handle, sep='\t', header=None, names=COLUMNS, keep_default_na=False)
            if frame.shape[1] != 14 or not set(frame['label']).issubset(LABELS):
                raise ValueError('Unexpected LIAR columns or labels.')
            if frame['id'].duplicated().any() or not frame['text'].str.strip().all():
                raise ValueError('LIAR contains duplicate IDs or blank statements.')
            frame['key'] = frame['text'].map(normalized)
            result[split] = frame
    return result


def infer(frame, model_path, batch_size, threads):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(threads)
    card = json.loads(Path(model_path, 'model-card.json').read_text(encoding='utf8'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).eval()
    id2label = {int(key): value.lower() for key, value in model.config.id2label.items()}
    fake_index = next((index for index, name in id2label.items() if name == 'fake'), None)
    if fake_index is None or len(id2label) != 2:
        raise ValueError('Expected a local two-class fake/real model.')
    values = []
    with torch.inference_mode():
        for offset in range(0, len(frame), batch_size):
            texts = frame['text'].iloc[offset:offset + batch_size].map(clean_text).tolist()
            tokens = tokenizer(texts, padding=True, truncation=True,
                               max_length=card.get('max_length', 192), return_tensors='pt')
            logits = model(**tokens).logits / card.get('temperature', 1)
            values.extend(logits.softmax(-1)[:, fake_index].tolist())
    return np.array(values), card


def binary_report(frame, probabilities, mapping):
    from ml.train_image import metrics
    included = frame['label'].isin(mapping)
    y = frame.loc[included, 'label'].map(mapping).to_numpy()
    report = metrics(y, probabilities[included.to_numpy()])
    report.update({'mapping': mapping, 'excluded_labels': [name for name in LABELS if name not in mapping],
                   'excluded_count': int((~included).sum()),
                   'majority_class_accuracy': float(max((y == 0).mean(), (y == 1).mean()))})
    return report


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    incoming = member_digests(args.news_archive)
    existing = member_digests(args.existing_news_archive)
    news = {'supplied_archive_sha256': sha256(args.news_archive),
            'previous_archive_sha256': sha256(args.existing_news_archive),
            'supplied_csv_sha256': incoming, 'previous_csv_sha256': existing,
            'content_identical': incoming == existing,
            'decision': 'No retraining: identical existing CSV contents.' if incoming == existing else 'Manual provenance audit required.'}
    splits = read_liar(args.liar_archive)
    overlap = {}
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        overlap[left + '_' + right] = {'ids': len(set(splits[left]['id']) & set(splits[right]['id'])),
                                     'normalized_statements': len(set(splits[left]['key']) & set(splits[right]['key']))}
    # Retain the official benchmark untouched, but separately prepare disjoint
    # train/validation/test inputs with test > validation > train precedence.
    seen = set()
    clean_counts = {}
    for split in ['test', 'validation', 'train']:
        frame = splits[split]
        clean = frame[~frame['key'].isin(seen)].drop_duplicates('key')
        seen.update(frame['key'])
        clean[['id', 'label', 'text']].to_csv(output / (split + '-six-label.csv'), index=False)
        strict = clean[clean['label'].isin(STRICT_MAPPING)].copy()
        strict['label'] = strict['label'].map({key: 'fake' if value else 'real' for key, value in STRICT_MAPPING.items()})
        strict[['text', 'label']].to_csv(output / (split + '-strict-binary.csv'), index=False)
        clean_counts[split] = {'all_six': len(clean), 'strict_binary': len(strict)}
    test = splits['test']
    existing_text_keys = set()
    for split in ['train', 'validation', 'test']:
        path = Path(args.existing_text_dir) / (split + '.csv')
        if path.is_file():
            existing_text_keys.update(pd.read_csv(path, usecols=['text'])['text'].map(normalized))
    test_overlap = test['key'].isin(existing_text_keys)
    probabilities, card = infer(test, args.model, args.batch_size, args.threads)
    per_label = {}
    for label in LABELS:
        values = probabilities[(test['label'] == label).to_numpy()]
        per_label[label] = {'count': len(values), 'mean_fake_probability': float(values.mean()),
                            'classified_fake_fraction': float((values >= .5).mean())}
    report = {'news_archive': news,
              'liar': {'archive_sha256': sha256(args.liar_archive),
                       'source': 'https://aclanthology.org/P17-2067/',
                       'license': 'Research purposes only; original publishers retain copyright (archive README).',
                       'languages': ['en'],
                       'original_split_counts': {key: len(value) for key, value in splits.items()},
                       'original_label_counts': {key: value['label'].value_counts().to_dict() for key, value in splits.items()},
                       'within_split_normalized_duplicate_rows': {key: int(value['key'].duplicated().sum()) for key, value in splits.items()},
                       'cross_split_overlap': overlap, 'prepared_disjoint_counts': clean_counts,
                       'prepared_policy': 'Normalized punctuation/case/whitespace deduplication; test > validation > train. Original six labels retained. Strict binary omits all partly true categories.',
                       'prohibited_features': ['Speaker credit history includes current statement and risks label leakage.', 'No metadata, source, label or speaker history fed to text model.'],
                       'evidence_contents': 'Claim statements and metadata only; no evidence sentences/full verdict reports.',
                       'isot_normalized_test_statement_overlap': int(test_overlap.sum())},
              'local_bert_external_diagnostic': {
                  'model_path': str(args.model), 'model_version': card['version'],
                  'model_card_sha256': sha256(Path(args.model) / 'model-card.json'),
                  'training_corpus': 'ISOT historical English news articles; model unchanged.',
                  'threshold': .5, 'calibration': 'Frozen original ISOT-validation temperature; no LIAR fitting.',
                  'strict_binary': binary_report(test, probabilities, STRICT_MAPPING),
                  'directional_binary': binary_report(test, probabilities, DIRECTIONAL_MAPPING),
                  'all_six_label_diagnostics': per_label,
                  'six_class_accuracy': None,
                  'six_class_explanation': 'This binary model cannot predict the original six truth levels.',
                  'limitations': ['LIAR political claims differ from ISOT articles; this measures domain shift.',
                                  'No LIAR test data used for fitting, selection or threshold changes.',
                                  'No claim evidence is supplied to this classifier; it cannot establish current truth.',
                                  'No AI-writing, Hindi or Telugu accuracy is measured.']}}
    (output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    predicted = test[['id', 'label']].copy()
    predicted['fake_probability'] = probabilities
    predicted.to_csv(output / 'isot-model-liar-test-predictions.csv', index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--liar-archive', required=True)
    parser.add_argument('--news-archive', required=True)
    parser.add_argument('--existing-news-archive', default='data/isot.zip')
    parser.add_argument('--existing-text-dir', default='data/text')
    parser.add_argument('--model', default='models/text')
    parser.add_argument('--output', default='data/text-liar')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    main(parser.parse_args())
