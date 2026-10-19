"""CPU forensic experiment: validation selection, independent calibration, one test.

The existing diagnostic test has been reported for prior experiments. This run
does not turn it into independent external evidence, even when scores improve.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import threading
import time
import zipfile
import joblib
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from ml.forensic_features import FEATURE_VERSION, image_features
from ml.train_image import metrics


def main(args):
    started = time.time()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf8'))
    with open(args.archive, 'rb') as archive_file:
        digest = hashlib.file_digest(archive_file, 'sha256').hexdigest()
    if digest != manifest['archive_sha256']:
        raise ValueError('Archive fingerprint differs from audited manifest.')
    selected = {}
    for split, cap in [('train', args.train), ('validation', args.validation), ('test', 0)]:
        entries = []
        for label in ['real', 'fake']:
            pool = [entry for entry in manifest['entries'] if entry['split'] == split and entry['label'] == label]
            random.Random(91).shuffle(pool)
            entries.extend(pool[:cap // 2] if cap else pool)
        random.Random(91).shuffle(entries)
        selected[split] = entries
    print('Split sizes: ' + str({split: len(entries) for split, entries in selected.items()}), flush=True)
    local = threading.local()
    def extract(entry):
        if not hasattr(local, 'archive'):
            local.archive = zipfile.ZipFile(args.archive)
        with Image.open(BytesIO(local.archive.read(entry['member']))) as image:
            return image_features(image)
    def features(split):
        fingerprint = hashlib.sha256((FEATURE_VERSION + json.dumps(selected[split], sort_keys=True)).encode()).hexdigest()
        cache = output / (split + '-features-' + fingerprint[:12] + '.npz')
        if cache.exists():
            loaded = np.load(cache)
            return loaded['x'], loaded['y']
        chunks = []
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            for index, vector in enumerate(executor.map(extract, selected[split])):
                chunks.append(vector)
                if (index + 1) % 2000 == 0:
                    print(f'{split}: {index + 1}/{len(selected[split])} | elapsed {time.time() - started:.0f}s', flush=True)
        x = np.stack(chunks)
        y = np.array([entry['label'] == 'fake' for entry in selected[split]], dtype=np.int64)
        np.savez_compressed(cache, x=x, y=y)
        return x, y
    x_train, y_train = features('train')
    x_validation, y_validation = features('validation')
    tune, calibration = np.arange(0, len(y_validation), 2), np.arange(1, len(y_validation), 2)
    candidates = [(f'logistic-C{c}', make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1500, random_state=91)))
                  for c in [.01, .1, 1]]
    candidates.extend([(f'histgradient-leaves{leaves}', HistGradientBoostingClassifier(max_leaf_nodes=leaves,
        max_iter=180, learning_rate=.08, l2_regularization=5, random_state=91, early_stopping=True)) for leaves in [15, 31]])
    candidates.append(('extra-trees', ExtraTreesClassifier(n_estimators=220, max_features=.5, min_samples_leaf=2,
        n_jobs=args.threads, random_state=91)))
    best, best_name, best_score = None, None, -1
    history = []
    with threadpool_limits(limits=args.threads):
        for name, classifier in candidates:
            classifier.fit(x_train, y_train)
            score = float(balanced_accuracy_score(y_validation[tune], classifier.predict(x_validation[tune])))
            history.append(dict(candidate=name, validation_balanced_accuracy=score))
            print(json.dumps(history[-1]) + f' | elapsed {time.time() - started:.0f}s', flush=True)
            if score > best_score:
                best, best_name, best_score = classifier, name, score
        probabilities = best.predict_proba(x_validation[calibration])[:, 1]
        logits = np.log(np.clip(probabilities, 1e-7, 1 - 1e-7) / np.clip(1 - probabilities, 1e-7, 1))
        sigmoid = lambda values: 1 / (1 + np.exp(-np.clip(values, -30, 30)))
        temperature = float(minimize_scalar(lambda value: log_loss(y_validation[calibration], sigmoid(logits / value), labels=[0, 1]),
            bounds=(.1, 10), method='bounded').x)
        # Fit and calibration are finished before any test feature extraction.
        print(f'Selected {best_name}, temperature {temperature:.4f}; now evaluating diagnostic test once.', flush=True)
        x_test, y_test = features('test')
        probabilities = best.predict_proba(x_test)[:, 1]
        test_logits = np.log(np.clip(probabilities, 1e-7, 1 - 1e-7) / np.clip(1 - probabilities, 1e-7, 1))
        evaluation = metrics(y_test, sigmoid(test_logits / temperature))
    card = dict(name='Native-resolution residual and frequency classifier', version='1.0-forensic-' + manifest['version'],
        architecture='Deterministic color, residual-cooccurrence, block-DCT and Fourier features + ' + best_name,
        feature_version=FEATURE_VERSION, feature_count=int(x_train.shape[1]), dataset_version=manifest['version'], dataset_group=manifest['group'],
        scope='Dataset-specific visual fake/real folder-label classification. Does not establish truth or AI generation.',
        evaluation=evaluation, split_counts={split: len(entries) for split, entries in selected.items()}, seed=91,
        selected_validation_balanced_accuracy=best_score, training_history=history, temperature=temperature,
        target_accuracy=.85, target_met=evaluation['accuracy'] >= .85, status='experimental', input_size=256,
        runtime_seconds=round(time.time() - started), limitations=manifest['limitations'] + [
            'This is a previously reported diagnostic test; independent external testing is still required.',
            'A 20-image training sample consisted of portraits/face crops; general news-photo performance is unvalidated.',
            'Native-resolution forensic statistics can learn compression or source artifacts and do not establish causal manipulation.',
            'Image transforms and video re-encoding may materially alter model reliability.'])
    joblib.dump(dict(kind='forensic', classifier=best, feature_version=FEATURE_VERSION, temperature=temperature, card=card), output / 'model.joblib')
    (output / 'model-card.json').write_text(json.dumps(card, indent=2), encoding='utf8')
    print(json.dumps(card, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--manifest', default='data/image-manifest.json')
    parser.add_argument('--output', default='models/image-forensic')
    parser.add_argument('--train', type=int, default=16000)
    parser.add_argument('--validation', type=int, default=4000)
    parser.add_argument('--threads', type=int, default=3)
    main(parser.parse_args())
