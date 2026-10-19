"""Train a local SigLIP embedding head with split isolation and resumable caches.

The archive test has been examined by earlier experiments; results are diagnostic,
not an independent confirmation of generalization or a universal accuracy claim.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import random
import threading
import time
import zipfile

import joblib
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from ml.train_image import metrics
from ml.visual_transfer import extract_independent_vectors, extract_vectors, load_extractor


def select_entries(manifest, split, cap):
    entries = []
    for label in ('real', 'fake'):
        pool = [entry for entry in manifest['entries'] if entry['split'] == split and entry['label'] == label]
        random.Random(42).shuffle(pool)
        entries.extend(pool[:cap // 2] if cap else pool)
    random.Random(42).shuffle(entries)
    if len(entries) < 20:
        raise ValueError(f'Too few examples for {split}.')
    return entries


def main(args):
    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf8'))
    with open(args.archive, 'rb') as handle:
        archive_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    if archive_sha != manifest['archive_sha256']:
        raise ValueError('Archive differs from the audited manifest.')
    selected = {split: select_entries(manifest, split, cap) for split, cap in
                [('train', args.train), ('validation', args.validation), ('test', args.test)]}
    if min(len(value) for value in selected.values()) < 100:
        raise ValueError('At least 100 examples per split are required for this experiment.')
    print('Precommitted splits: ' + str({key: len(value) for key, value in selected.items()}), flush=True)
    backbone = Path(args.backbone).resolve()
    # A snapshot change must never silently reuse a different model's embeddings.
    snapshot_hashes = {}
    for name in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
        with (backbone / name).open('rb') as handle:
            snapshot_hashes[name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    fingerprint = hashlib.sha256(json.dumps(dict(entries=selected, source=snapshot_hashes,
        precision=args.precision, pooling='last_hidden_state_mean_v1'), sort_keys=True).encode()).hexdigest()
    # The original cache was computed with 16-image batches. Dynamic INT8 input
    # scaling may depend on batch composition, so another batch size needs its own cache.
    if args.batch_size != 16:
        fingerprint = hashlib.sha256((fingerprint + f'-batch{args.batch_size}').encode()).hexdigest()
    if args.train_threads != 3:
        fingerprint = hashlib.sha256((fingerprint + f'-trainthreads{args.train_threads}').encode()).hexdigest()
    split_batch_sizes = dict(train=args.batch_size, validation=1, test=1, serving=1)
    split_threads = dict(train=args.train_threads, validation=args.threads, test=args.threads, serving=args.threads)
    split_feature_fingerprints = {split: hashlib.sha256(json.dumps(dict(entries=entries,
        source=snapshot_hashes, precision=args.precision, pooling='last_hidden_state_mean_v1',
        extraction_batch_size=split_batch_sizes[split], torch_threads=split_threads[split]), sort_keys=True).encode()).hexdigest()
        for split, entries in selected.items()}
    experiment_fingerprint = hashlib.sha256((fingerprint + json.dumps(split_feature_fingerprints, sort_keys=True)
        + '-logistic6-rbf2-v1').encode()).hexdigest()
    cache = output / 'feature-cache' / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=True)
    processor, vision = load_extractor(backbone, args.precision, args.train_threads)
    run_plan = dict(archive_sha256=archive_sha, source_hashes=snapshot_hashes,
                   selection_fingerprint=fingerprint, precision=args.precision, seed=42,
                   feature_batch_size=args.batch_size, runtime_threads=args.threads,
                   training_extraction_threads=args.train_threads, evaluation_workers=args.workers,
                   split_threads=split_threads,
                   split_batch_sizes=split_batch_sizes, split_feature_fingerprints=split_feature_fingerprints,
                   experiment_fingerprint=experiment_fingerprint,
                   split_counts={key: len(value) for key, value in selected.items()},
                   candidate_grid=dict(logistic_C=[.0001, .001, .01, .1, 1., 10.], rbf_svc_C=[1., 10.], rbf_gamma='scale'),
                   grid_note='RBF candidates added after a training-only representation diagnostic, before validation or test extraction/scoring.',
                   test_usage='Repeatedly inspected diagnostic split; no independent accuracy claim.')
    (output / 'run-plan.json').write_text(json.dumps(run_plan, indent=2), encoding='utf8')

    def features(split):
        vectors, labels = [], []
        batch_size = split_batch_sizes[split]
        local = threading.local()
        worker_archives = []
        processor_configuration = processor.to_dict()

        def extract_one(entry):
            if not hasattr(local, 'archive'):
                local.archive = zipfile.ZipFile(args.archive)
                worker_archives.append(local.archive)
                local.processor = processor.__class__.from_dict(processor_configuration)
            with Image.open(BytesIO(local.archive.read(entry['member']))) as image:
                return extract_vectors(local.processor, vision, [image.convert('RGB').copy()])

        executor = ThreadPoolExecutor(max_workers=args.workers) if split != 'train' else None
        with zipfile.ZipFile(args.archive) as archive:
            try:
                for start in range(0, len(selected[split]), args.cache_chunk):
                    entries = selected[split][start:start + args.cache_chunk]
                    # Preserve the verified batch16 training cache; evaluation uses
                    # its own explicit fingerprint and single-image extraction.
                    prefix = split if split == 'train' else split + '-' + split_feature_fingerprints[split][:12]
                    cached = cache / f'{prefix}-{start:06}.npz'
                    if cached.is_file():
                        with np.load(cached) as stored:
                            x, y = stored['x'], stored['y']
                        if len(x) != len(entries):
                            raise ValueError('Feature-cache size does not match the run plan.')
                    else:
                        chunk_vectors = []
                        if executor is not None:
                            # Each worker has its own decoder, processor and input
                            # tensor. Read-only vision weights are shared safely.
                            chunk_vectors = list(executor.map(extract_one, entries))
                        else:
                            for offset in range(0, len(entries), batch_size):
                                batch = entries[offset:offset + batch_size]
                                images = []
                                for entry in batch:
                                    with Image.open(BytesIO(archive.read(entry['member']))) as image:
                                        images.append(image.convert('RGB').copy())
                                chunk_vectors.append(extract_vectors(processor, vision, images))
                        x = np.concatenate(chunk_vectors)
                        y = np.asarray([int(entry['label'] == 'fake') for entry in entries])
                        temporary = cached.with_suffix('.tmp')
                        with temporary.open('wb') as handle:
                            np.savez_compressed(handle, x=x, y=y)
                        temporary.replace(cached)
                    vectors.append(x)
                    labels.append(y)
                    print(f'{split}: {start + len(entries)}/{len(selected[split])} | '
                          f'{time.monotonic() - started:.0f}s', flush=True)
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)
                for handle in worker_archives:
                    handle.close()
        return np.concatenate(vectors), np.concatenate(labels)

    x_train, y_train = features('train')
    import torch
    torch.set_num_threads(args.threads)
    if args.benchmark_workers:
        benchmark_images = []
        with zipfile.ZipFile(args.archive) as archive:
            for entry in selected['train'][:24]:
                with Image.open(BytesIO(archive.read(entry['member']))) as image:
                    benchmark_images.append(image.convert('RGB').copy())
        baseline_started = time.monotonic()
        reference = extract_independent_vectors(processor, vision, benchmark_images, workers=1)
        benchmark = [dict(workers=1, seconds=time.monotonic() - baseline_started, exactly_equal=True)]
        for count in (4, 6, 8):
            benchmark_started = time.monotonic()
            candidate_vectors = extract_independent_vectors(processor, vision, benchmark_images, workers=count)
            record = dict(workers=count, seconds=time.monotonic() - benchmark_started,
                          exactly_equal=bool(np.array_equal(reference, candidate_vectors)),
                          max_absolute_difference=float(abs(reference - candidate_vectors).max()))
            benchmark.append(record)
            print('Worker benchmark: ' + json.dumps(record), flush=True)
        valid = [record for record in benchmark if record['exactly_equal']]
        args.workers = min(valid, key=lambda record: record['seconds'])['workers']
        (output / 'worker-benchmark.json').write_text(json.dumps(benchmark, indent=2), encoding='utf8')
        run_plan['evaluation_workers'] = args.workers
        run_plan['worker_benchmark'] = benchmark
        (output / 'run-plan.json').write_text(json.dumps(run_plan, indent=2), encoding='utf8')
        print(f'Selected {args.workers} independent batch-one workers.', flush=True)
    x_validation, y_validation = features('validation')
    tune, calibrate = np.arange(0, len(y_validation), 2), np.arange(1, len(y_validation), 2)
    best, best_score, selected_c, selected_head = None, -1., None, None
    candidates = []
    grid = [('logistic', c, LogisticRegression(C=c, max_iter=3000, random_state=42))
            for c in (.0001, .001, .01, .1, 1., 10.)]
    grid.extend(('rbf_svc', c, SVC(C=c, gamma='scale', probability=False, random_state=42)) for c in (1., 10.))
    with threadpool_limits(limits=args.threads):
        for head, c, estimator in grid:
            classifier = make_pipeline(StandardScaler(), estimator)
            classifier.fit(x_train, y_train)
            score = balanced_accuracy_score(y_validation[tune], classifier.predict(x_validation[tune]))
            candidates.append(dict(head=head, C=c, validation_selection_balanced_accuracy=float(score)))
            print(f'{head} C={c}: validation selection balanced accuracy={score:.4f}', flush=True)
            if score > best_score:
                best, best_score, selected_c, selected_head = classifier, float(score), c, head
    sigmoid = lambda values: 1 / (1 + np.exp(-np.clip(values, -30, 30)))
    logits = best.decision_function(x_validation[calibrate])
    calibrated = minimize_scalar(lambda temperature: log_loss(y_validation[calibrate],
        sigmoid(logits / temperature), labels=[0, 1]), bounds=(.1, 10.), method='bounded')
    temperature = float(calibrated.x)
    selection = dict(candidates=candidates, selected_head=selected_head, selected_C=selected_c, temperature=temperature,
                     selection_fingerprint=fingerprint, test_not_used_for_selection=True,
                     experiment_fingerprint=experiment_fingerprint, split_batch_sizes=split_batch_sizes,
                     validation_calibration_metrics=metrics(y_validation[calibrate], sigmoid(logits / temperature)))
    (output / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf8')
    # Freeze and persist the choice before accessing the diagnostic test features.
    joblib.dump(dict(classifier=best, temperature=temperature, kind='siglip-transfer'), output / 'selected-head.joblib')
    print('Head selection and calibration frozen; starting diagnostic test.', flush=True)
    x_test, y_test = features('test')
    evaluation = metrics(y_test, sigmoid(best.decision_function(x_test) / temperature))
    matrix = np.asarray(evaluation['confusion_matrix'])
    evaluation['per_class_recall'] = dict(real=float(matrix[0, 0] / matrix[0].sum()),
                                        fake=float(matrix[1, 1] / matrix[1].sum()))
    limitations = list(manifest.get('limitations', [])) + [
        'The original test split was repeatedly inspected by earlier experiments. This score is diagnostic, not an independent confirmation.',
        'The supplied archive consists of face crops/portraits; arbitrary news photographs and non-face video are outside this evaluation.',
        'The pretrained SigLIP AI-image snapshot may overlap public generators or image collections in the archive. Independence is not established.',
        'Unknown subject, generator and source provenance prevents a person/generator-independent split audit.',
        'A fake/real archive label does not identify AI authorship, temporal deepfakes, audio manipulation or factual truth.',
        'Only the classification head is trained locally; the pretrained representation is frozen and dynamically quantized for CPU inference.',
        'Training embeddings use batches of16, with dynamic INT8 quantization acting as representation noise. All validation, calibration, test and serving embeddings are extracted one image at a time for consistent activation ranges.'
    ]
    card = dict(name='Local SigLIP representation + calibrated fake/real head',
        version='1.0-' + experiment_fingerprint[:12], architecture='Frozen SigLIP ViT mean patch embeddings + StandardScaler + ' + selected_head,
        source_repository='Ateeqq/ai-vs-human-image-detector', source_revision='60e82406916921b823616bee33397baab38af3f0',
        source_license='Apache-2.0', precision=args.precision, pooling='last_hidden_state_mean', input_size=224,
        feature_batch_size=args.batch_size, runtime_threads=args.threads,
        training_extraction_threads=args.train_threads, evaluation_workers=args.workers,
        split_threads=split_threads,
        split_batch_sizes=split_batch_sizes, split_feature_fingerprints=split_feature_fingerprints,
        experiment_fingerprint=experiment_fingerprint, source_hashes=snapshot_hashes,
        dataset_version=manifest['version'], dataset_group=manifest['group'], dataset_archive_sha256=archive_sha,
        split_counts=run_plan['split_counts'], seed=42, selected_head=selected_head, selected_C=selected_c,
        validation_selection_balanced_accuracy=best_score, temperature=temperature,
        validation_calibration=selection['validation_calibration_metrics'], evaluation=evaluation,
        target_accuracy=.90, target_met=evaluation['accuracy'] >= .90, independent_validation=False,
        status='experimental', scope='Dataset-specific portrait fake/real classification; not factual truth or general AI authorship.',
        limitations=limitations, runtime_seconds=round(time.monotonic() - started))
    relative = os.path.relpath(backbone, output)
    joblib.dump(dict(kind='siglip-transfer', classifier=best, temperature=temperature,
        backbone_relative_path=relative, precision=args.precision, source_hashes=snapshot_hashes,
        inference_workers=args.workers, card=card), output / 'model.joblib')
    (output / 'model-card.json').write_text(json.dumps(card, indent=2), encoding='utf8')
    print(json.dumps(card, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--manifest', default='data/image-manifest.json')
    parser.add_argument('--backbone', default='models/ai-image')
    parser.add_argument('--output', default='models/image-transfer')
    parser.add_argument('--train', type=int, default=6000)
    parser.add_argument('--validation', type=int, default=2000)
    parser.add_argument('--test', type=int, default=0, help='0 evaluates all available diagnostic test examples.')
    parser.add_argument('--threads', type=int, default=1, help='Threads per independent validation/test inference.')
    parser.add_argument('--train-threads', type=int, default=3)
    parser.add_argument('--workers', type=int, default=4, help='Independent batch-one evaluation workers.')
    parser.add_argument('--benchmark-workers', action='store_true', help='Benchmark shared-model workers1/4/6/8; use the fastest exactly equal variant.')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--cache-chunk', type=int, default=512)
    parser.add_argument('--precision', choices=('int8', 'float32'), default='int8')
    main(parser.parse_args())
