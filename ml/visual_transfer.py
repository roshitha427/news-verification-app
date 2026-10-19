"""Local SigLIP representation with a dataset-trained, calibrated visual head.

The pretrained AI/human classifier head is discarded. This classifier estimates
the archive's fake/real labels and does not establish factual truth or authorship.
"""
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np


def load_extractor(backbone_path, precision='int8', threads=3):
    """Load administrator-managed local weights without any network fallback."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    torch.set_num_threads(threads)
    processor = AutoImageProcessor.from_pretrained(str(backbone_path), local_files_only=True, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(str(backbone_path), local_files_only=True).eval()
    vision = model.vision_model
    if precision == 'int8':
        vision = torch.ao.quantization.quantize_dynamic(vision, {torch.nn.Linear}, dtype=torch.qint8)
    elif precision != 'float32':
        raise ValueError('Supported precisions: int8 or float32.')
    return processor, vision.eval()


def extract_vectors(processor, vision, images):
    import torch
    from PIL import ImageOps

    images = [ImageOps.exif_transpose(image).convert('RGB') for image in images]
    pixels = processor(images=images, return_tensors='pt')['pixel_values']
    with torch.inference_mode():
        # The source SigLIP classification implementation uses mean patch pooling.
        vectors = vision(pixels).last_hidden_state.mean(dim=1)
    return vectors.float().cpu().numpy()


def extract_independent_vectors(processor, vision, images, workers=4):
    """Parallel independent batch-one calls; never mix activation ranges."""
    if len(images) == 1 or workers == 1:
        return np.concatenate([extract_vectors(processor, vision, [image]) for image in images])
    # Load/copy PIL inputs before workers start, so repeated references to a
    # lazily opened image never share a mutable decoder/file handle.
    images = [image.copy() for image in images]
    configuration = processor.to_dict()
    local = threading.local()

    def worker(image):
        if not hasattr(local, 'processor'):
            local.processor = processor.__class__.from_dict(configuration)
        return extract_vectors(local.processor, vision, [image])

    with ThreadPoolExecutor(max_workers=min(workers, len(images))) as executor:
        return np.concatenate(list(executor.map(worker, images)))


class SiglipTransferModel:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.bundle = None
        self.processor = None
        self.vision = None

    def ready(self):
        return self.path.is_file()

    def _load(self):
        if self.bundle is not None:
            return
        import joblib

        # Only local, administrator-produced artifacts may be loaded with joblib.
        bundle = joblib.load(self.path)
        if bundle.get('kind') != 'siglip-transfer':
            raise ValueError('Artifact is not a trained SigLIP transfer model.')
        backbone = (self.path.parent / bundle['backbone_relative_path']).resolve()
        for name, expected in bundle['source_hashes'].items():
            with (backbone / name).open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                    raise ValueError('Local representation weights/configuration differ from the trained artifact.')
        processor, vision = load_extractor(backbone, bundle['precision'], threads=1)
        self.processor, self.vision, self.bundle = processor, vision, bundle

    def predict(self, images):
        with self.lock:
            self._load()
            # Dynamic INT8 activation ranges depend on batch composition. Serving
            # each image separately matches the validation/calibration/test path
            # and makes an image's result independent of other video frames.
            vectors = extract_independent_vectors(self.processor, self.vision, images,
                workers=self.bundle.get('inference_workers', 4))
            logits = self.bundle['classifier'].decision_function(vectors)
            probability = 1 / (1 + np.exp(-np.clip(logits / self.bundle['temperature'], -30, 30)))
            return probability.tolist(), self.bundle['card']
