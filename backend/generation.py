"""Separate, local AI-image scoring; never used as a factual-truth classifier."""
import json
from pathlib import Path
import threading

from PIL import ImageOps


def media_origin_assessment(score, assessment, visual_type='unclear'):
    """An unvalidated classifier score cannot establish an image's origin."""
    assessed = isinstance(score, (int, float)) and not isinstance(score, bool)
    validated = assessment.get('locally_validated') is True
    verdict = ('ai_generated' if score > 50 else 'likely_real' if score < 50 else 'inconclusive') \
        if assessed and validated else 'inconclusive' if assessed else 'classification_unavailable'
    reason = ('The detector has not been locally validated for origin classification. '
              'Its experimental score cannot reliably distinguish a photograph from an AI-generated image.'
              if assessed and not validated else
              'The model score is not proof of origin.' if assessed else
              'The AI-origin detector did not return a score.')
    return {'policy_version': 1, 'status': assessment.get('status', 'not_assessed'),
            'verdict': verdict, 'score': score, 'visual_type': visual_type, 'reason': reason}


class GenerationDetector:
    def __init__(self, directory):
        self.directory = Path(directory) if directory else None
        self.lock = threading.Lock()
        self.model = None
        self.processor = None
        self.card = None
        self.ai_index = None

    def ready(self):
        return bool(self.directory and all((self.directory / name).is_file() for name in (
            'config.json', 'preprocessor_config.json', 'model.safetensors', 'model-card.json')))

    def _load(self):
        if self.model is not None:
            return
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        directory = str(self.directory)
        card = json.loads((self.directory / 'model-card.json').read_text(encoding='utf8'))
        processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False, use_fast=False)
        model = AutoModelForImageClassification.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()
        labels = {int(index): label for index, label in model.config.id2label.items()}
        matches = [index for index, label in labels.items() if label == card.get('ai_label')]
        if len(matches) != 1 or len(labels) != 2:
            raise ValueError('The AI-image model must have two classes and an explicit, matching ai_label.')
        self.processor, self.model = processor, model
        self.card, self.ai_index = card, matches[0]

    def predict(self, images, batch_size=2):
        """Return uncalibrated AI-class scores, in input order, in bounded batches."""
        import torch

        with self.lock:
            self._load()
            scores = []
            for offset in range(0, len(images), batch_size):
                batch = [ImageOps.exif_transpose(image).convert('RGB') for image in images[offset:offset + batch_size]]
                inputs = self.processor(images=batch, return_tensors='pt')
                with torch.inference_mode():
                    probabilities = self.model(**inputs).logits.softmax(dim=-1)
                scores.extend(probabilities[:, self.ai_index].tolist())
            return scores, dict(self.card)
