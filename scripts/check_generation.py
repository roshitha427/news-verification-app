"""Local inference smoke check. This does not measure accuracy."""
import json
from pathlib import Path
import time

from PIL import Image

from backend.generation import GenerationDetector


def main():
    root = Path(__file__).resolve().parents[1]
    detector = GenerationDetector(root / 'models' / 'ai-image')
    if not detector.ready():
        raise SystemExit('Run python -m scripts.setup_generation first.')
    images = []
    for path in (root / 'data' / 'fake.jpg', root / 'data' / 'real.jpg'):
        if path.is_file():
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
    if not images:
        raise SystemExit('Add data/fake.jpg or data/real.jpg for an inference smoke check.')
    started = time.monotonic()
    import torch
    torch.set_num_threads(3)
    scores, card = detector.predict(images)
    assert len(scores) == len(images) and all(0 <= score <= 1 for score in scores)
    # Warm inference exercises the cached model without downloading or rebuilding it.
    repeat, _ = detector.predict(images[:1])
    assert abs(scores[0] - repeat[0]) < 1e-5
    print(json.dumps({'ready': True, 'images': len(images), 'ai_model_scores': scores,
                      'seconds': round(time.monotonic() - started, 2), 'revision': card['version'],
                      'accuracy_measured': False, 'status': 'experimental'}))


if __name__ == '__main__':
    main()
