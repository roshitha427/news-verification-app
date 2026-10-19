"""Deterministic visual forensic features; shared by training and inference.

These statistics inspect texture, spatial residuals and frequency energy. They
cannot establish a news claim's truth or identify a particular image generator.
No filenames, JPEG metadata, labels or publisher information are features.
"""
import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn
import cv2

FEATURE_VERSION = 'residual-spectrum-v1'
_YY, _XX = np.indices((256, 256))
_RADII = np.floor(np.hypot(_XX - 128, _YY - 128)).astype(np.int32)
_RADIUS_COUNTS = np.maximum(np.bincount(_RADII.ravel(), minlength=182), 1)


def image_features(image):
    image = ImageOps.exif_transpose(image).convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32) / 255.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    vectors = []
    for channel in range(3):
        values = rgb[:, :, channel]
        vectors.extend([values.mean(), values.std()])
        vectors.extend(np.quantile(values, [.01, .05, .1, .25, .5, .75, .9, .95, .99]))
    # Distribution of color and saturation, independent of the subject/location.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    for channel, high in [(0, 360), (1, 1), (2, 1)]:
        vectors.extend(np.histogram(hsv[:, :, channel], bins=16, range=(0, high))[0] / gray.size)
    # Spectrum statistics retain information lost by the 128-pixel CNN resize.
    for channel in [gray, rgb[:, :, 0] - rgb[:, :, 1], rgb[:, :, 1] - rgb[:, :, 2]]:
        spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(channel))) ** 2)
        # Exclude DC; it otherwise mostly describes brightness.
        spectrum[128, 128] = 0.
        vectors.extend(spectrum.reshape(16, 16, 16, 16).mean(axis=(1, 3)).ravel())
        sums = np.bincount(_RADII.ravel(), weights=spectrum.ravel(), minlength=182)
        profile = sums / _RADIUS_COUNTS
        vectors.extend(part.mean() for part in np.array_split(profile[1:177], 32))
    # JPEG-aligned block frequency distributions, without using metadata.
    blocks = gray.reshape(32, 8, 32, 8).transpose(0, 2, 1, 3)
    coefficients = np.abs(dctn(blocks, axes=(-2, -1), norm='ortho'))
    vectors.extend(np.log1p(coefficients.mean(axis=(0, 1)) * 255).ravel())
    vectors.extend(np.log1p(coefficients.std(axis=(0, 1)) * 255).ravel())
    # Neighbor co-occurrences of high-pass residuals: bounded rich-model subset.
    kernels = [np.array([[1, -2, 1]], dtype=np.float32),
               np.array([[1], [-2], [1]], dtype=np.float32),
               np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)]
    for kernel in kernels:
        residual = cv2.filter2D(gray, -1, kernel)[2:-2, 2:-2]
        vectors.extend([residual.std(), np.abs(residual).mean(), np.quantile(np.abs(residual), .95)])
        quantized = np.clip(np.rint(residual * 255 / 3), -3, 3).astype(np.int32) + 3
        for axis in [0, 1]:
            left = quantized[:-1, :] if axis == 0 else quantized[:, :-1]
            right = quantized[1:, :] if axis == 0 else quantized[:, 1:]
            histogram = np.bincount((left * 7 + right).ravel(), minlength=49)
            vectors.extend(histogram / histogram.sum())
    # Multi-scale directional Haar energies.
    current = gray
    for _ in range(4):
        a, b, c, d = current[::2, ::2], current[::2, 1::2], current[1::2, ::2], current[1::2, 1::2]
        for detail in [a - b + c - d, a + b - c - d, a - b - c + d]:
            vectors.extend([np.abs(detail).mean(), detail.std(), np.quantile(np.abs(detail), .95)])
        current = (a + b + c + d) / 4
    result = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Visual features contain non-finite values.')
    return result


class ForensicModel:
    """Load only trusted locally produced joblib artifacts."""
    def __init__(self, path):
        self.path = path
        self.bundle = None

    def ready(self):
        from pathlib import Path
        return Path(self.path).is_file()

    def predict(self, images):
        import joblib
        if self.bundle is None:
            self.bundle = joblib.load(self.path)
        if self.bundle['feature_version'] != FEATURE_VERSION:
            raise ValueError('Unsupported forensic feature version.')
        features = np.stack([image_features(image) for image in images])
        probabilities = self.bundle['classifier'].predict_proba(features)[:, 1]
        logits = np.log(np.clip(probabilities, 1e-7, 1 - 1e-7) / np.clip(1 - probabilities, 1e-7, 1))
        logits /= self.bundle['temperature']
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        return probabilities.tolist(), self.bundle['card']
