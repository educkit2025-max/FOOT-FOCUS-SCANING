"""Image I/O, scaling and camera-intrinsics helpers shared by the agents."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# Laplacian variance measured on a copy whose long side is 640 px.
SHARP_MIN = 25.0
SHARP_OK = 60.0

# f_px ≈ 0.77 × long side for typical phone main cameras (Pixel 3a: 4.44 mm lens on a 5.76 mm sensor).
DEFAULT_FOCAL_FACTOR = 0.77


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    source: str

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


def load_image_bytes(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        arr = np.asarray(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def load_image_path(path: Path | str) -> np.ndarray:
    return load_image_bytes(Path(path).read_bytes())


def downscale(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return image, 1.0
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def intrinsics_for(shape: tuple[int, ...], focal_mm: float | None = None, sensor_long_mm: float | None = None) -> Intrinsics:
    h, w = shape[:2]
    long_side = float(max(h, w))
    if focal_mm and sensor_long_mm and focal_mm > 0 and sensor_long_mm > 0:
        f = float(focal_mm) / float(sensor_long_mm) * long_side
        source = "device_metadata"
    else:
        f = DEFAULT_FOCAL_FACTOR * long_side
        source = "default_phone_main_camera"
    return Intrinsics(fx=f, fy=f, cx=w / 2.0, cy=h / 2.0, source=source)


def sharpness_score(image: np.ndarray) -> float:
    small, _ = downscale(image, 640)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sharpness_factor(score: float) -> float:
    if score >= SHARP_OK:
        return 1.0
    if score >= SHARP_MIN:
        return 0.85
    return 0.6


def tilt_factor(tilt_deg: float) -> float:
    if tilt_deg <= 10.0:
        return 1.0
    if tilt_deg <= 15.0:
        return 1.0 - 0.15 * (tilt_deg - 10.0) / 5.0
    return 0.6


def roll_factor(roll_deg: float) -> float:
    r = abs(roll_deg)
    if r <= 4.0:
        return 1.0
    if r <= 8.0:
        return 0.9
    return 0.75
