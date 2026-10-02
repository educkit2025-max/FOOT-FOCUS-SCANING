"""Extract and select useful frames from a scan video (Milestone 1)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.config import settings


@dataclass
class FrameScore:
    index: int
    path: str
    sharpness: float
    selected: bool
    discard_reason: str | None = None


def _run_ffmpeg_extract(video_path: Path, frames_dir: Path) -> None:
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Extract at ~8 fps (30s scan → ~240 candidates; then keep ~60–80 best).
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "fps=8",
        str(frames_dir / "cand_%05d.png"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-800:]}")


def _laplacian_sharpness(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    ha = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hb = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))


def _write_thumbnail(src: Path, dest: Path, max_width: int) -> None:
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w > max_width:
            nh = int(h * (max_width / w))
            im = im.resize((max_width, nh), Image.Resampling.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, format="JPEG", quality=85)


def select_frames(
    video_path: Path,
    frames_dir: Path,
    selected_dir: Path,
    thumbnails_dir: Path,
    target_frames: int,
    blur_min: float | None = None,
) -> dict:
    blur_min = settings.blur_laplacian_min if blur_min is None else blur_min

    _run_ffmpeg_extract(video_path, frames_dir)
    candidates = sorted(frames_dir.glob("cand_*.png"))
    if not candidates:
        raise RuntimeError("No frames extracted from video")

    scored: list[FrameScore] = []
    images: list[np.ndarray] = []
    for i, path in enumerate(candidates):
        img = cv2.imread(str(path))
        if img is None:
            scored.append(FrameScore(i, path.name, 0.0, False, "unreadable"))
            images.append(None)  # type: ignore[arg-type]
            continue
        sharp = _laplacian_sharpness(img)
        reason = None
        selected = True
        if sharp < blur_min:
            selected = False
            reason = "blur"
        scored.append(FrameScore(i, path.name, sharp, selected, reason))
        images.append(img)

    # Novelty filter among sharp frames: keep temporally spread unique views.
    sharp_indices = [s.index for s in scored if s.selected]
    if not sharp_indices:
        # Fallback: take sharpest N regardless of threshold.
        order = sorted(range(len(scored)), key=lambda i: scored[i].sharpness, reverse=True)
        sharp_indices = order[: min(target_frames, len(order))]
        for s in scored:
            s.selected = s.index in sharp_indices
            s.discard_reason = None if s.selected else (s.discard_reason or "low_sharpness_pool")

    kept: list[int] = []
    last_img: np.ndarray | None = None
    for idx in sharp_indices:
        img = images[idx]
        if img is None:
            scored[idx].selected = False
            scored[idx].discard_reason = "unreadable"
            continue
        if last_img is not None and _hist_distance(last_img, img) < 0.08:
            scored[idx].selected = False
            scored[idx].discard_reason = "duplicate_view"
            continue
        kept.append(idx)
        last_img = img

    # Uniform subsample to target_frames if still too many.
    if len(kept) > target_frames:
        step = len(kept) / target_frames
        chosen = sorted({kept[int(i * step)] for i in range(target_frames)})
        chosen_set = set(chosen)
        for idx in kept:
            if idx not in chosen_set:
                scored[idx].selected = False
                scored[idx].discard_reason = "subsampled"
        kept = chosen
    else:
        for idx in kept:
            scored[idx].selected = True
            scored[idx].discard_reason = None

    # Mark non-kept sharp candidates discarded if still marked selected incorrectly.
    kept_set = set(kept)
    for s in scored:
        if s.index not in kept_set and s.discard_reason is None and not s.selected:
            pass
        if s.index not in kept_set:
            s.selected = False
            if s.discard_reason is None:
                s.discard_reason = "not_selected"

    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True, exist_ok=True)
    if thumbnails_dir.exists():
        shutil.rmtree(thumbnails_dir)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    frame_records = []
    for rank, idx in enumerate(kept):
        src = candidates[idx]
        out_name = f"{rank:04d}_{src.name}"
        dest = selected_dir / out_name
        shutil.copy2(src, dest)
        thumb_name = f"{rank:04d}.jpg"
        _write_thumbnail(src, thumbnails_dir / thumb_name, settings.thumbnail_max_width)
        frame_records.append(
            {
                "index": rank,
                "source_index": idx,
                "filename": out_name,
                "thumbnail": thumb_name,
                "sharpness": scored[idx].sharpness,
                "selected": True,
                "discard_reason": None,
            }
        )

    report = {
        "original_candidate_count": len(candidates),
        "selected_frame_count": len(kept),
        "target_frames": target_frames,
        "blur_laplacian_min": blur_min,
        "frames": frame_records,
        "all_scores": [asdict(s) for s in scored],
    }
    (frames_dir.parent / "frame_selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
