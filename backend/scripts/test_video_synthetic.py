"""End-to-end check of the 30 s video flow on a rendered foot of known size.

Run from backend/:  .venv\\Scripts\\python -m scripts.test_video_synthetic
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.imaging import load_image_bytes  # noqa: E402
from app.services import storage  # noqa: E402
from app.services.pipeline import process_scan_frames  # noqa: E402
from app.services.video_measure import measure_scan  # noqa: E402
from scripts.test_profile_synthetic import render_side, render_top  # noqa: E402

FOOT = "right"
TRUTH = {"foot_length": 262.0, "max_foot_width": 100.0, "instep_height": 68.0}
FPS = 8
SECONDS = 30
SIZE = (720, 960)


def frame(data: bytes) -> np.ndarray:
    img = cv2.resize(load_image_bytes(data), SIZE, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 1.8, blur, -0.8, 0)


def main() -> int:
    rng = np.random.default_rng(3)
    tops = [frame(render_top(TRUTH["foot_length"], TRUTH["max_foot_width"], FOOT, rng.normal(0, 6, 2), rng)) for _ in range(5)]
    sides = [frame(render_side(TRUTH["foot_length"], TRUTH["instep_height"], FOOT, rng)) for _ in range(3)]
    outer = [cv2.flip(s, 1) for s in sides]

    video = Path(tempfile.gettempdir()) / "nikonyx_synthetic_scan.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for i in range(FPS * SECONDS):
        t = i / FPS
        if t < 5 or t >= 22:
            img = tops[i % len(tops)]
        elif t < 10:
            img = sides[i % len(sides)]
        elif t < 16:
            img = sides[(i + 1) % len(sides)]
        else:
            img = outer[i % len(outer)]
        # Exposure drift, as on a real phone, so the duplicate-view filter keeps distinct frames.
        gain = 0.8 + 0.4 * ((i * 7) % 11) / 10.0
        noise = rng.normal(0, 2, img.shape)
        writer.write(np.clip(img * gain + noise, 0, 255).astype(np.uint8))
    writer.release()

    meta = storage.create_scan({"foot": FOOT, "device": {}, "capture": {}, "notes": "synthetic"})
    sid = meta["scan_id"]
    dest = storage.scan_dir(sid) / "video" / "capture.mp4"
    dest.write_bytes(video.read_bytes())
    storage.set_status(sid, storage.ScanStatus.ready, extra={"paths": {**meta["paths"], "video": str(dest)}})

    step1 = process_scan_frames(sid, 70)
    print("scan", sid, "selected", step1["frames"]["selected_frame_count"])
    for b in step1["best"]:
        print("  best", b["rank"], b["label"], "t=%.1fs" % b["time_s"], "A4" if b["a4_detected"] else "-", b["quality_score"])
    r = measure_scan(sid)
    print("status", r["status"], r["summary"], r["frames_used"])
    print("retake", r["retake"])
    worst = 0.0
    for m in r["foot_result"]["measurements"]:
        exp = TRUTH.get(m["key"])
        err = ""
        if exp is not None and m["value_mm"] is not None:
            err = f"  truth {exp:.0f}  err {m['value_mm'] - exp:+.1f} mm"
            worst = max(worst, abs(m["value_mm"] - exp))
        print(f"  {m['number']:>2} {m['measurement']:<22} {m['display_value']:<26} {m['confidence'] * 100:5.1f}% {m['status']}{err}")
    print("footwear", r["footwear"])
    print("worst error %.1f mm" % worst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
