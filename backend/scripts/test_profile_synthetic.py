"""End-to-end check of the NIKONYX profile pipeline on rendered feet of known size.

Run from backend/:  .venv\\Scripts\\python -m scripts.test_profile_synthetic
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.orchestrator import process_profile  # noqa: E402
from app.services import profile_storage as store  # noqa: E402

W, H = 1500, 2000
F = 0.77 * H
SKIN = np.array([125, 155, 205], np.float64)


def foot_outline(L: float, Wmax: float, side: str, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """Boundary (u, v) in mm and the boundary height for each point. u: heel→toe, v: user's right."""
    t = np.linspace(0, 1, n)
    half = Wmax / 2 * np.interp(t, [0, 0.03, 0.12, 0.3, 0.5, 0.7, 0.8, 0.9, 0.96, 1.0], [0.0, 0.45, 0.68, 0.72, 0.78, 1.0, 0.98, 0.82, 0.55, 0.0])
    medial = -1.0 if side == "right" else 1.0
    toe_drop = np.clip((t - 0.80) / 0.20, 0, 1) ** 2 * 18.0
    u_med = t * L
    u_lat = t * L - toe_drop
    med = np.column_stack([u_med, medial * half])
    lat = np.column_stack([np.minimum(u_lat, L - 18), -medial * half])
    pts = np.vstack([med, lat[::-1]])
    heights = np.where(pts[:, 0] < 0.15 * L, 18.0, np.where(pts[:, 0] > 0.85 * L, 12.0, 15.0))
    return pts, heights


def look_at(cam: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    z = target - cam
    z /= np.linalg.norm(z)
    x = np.cross(z, up_hint)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def project(R, cam, pts3):
    pc = (pts3 - cam) @ R.T
    return np.column_stack([F * pc[:, 0] / pc[:, 2] + W / 2, F * pc[:, 1] / pc[:, 2] + H / 2])


def render_top(L, Wmax, side, jitter, rng) -> bytes:
    img = np.clip(rng.normal(95, 8, (H, W, 3)), 0, 255).astype(np.uint8)
    cam = np.array([105.0 + jitter[0], 175.0 + jitter[1], 520.0])
    target = np.array([105.0, 150.0, 0.0]) + np.array([np.tan(np.radians(6)) * 520, 0, 0])
    R = look_at(cam, target, np.array([0.0, 1.0, 0.0]))
    paper = np.array([[0, 0, 0], [210, 0, 0], [210, 297, 0], [0, 297, 0]], np.float64)
    cv2.fillPoly(img, [project(R, cam, paper).astype(np.int32)], (238, 240, 242))
    uv, hz = foot_outline(L, Wmax, side)
    ang = np.radians(3.0)
    d = np.array([np.sin(ang), np.cos(ang)])
    r = np.array([d[1], -d[0]])
    origin = np.array([105.0, 25.0])
    xy = origin + uv[:, :1] * d + uv[:, 1:2] * r
    pts3 = np.column_stack([xy, hz])
    poly = project(R, cam, pts3).astype(np.int32)
    foot = np.zeros((H, W), np.uint8)
    cv2.fillPoly(foot, [poly], 255)
    noise = rng.normal(0, 6, (H, W, 3))
    img = np.where(foot[..., None] > 0, np.clip(SKIN + noise, 0, 255), img).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


def instep_curve(L: float, instep: float):
    u = np.linspace(0, L, 200)
    z = np.interp(u / L, [0, 0.05, 0.2, 0.42, 0.55, 0.75, 0.88, 0.95, 1.0], [0, 45, 80, 110, instep, 0.62 * instep, 28, 20, 0])
    return u, z


def render_side(L, instep, side, rng) -> bytes:
    img = np.clip(rng.normal(150, 6, (H, W, 3)), 0, 255).astype(np.uint8)
    img[:, :] = img * np.array([0.85, 0.9, 1.0])
    s = 4.4
    ground = int(H * 0.72)
    img[ground:] = np.clip(rng.normal(90, 6, (H - ground, W, 3)), 0, 255)
    cv2.rectangle(img, (int(W * 0.05), ground), (int(W * 0.95), ground + 60), (236, 238, 240), -1)
    x0 = (W - L * s) / 2
    u, z = instep_curve(L, instep)
    toes_right = side == "left"
    xs = x0 + (u if toes_right else (L - u)) * s
    top = np.column_stack([xs, ground - z * s])
    poly = np.vstack([top, [[xs[-1], ground], [xs[0], ground]]]).astype(np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    heel_x = x0 + (0.02 * L if toes_right else 0.98 * L) * s
    ankle_x = x0 + (0.40 * L if toes_right else 0.60 * L) * s
    lx0, lx1 = sorted([heel_x, ankle_x])
    cv2.rectangle(mask, (int(lx0), 0), (int(lx1), ground - int(40 * s)), 255, -1)
    noise = rng.normal(0, 5, (H, W, 3))
    img = np.where(mask[..., None] > 0, np.clip(SKIN + noise, 0, 255), img).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


TRUTH = {
    "left": {"foot_length": 258.0, "max_foot_width": 98.0, "instep_height": 66.0},
    "right": {"foot_length": 262.0, "max_foot_width": 100.0, "instep_height": 68.0},
}


def main() -> int:
    rng = np.random.default_rng(7)
    pid = store.create_profile({"device_model": "synthetic"})
    for side, t in TRUTH.items():
        tops = [render_top(t["foot_length"], t["max_foot_width"], side, rng.normal(0, 4, 2), rng) for _ in range(3)]
        store.save_view(pid, f"{side}_top", tops, {"focal_mm": None, "sensor_long_mm": None})
        sides = [render_side(t["foot_length"], t["instep_height"], side, rng) for _ in range(3)]
        store.save_view(pid, f"{side}_medial", sides, {"gravity_x": 0.0, "gravity_y": 9.8, "gravity_z": 0.3})
    process_profile(pid)
    res = store.read_result(pid)
    print("profile:", pid, "status:", res.get("status"))
    if res.get("status") == "failed":
        print(res.get("trace"))
        return 1
    print("summary:", res["summary"])
    print("retake:", res["retake"])
    print("warnings:", res["warnings"])
    worst = 0.0
    for side, truth in TRUTH.items():
        print(f"\n{side.upper()} foot")
        for m in res["feet"][side]["measurements"]:
            exp = truth.get(m["key"])
            err = "" if exp is None or m["value_mm"] is None else f"  truth {exp:.1f}  err {m['value_mm'] - exp:+.1f} mm"
            if exp is not None and m["value_mm"] is not None:
                worst = max(worst, abs(m["value_mm"] - exp))
            print(f"  {m['number']:>2} {m['measurement']:<22} {m['display_value']:<26} {m['confidence'] * 100:5.1f}% {m['status']}{err}")
    print("\nfootwear:", res["footwear"])
    print("worst error on truth fields: %.1f mm" % worst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
