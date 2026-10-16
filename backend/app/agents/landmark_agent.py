"""Agent 4 — Landmarks: anatomical points and profiles from the top and medial views.

Foot frame (mm): u runs from the posterior heel (u=0) toward the toes, v points to the user's right.
For a right foot the medial side is −v, for a left foot it is +v.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.agents.base import AgentOutput, failed
from app.agents.calibration_agent import PaperGeometry, RectifiedView

AGENT = "landmarks"
RASTER_PX_PER_MM = 4.0

# Height above the paper at which the silhouette boundary sits (heel bulge, side walls, toe tips).
BOUNDARY_HEIGHT_MM = {"heel": 18.0, "side": 15.0, "toe": 12.0}


def _pca_direction(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((pts - c).T))
    d = vecs[:, -1]
    return d / np.linalg.norm(d)


def _to_frame(pts: np.ndarray, origin: np.ndarray, d: np.ndarray) -> np.ndarray:
    r = np.array([d[1], -d[0]])
    rel = pts - origin
    return np.column_stack([rel @ d, rel @ r])


def _from_frame(uv: np.ndarray, origin: np.ndarray, d: np.ndarray) -> np.ndarray:
    r = np.array([d[1], -d[0]])
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    return origin + uv[:, :1] * d + uv[:, 1:2] * r


def _raster_profile(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Rows every 1/RASTER_PX_PER_MM mm along u: (u, v_min, v_max), NaN where empty; returns u0 offset."""
    s = RASTER_PX_PER_MM
    u0, v0 = uv[:, 0].min() - 2, uv[:, 1].min() - 2
    pts = np.round((uv - [u0, v0]) * s).astype(np.int32)
    h = int(pts[:, 0].max()) + 4
    w = int(pts[:, 1].max()) + 4
    img = np.zeros((h, w), np.uint8)
    cv2.fillPoly(img, [pts[:, ::-1].reshape(-1, 1, 2)], 255)
    rows = img > 0
    has = rows.any(axis=1)
    vmin = np.where(has, np.argmax(rows, axis=1), np.nan).astype(np.float64)
    vmax = np.where(has, w - 1 - np.argmax(rows[:, ::-1], axis=1), np.nan).astype(np.float64)
    u = np.arange(h) / s + u0
    return u, vmin / s + v0, vmax / s + v0, u0


def _correct_parallax(pts: np.ndarray, cam: np.ndarray, d: np.ndarray, origin: np.ndarray, length: float) -> np.ndarray:
    """Move silhouette points from where their ray hits the paper back to their true footprint."""
    cz = float(abs(cam[2]))
    if cz < 50:
        return pts
    t = _to_frame(pts, origin, d)[:, 0] / max(length, 1.0)
    h = np.where(t < 0.15, BOUNDARY_HEIGHT_MM["heel"], np.where(t > 0.85, BOUNDARY_HEIGHT_MM["toe"], BOUNDARY_HEIGHT_MM["side"]))
    nadir = cam[:2]
    return nadir + (pts - nadir) * (1.0 - h / cz)[:, None]


def _orient(uv_pts: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, bool]:
    """Flip the axis if the narrow (heel) end is at +u; forefoot is always wider than the heel."""
    u = uv_pts[:, 0]
    lo, hi = u.min(), u.max()
    span = hi - lo
    rear = uv_pts[u < lo + 0.2 * span, 1]
    front = uv_pts[u > hi - 0.35 * span, 1]
    rear_w = np.ptp(rear) if len(rear) else 0
    front_w = np.ptp(front) if len(front) else 0
    if rear_w > front_w:
        return -d, True
    return d, False


def _window(u, vmin, vmax, lo, hi):
    sel = (u >= lo) & (u <= hi) & ~np.isnan(vmin)
    return np.nonzero(sel)[0]


def _pt(u: float, v: float) -> list[float]:
    return [round(float(u), 2), round(float(v), 2)]


def locate_top(mask: np.ndarray, contour: np.ndarray, rect: RectifiedView, paper: PaperGeometry, expected_side: str) -> AgentOutput:
    pts = rect.rect_to_plane(contour.reshape(-1, 2).astype(np.float64))
    if len(pts) < 50:
        return failed(AGENT, "CONTOUR_TOO_SMALL", ["Foot outline too small"])

    d = _pca_direction(pts)
    if d[1] < 0:
        d = -d
    corrected = pts
    origin = pts.mean(axis=0)
    for _ in range(3):
        uv = _to_frame(pts, origin, d)
        length = float(np.ptp(uv[:, 0]))
        origin = _from_frame([[uv[:, 0].min(), 0.0]], origin, d)[0]
        corrected = _correct_parallax(pts, paper.camera_center_mm, d, origin, length)
        uv = _to_frame(corrected, origin, d)
        d, flipped = _orient(uv, d)
        uv = _to_frame(corrected, origin, d)
        L = float(np.ptp(uv[:, 0]))
        u_rel = uv[:, 0] - uv[:, 0].min()
        heel_c = corrected[u_rel < 0.10 * L].mean(axis=0)
        fore_c = corrected[(u_rel > 0.60 * L) & (u_rel < 0.75 * L)].mean(axis=0)
        nd = fore_c - heel_c
        nd /= np.linalg.norm(nd)
        if np.degrees(np.arccos(np.clip(nd @ d, -1, 1))) <= 25.0:
            d = nd
        origin = heel_c

    uv = _to_frame(corrected, origin, d)
    uv[:, 0] -= uv[:, 0].min()
    L = float(uv[:, 0].max())
    u, vmin, vmax, _ = _raster_profile(uv)
    width = vmax - vmin

    heel_i = int(np.argmin(uv[:, 0]))
    toe_i = int(np.argmax(uv[:, 0]))
    posterior_heel = uv[heel_i]
    longest_toe = uv[toe_i]

    # Toe-end asymmetry: the medial (hallux) side reaches further forward than the lateral side.
    tip_zone = uv[uv[:, 0] > 0.80 * L]
    v_mid = float(np.nanmedian((vmin + vmax)[_window(u, vmin, vmax, 0.80 * L, 0.90 * L)] / 2.0))
    left_reach = tip_zone[tip_zone[:, 1] < v_mid, 0].max(initial=0.0)
    right_reach = tip_zone[tip_zone[:, 1] >= v_mid, 0].max(initial=0.0)
    reach_diff = float(left_reach - right_reach)
    detected_side = "right" if reach_diff > 0 else "left"
    side_conf = float(np.clip(abs(reach_diff) / 8.0, 0.0, 1.0))
    side = expected_side
    medial_sign = -1.0 if side == "right" else 1.0
    hallux_pool = tip_zone[(tip_zone[:, 1] - v_mid) * medial_sign > 0]
    hallux_tip = hallux_pool[np.argmax(hallux_pool[:, 0])] if len(hallux_pool) else longest_toe

    def extreme(lo, hi, sign):
        idx = _window(u, vmin, vmax, lo * L, hi * L)
        if len(idx) == 0:
            return None
        vals = (vmax[idx] if sign > 0 else vmin[idx]) * sign
        # Flat borders make a single argmax jump; take the middle of the near-extreme run.
        near = idx[vals >= vals.max() - 1.0]
        j = int(near[len(near) // 2])
        return np.array([u[j], vmax[j] if sign > 0 else vmin[j]])

    def max_width(lo, hi):
        idx = _window(u, vmin, vmax, lo * L, hi * L)
        if len(idx) == 0:
            return None
        j = idx[int(np.nanargmax(width[idx]))]
        return float(width[j]), np.array([u[j], vmin[j]]), np.array([u[j], vmax[j]])

    def width_at(frac):
        idx = _window(u, vmin, vmax, frac * L - 0.5, frac * L + 0.5)
        if len(idx) == 0:
            return None
        j = idx[len(idx) // 2]
        return float(width[j]), np.array([u[j], vmin[j]]), np.array([u[j], vmax[j]])

    first_mth = extreme(0.62, 0.82, medial_sign)
    fifth_mth = extreme(0.55, 0.78, -medial_sign)
    heel = max_width(0.06, 0.22)
    widest = max_width(0.25, 0.95)
    forefoot = max_width(0.55, 0.85)
    midfoot = width_at(0.50)
    toe_box = width_at(0.85)
    toe_span = width_at(0.93)
    if any(x is None for x in (first_mth, fifth_mth, heel, widest, forefoot, midfoot, toe_box, toe_span)):
        return failed(AGENT, "LANDMARKS_INCOMPLETE", ["Foot outline incomplete — retake the top view"])

    ball_mid = (first_mth + fifth_mth) / 2.0

    flags = {"leg_intrusion_suspected": False, "heel_boundary_uncertain": False}
    rear = width_at(0.02)
    if rear and heel and rear[0] > 0.80 * heel[0]:
        flags["leg_intrusion_suspected"] = True
        flags["heel_boundary_uncertain"] = True

    landmarks = {
        "posterior_heel": _pt(*posterior_heel),
        "longest_toe": _pt(*longest_toe),
        "hallux_tip": _pt(*hallux_tip),
        "first_metatarsal_head": _pt(*first_mth),
        "fifth_metatarsal_head": _pt(*fifth_mth),
        "ball_line_midpoint": _pt(*ball_mid),
        "heel_medial": _pt(*(heel[1] if medial_sign < 0 else heel[2])),
        "heel_lateral": _pt(*(heel[2] if medial_sign < 0 else heel[1])),
        "max_width_a": _pt(*widest[1]),
        "max_width_b": _pt(*widest[2]),
        "forefoot_a": _pt(*forefoot[1]),
        "forefoot_b": _pt(*forefoot[2]),
        "midfoot_a": _pt(*midfoot[1]),
        "midfoot_b": _pt(*midfoot[2]),
        "toe_box_a": _pt(*toe_box[1]),
        "toe_box_b": _pt(*toe_box[2]),
        "toe_span_a": _pt(*toe_span[1]),
        "toe_span_b": _pt(*toe_span[2]),
    }
    raw = {
        "foot_length": float(longest_toe[0] - posterior_heel[0]),
        "max_foot_width": widest[0],
        "ball_width": float(np.linalg.norm(first_mth - fifth_mth)),
        "forefoot_width": forefoot[0],
        "heel_width": heel[0],
        "arch_length": float(first_mth[0] - posterior_heel[0]),
        "heel_to_ball_length": float(ball_mid[0] - posterior_heel[0]),
        "midfoot_width": midfoot[0],
        "toe_box_width": toe_box[0],
        "toe_span": toe_span[0],
    }

    step = max(1, int(round(RASTER_PX_PER_MM * 2)))
    keep = np.nonzero(~np.isnan(vmin))[0][::step]
    outline = [[round(float(u[i]), 1), round(float(vmin[i]), 1), round(float(vmax[i]), 1)] for i in keep]

    conf = 0.95
    messages: list[str] = []
    if flags["leg_intrusion_suspected"]:
        conf *= 0.7
        messages.append("Leg may overlap the heel — hold the phone further forward over the toes")
    if side_conf >= 0.5 and detected_side != expected_side:
        conf *= 0.5
        messages.append(f"This looks like the {detected_side} foot — capture the {expected_side} foot")

    data = {
        "side": side,
        "detected_side": detected_side,
        "side_confidence": round(side_conf, 3),
        "axis_flipped": bool(flipped),
        "axis_dir_plane": [round(float(d[0]), 4), round(float(d[1]), 4)],
        "origin_plane_mm": [round(float(origin[0]), 2), round(float(origin[1]), 2)],
        "landmarks_mm": landmarks,
        "raw_mm": {k: round(v, 2) for k, v in raw.items()},
        "outline_uv": outline,
        "flags": flags,
        "landmark_confidence": round(conf, 3),
    }
    return AgentOutput(AGENT, True, conf, data, messages=messages, artifacts={"uv": uv, "origin": origin, "d": d})


def locate_side(mask: np.ndarray, view_side: str, top_length_mm: float | None) -> AgentOutput:
    """Medial profile: ground line, heel/toe ends, dorsal height curve in mm (scaled from the top view)."""
    h, w = mask.shape[:2]
    cols = np.nonzero(mask.any(axis=0))[0]
    if len(cols) < 40:
        return failed(AGENT, "SIDE_PROFILE_TOO_SMALL", ["Side view too small — move closer"])
    bottoms = np.array([h - 1 - np.argmax(mask[::-1, c] > 0) for c in cols], np.float64)
    lowest = np.percentile(bottoms, 90)
    ground_sel = bottoms >= lowest - 0.03 * h
    if ground_sel.sum() < 10:
        return failed(AGENT, "GROUND_NOT_FOUND", ["Ground line not visible — keep the floor in view"])
    line = cv2.fitLine(np.column_stack([cols[ground_sel], bottoms[ground_sel]]).astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    vx, vy, x0, y0 = map(float, line)
    if vx < 0:
        vx, vy = -vx, -vy
    roll = float(np.degrees(np.arctan2(vy, vx)))
    Rm = cv2.getRotationMatrix2D((x0, y0), roll, 1.0)
    level = cv2.warpAffine(mask, Rm, (w, h), flags=cv2.INTER_NEAREST)
    ground_y = y0

    rows_any = level.any(axis=1)
    top_row = int(np.argmax(rows_any))
    band_top = int(max(top_row, ground_y - 0.06 * w))
    band = level[band_top : int(ground_y) + 2]
    bcols = np.nonzero(band.any(axis=0))[0]
    if len(bcols) < 30:
        return failed(AGENT, "SIDE_PROFILE_INCOMPLETE", ["Foot sole not visible in the side view"])
    x_lo, x_hi = int(bcols.min()), int(bcols.max())
    length_px = float(x_hi - x_lo)

    heights = np.array([ground_y - np.argmax(level[:, c] > 0) if level[:, c].any() else 0.0 for c in range(x_lo, x_hi + 1)])
    n = len(heights)
    left_h = heights[: int(0.3 * n)].max(initial=0)
    right_h = heights[int(0.7 * n) :].max(initial=0)
    toes_right = left_h > right_h
    expect_right = view_side == "left"
    wrong_direction = toes_right != expect_right
    if not toes_right:
        heights = heights[::-1]

    if not top_length_mm:
        return failed(AGENT, "NO_SCALE", ["Top view needed to scale the side view"])
    mm_per_px = float(top_length_mm) / length_px
    u_mm = np.arange(n) * mm_per_px
    h_mm = heights * mm_per_px
    L = float(top_length_mm)

    def height_at(frac, half_window=0.02):
        sel = (u_mm >= (frac - half_window) * L) & (u_mm <= (frac + half_window) * L)
        return float(np.median(h_mm[sel])) if sel.any() else None

    instep = height_at(0.55)
    toe_sel = (u_mm >= 0.86 * L) & (u_mm <= 0.96 * L)
    hallux_h = float(h_mm[toe_sel].max()) if toe_sel.any() else None
    ankle_cut = 0.42 * L
    profile = []
    for uu in np.arange(0, L + 0.01, 2.0):
        j = min(n - 1, int(round(uu / mm_per_px)))
        profile.append([round(float(uu), 1), round(float(min(h_mm[j], ankle_cut)), 1)])

    conf = 0.88
    messages: list[str] = []
    if abs(roll) > 8:
        conf *= 0.75
        messages.append("Camera not level — hold the phone straight")
    if wrong_direction:
        conf *= 0.6
        messages.append("Wrong side — show the inner side of the foot")
    if instep is None or instep > 0.45 * L or instep < 0.12 * L:
        conf *= 0.6
        messages.append("Instep profile unclear — keep the ankle and top of the foot visible")

    data = {
        "view_side": view_side,
        "roll_deg": round(roll, 2),
        "toes_right": bool(toes_right),
        "wrong_direction": bool(wrong_direction),
        "length_px": round(length_px, 1),
        "mm_per_px": round(mm_per_px, 4),
        "raw_mm": {
            "instep_height": round(instep, 2) if instep is not None else None,
            "hallux_height": round(hallux_h, 2) if hallux_h is not None else None,
        },
        "landmarks_mm": {
            "instep_point": [round(0.55 * L, 2), round(instep or 0.0, 2)],
            "hallux_top": [round(0.91 * L, 2), round(hallux_h or 0.0, 2)],
        },
        "profile_uh": profile,
        "landmark_confidence": round(conf, 3),
    }
    return AgentOutput(AGENT, True, conf, data, messages=messages)
