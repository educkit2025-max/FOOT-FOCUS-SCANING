"""Agent 3 — Segmentation: isolate the foot from paper, floor, leg and shadow."""

from __future__ import annotations

import cv2
import numpy as np

from app.agents.base import AgentOutput, failed
from app.agents.calibration_agent import RectifiedView
from app.agents.imaging import downscale

AGENT = "segmentation"
EDGE_BAND_MM = 2.5
MIN_FOOT_AREA_MM2 = 6000.0


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == idx, 255, 0).astype(np.uint8)


def segment_top(rect: RectifiedView) -> AgentOutput:
    s = rect.px_per_mm
    x0, y0, x1, y1 = rect.paper_box_px()
    lab = cv2.cvtColor(rect.image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]

    paper_region = np.zeros(rect.valid.shape, np.uint8)
    paper_region[y0:y1, x0:x1] = 255
    paper_region &= rect.valid
    inside = paper_region > 0
    if inside.sum() < 1000:
        return failed(AGENT, "PAPER_NOT_VISIBLE", ["A4 paper not visible in the top view"])

    l_in = L[inside]
    l_ref = float(np.percentile(l_in, 85))
    bright = inside & (L > l_ref - 20)
    a_ref = float(np.median(A[bright])) if bright.any() else 128.0
    b_ref = float(np.median(B[bright])) if bright.any() else 128.0

    chroma = np.hypot(A - a_ref, B - b_ref)
    warmth = (A - a_ref) + (B - b_ref)
    # Shadows lower L and keep or cool the paper colour, even hard cast shadows (L down by ~110); skin of
    # every tone shifts warm (red/yellow). Strongly coloured socks and near-black socks also count.
    foot = (
        ((warmth > 10.0) & (L < l_ref - 6))
        | ((warmth > 4.0) & (L < l_ref - 55))
        | ((chroma > 30.0) & (L < l_ref - 25))
        | (L < l_ref - 140)
    )
    mask = (foot & inside).astype(np.uint8) * 255

    k = max(3, int(round(2.0 * s)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = _largest_component(mask)
    mask = _fill_holes(mask)

    area_mm2 = float(mask.sum() / 255) / (s * s)
    if area_mm2 < MIN_FOOT_AREA_MM2:
        return failed(AGENT, "FOOT_NOT_DETECTED", ["Foot not detected on the paper"], {"area_mm2": round(area_mm2, 1)})

    band = int(round(EDGE_BAND_MM * s))
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    contact = {
        "heel_edge": bool(mask[y1 - band : y1, x0:x1].any()),
        "toe_edge": bool(mask[y0 : y0 + band, x0:x1].any()),
        "left_edge": bool(mask[y0:y1, x0 : x0 + band].any()),
        "right_edge": bool(mask[y0:y1, x1 - band : x1].any()),
    }
    toe_band_cols = mask[y0 : y0 + band, x0:x1].sum(axis=0) > 0
    toe_contact_mm = float(toe_band_cols.sum()) / s

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    hull_area = max(cv2.contourArea(cv2.convexHull(cnt)), 1.0)
    solidity = float(cv2.contourArea(cnt)) / hull_area

    shadow_like = inside & (L < l_ref - 6) & (chroma <= 14.0) & (mask == 0)
    shadow_frac = float(shadow_like.sum()) / max(float(mask.sum() / 255), 1.0)

    conf = 0.97
    messages: list[str] = []
    if contact["toe_edge"] and toe_contact_mm > 8:
        conf *= 0.55
        messages.append("Toes outside the paper — move the foot back")
    if contact["heel_edge"]:
        conf *= 0.8
        messages.append("Heel touches the paper edge — place the heel fully on the paper")
    if contact["left_edge"] or contact["right_edge"]:
        conf *= 0.75
        messages.append("Foot touches the side of the paper — centre it")
    if solidity < 0.80:
        conf *= 0.8
        messages.append("Foot outline irregular — check lighting and remove clutter")
    if shadow_frac > 0.25:
        conf *= 0.9
        messages.append("Strong shadow next to the foot — use even lighting")

    data = {
        "area_mm2": round(area_mm2, 1),
        "solidity": round(solidity, 3),
        "edge_contact": contact,
        "toe_contact_mm": round(toe_contact_mm, 1),
        "shadow_fraction": round(shadow_frac, 3),
        "bbox_rows": [int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))],
        "bbox_cols": [int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))],
        "segmentation_confidence": round(conf, 3),
    }
    return AgentOutput(AGENT, True, conf, data, messages=messages, artifacts={"mask": mask, "contour": cnt})


def foot_seed_mask(img: np.ndarray) -> np.ndarray:
    """Warm saturated pixels (skin of any tone) or pixels far darker than the bright background."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    warm = ((h <= 25) | (h >= 165)) & (s >= 60) & (v >= 30)
    dark = v < 0.4 * float(np.percentile(v, 95))
    return (warm | dark).astype(np.uint8) * 255


def _central_component(mask: np.ndarray) -> np.ndarray:
    """Largest blob weighted toward the frame centre, where the user keeps the foot."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    h, w = mask.shape[:2]
    diag = float(np.hypot(h, w))
    d = np.hypot(cents[1:, 0] - w / 2.0, cents[1:, 1] - h / 2.0) / (0.35 * diag)
    score = stats[1:, cv2.CC_STAT_AREA] * np.exp(-(d**2))
    idx = 1 + int(np.argmax(score))
    return np.where(labels == idx, 255, 0).astype(np.uint8)


def segment_side(image: np.ndarray, work_side: int = 600) -> AgentOutput:
    small, scale = downscale(image, work_side)
    h, w = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    skin = foot_seed_mask(small)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    skin = _central_component(skin)

    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    paper = (hsv[..., 2] > 170) & (hsv[..., 1] < 45)
    gc[paper] = cv2.GC_BGD
    border = max(2, int(0.01 * min(h, w)))
    gc[:border, :] = cv2.GC_BGD
    gc[:, :border] = cv2.GC_BGD
    gc[:, -border:] = cv2.GC_BGD

    skin_px = int(skin.sum() / 255)
    if skin_px > 0.02 * h * w:
        gc[skin > 0] = cv2.GC_PR_FGD
        core = cv2.erode(skin, np.ones((9, 9), np.uint8), iterations=2)
        gc[core > 0] = cv2.GC_FGD
        seeded = "skin"
    else:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        dark = (gray < np.percentile(gray, 40)).astype(np.uint8) * 255
        dark = _largest_component(cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)))
        if dark.sum() == 0:
            return failed(AGENT, "FOOT_NOT_DETECTED", ["Foot not detected in the side view"])
        gc[dark > 0] = cv2.GC_PR_FGD
        seeded = "contrast"

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return failed(AGENT, "SEGMENTATION_FAILED", ["Could not separate the foot in the side view"])
    mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = _fill_holes(_largest_component(mask))

    frac = float(mask.sum() / 255) / (h * w)
    if frac < 0.02:
        return failed(AGENT, "FOOT_NOT_DETECTED", ["Foot not detected in the side view"])

    ys, xs = np.nonzero(mask)
    touches = {
        "left": bool(xs.min() <= border + 1),
        "right": bool(xs.max() >= w - border - 2),
        "bottom": bool(ys.max() >= h - border - 2),
    }
    conf = 0.9 if seeded == "skin" else 0.75
    messages: list[str] = []
    if touches["left"] or touches["right"]:
        conf *= 0.6
        messages.append("Foot outside frame — step back so heel and toes are visible")
    data = {
        "area_fraction": round(frac, 3),
        "seed": seeded,
        "touches_frame": touches,
        "work_scale": round(scale, 4),
        "segmentation_confidence": round(conf, 3),
    }
    return AgentOutput(AGENT, True, conf, data, messages=messages, artifacts={"mask": mask, "image": small, "scale": scale})
