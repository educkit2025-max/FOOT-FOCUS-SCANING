"""Agent 9 — Visualization: overlay geometry for the 3D viewer and debug images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.agents.base import AgentOutput

AGENT = "visualization"

LANDMARK_LABELS = {
    "posterior_heel": "Heel",
    "longest_toe": "Longest toe",
    "hallux_tip": "Big toe",
    "first_metatarsal_head": "1st metatarsal head",
    "fifth_metatarsal_head": "5th metatarsal head",
    "ball_line_midpoint": "Ball line",
    "instep_point": "Instep",
    "hallux_top": "Big toe top",
}


def run(side: str, top: dict, side_view: dict | None, measurements: list[dict], mesh: dict | None) -> AgentOutput:
    overlays = []
    for m in measurements:
        if m["value_mm"] is None:
            continue
        if m.get("ring"):
            overlays.append({"key": m["key"], "type": "ring", "points": m["ring"], "color": m["color"], "label": m["display_value"]})
        elif m.get("start_xyz") and m.get("end_xyz"):
            overlays.append(
                {"key": m["key"], "type": "line", "start": m["start_xyz"], "end": m["end_xyz"], "color": m["color"], "label": m["display_value"]}
            )
    landmarks = []
    for key, uv in (top or {}).get("landmarks_mm", {}).items():
        if key in LANDMARK_LABELS:
            landmarks.append({"id": key, "index": len(landmarks) + 1, "label": LANDMARK_LABELS[key], "xyz": [uv[1], uv[0], 0.0]})
    if side_view:
        lm = (top or {}).get("landmarks_mm") or {}
        mid = lm.get("midfoot_a"), lm.get("midfoot_b")
        mid_v = float(np.mean([mid[0][1], mid[1][1]])) if None not in mid else 0.0
        for key, uh in side_view.get("landmarks_mm", {}).items():
            if key in LANDMARK_LABELS:
                landmarks.append({"id": key, "index": len(landmarks) + 1, "label": LANDMARK_LABELS[key], "xyz": [round(mid_v, 1), uh[0], uh[1]]})
    if not landmarks:
        length = next((m for m in measurements if m["key"] == "foot_length" and m.get("start_xyz") and m.get("end_xyz")), None)
        if length:
            landmarks = [
                {"id": "posterior_heel", "index": 1, "label": "Heel", "xyz": length["start_xyz"]},
                {"id": "longest_toe", "index": 2, "label": "Longest toe", "xyz": length["end_xyz"]},
            ]
    data = {
        "side": side,
        "units": "mm",
        "overlays": overlays,
        "landmarks": landmarks,
        "contour_top": (top or {}).get("outline_uv", []),
        "profile_side": (side_view or {}).get("profile_uh", []),
        "has_mesh": bool(mesh),
        "camera_presets": {
            "top": {"yaw": 0, "pitch": 90},
            "medial": {"yaw": 90 if side == "right" else -90, "pitch": 0},
            "lateral": {"yaw": -90 if side == "right" else 90, "pitch": 0},
            "oblique": {"yaw": 35 if side == "right" else -35, "pitch": 32},
            "animation_ms": 420,
        },
    }
    return AgentOutput(AGENT, True, 1.0, data)


def draw_top_debug(rect_image: np.ndarray, mask: np.ndarray, path: Path) -> None:
    out = rect_image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (200, 125, 20), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)
