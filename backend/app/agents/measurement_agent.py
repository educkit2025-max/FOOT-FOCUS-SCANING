"""Agent 6 — Measurement: turn landmarks and geometry into calibrated values with confidence.

Values are computed in mm and only reported from real geometry. Each method has a confidence
ceiling reflecting how directly the capture observes it; view quality scales it down further.
"""

from __future__ import annotations

import numpy as np

from app.agents.base import AgentOutput
from app.agents.imaging import sharpness_factor, tilt_factor

AGENT = "measurement"
FINAL = 0.90
ESTIMATED = 0.75

COLOR_FINAL = "#147DFF"
COLOR_ESTIMATED = "#F5A623"
COLOR_MISSING = "#EF4444"

# key: (label, number, source, start landmark, end landmark, ceiling)
DEFINITIONS = {
    "foot_length": ("Foot length", 1, "top", "posterior_heel", "longest_toe", 0.97),
    "max_foot_width": ("Maximum foot width", 2, "top", "max_width_a", "max_width_b", 0.95),
    "ball_width": ("Ball width", 3, "top", "first_metatarsal_head", "fifth_metatarsal_head", 0.94),
    "forefoot_width": ("Forefoot width", 4, "top", "forefoot_a", "forefoot_b", 0.94),
    "heel_width": ("Heel width", 5, "top", "heel_medial", "heel_lateral", 0.86),
    "arch_length": ("Arch length", 6, "top", "posterior_heel", "first_metatarsal_head", 0.92),
    "heel_to_ball_length": ("Heel-to-ball length", 7, "top", "posterior_heel", "ball_line_midpoint", 0.92),
    "instep_height": ("Instep height", 8, "side", "floor", "instep_point", 0.88),
    "arch_height": ("Arch height", 9, "plantar", "floor", "navicular", 0.0),
    "midfoot_width": ("Midfoot width", 10, "top", "midfoot_a", "midfoot_b", 0.90),
    "toe_box_width": ("Toe box width", 11, "top", "toe_box_a", "toe_box_b", 0.90),
    "toe_span": ("Toe span", 12, "top", "toe_span_a", "toe_span_b", 0.78),
    "ball_girth": ("Ball girth", 13, "girth", "first_metatarsal_head", "fifth_metatarsal_head", 0.60),
    "forefoot_girth": ("Forefoot girth", 14, "girth", "forefoot_a", "forefoot_b", 0.60),
    "instep_girth": ("Instep girth", 15, "girth", "instep_point", "instep_point", 0.60),
    "heel_girth": ("Heel girth", 16, "girth", "posterior_heel", "instep_point", 0.0),
    "ankle_girth": ("Ankle girth", 17, "girth", "ankle", "ankle", 0.0),
    "hallux_height": ("Big toe height", 18, "side", "floor", "hallux_top", 0.80),
    "foot_volume": ("Foot volume", 19, "3d", None, None, 0.55),
}

UNAVAILABLE_REASON = {
    "arch_height": "Needs a plantar or rear heel view — additional scan required",
    "heel_girth": "Needs the rear heel and outer-side views — additional scan required",
    "ankle_girth": "Needs the ankle in multiple views — additional scan required",
}


def status_for(conf: float) -> str:
    if conf >= FINAL:
        return "final"
    if conf >= ESTIMATED:
        return "estimated"
    return "additional_scan_required"


def color_for(status: str) -> str:
    return {"final": COLOR_FINAL, "estimated": COLOR_ESTIMATED}.get(status, COLOR_MISSING)


def repeatability(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.95, 0.0
    spread = float(max(vals) - min(vals))
    if spread <= 2.0:
        return 1.0, spread
    if spread <= 4.0:
        return 0.9, spread
    return 0.75, spread


def view_quality(view: dict | None) -> float:
    if not view:
        return 0.0
    q = 1.0
    q *= min(1.0, float(view.get("calibration_confidence", 0.9)) / 0.9)
    q *= sharpness_factor(float(view.get("sharpness", 100.0)))
    q *= tilt_factor(float(view.get("tilt_deg", 0.0)))
    q *= min(1.0, float(view.get("segmentation_confidence", 0.9)) / 0.9)
    q *= min(1.0, float(view.get("landmark_confidence", 0.9)) / 0.9)
    return float(np.clip(q, 0.0, 1.0))


def _lift(pt: list | None, z: float) -> list | None:
    if pt is None:
        return None
    return [round(float(pt[1]), 1), round(float(pt[0]), 1), round(z, 1)]


def run(side_name: str, top: dict, top_samples: list[dict], side: dict | None, side_samples: list[dict], mesh: dict | None) -> AgentOutput:
    """top/side: aggregated view records (quality metrics + raw_mm medians + landmarks)."""
    top_q = view_quality(top)
    side_q = view_quality(side) if side else 0.0
    top_lm = (top or {}).get("landmarks_mm", {})
    side_lm = (side or {}).get("landmarks_mm", {})
    girths = (mesh or {}).get("girths", {})
    items = []

    for key, (label, number, source, start, end, ceiling) in DEFINITIONS.items():
        value = None
        spread = 0.0
        conf = 0.0
        reason = None
        start_xyz = end_xyz = None
        ring = None

        if ceiling <= 0.0:
            reason = UNAVAILABLE_REASON.get(key, "Not observable from the captured views")
        elif source == "top":
            samples = [s.get("raw_mm", {}).get(key) for s in top_samples]
            value = (top or {}).get("raw_mm", {}).get(key)
            rep, spread = repeatability(samples)
            conf = ceiling * top_q * rep
            z = 2.0 if key in ("foot_length", "arch_length", "heel_to_ball_length") else 15.0
            start_xyz, end_xyz = _lift(top_lm.get(start), z), _lift(top_lm.get(end), z)
            if key in ("foot_length", "arch_length", "heel_to_ball_length") and start_xyz and end_xyz:
                end_xyz = [start_xyz[0], end_xyz[1], z]
        elif source == "side":
            if side:
                samples = [s.get("raw_mm", {}).get(key) for s in side_samples]
                value = side.get("raw_mm", {}).get(key)
                rep, spread = repeatability(samples)
                conf = ceiling * min(side_q, top_q) * rep
                pt = side_lm.get(end)
                if pt:
                    mid_v = float(np.mean([top_lm.get("midfoot_a", [0, 0])[1], top_lm.get("midfoot_b", [0, 0])[1]]))
                    start_xyz = [round(mid_v, 1), round(pt[0], 1), 0.0]
                    end_xyz = [round(mid_v, 1), round(pt[0], 1), round(pt[1], 1)]
            else:
                reason = "Inner-side view missing"
        elif source == "girth":
            g = girths.get(key)
            if g:
                value = g["value_mm"]
                conf = ceiling * min(side_q, top_q)
                ring = g["ring"]
            else:
                reason = "3D shape not available"
        elif source == "3d":
            if mesh:
                value = float(mesh.get("volume_cm3", 0.0)) * 1000.0
                conf = ceiling * min(side_q, top_q)
            else:
                reason = "3D shape not available"

        if value is None and reason is None:
            reason = "Landmarks not found in this view"
        status = status_for(conf) if value is not None else "additional_scan_required"
        if status == "additional_scan_required" and reason is None:
            reason = "Confidence below 75% — additional scan required"
        if key == "foot_volume":
            display = f"{value / 1000.0:.0f} cm³" if value is not None else "—"
        else:
            display = f"{value / 10.0:.1f} cm" if value is not None else "Additional scan required"
        items.append(
            {
                "key": key,
                "number": number,
                "measurement": label,
                "view": source,
                "value_mm": round(float(value), 1) if value is not None else None,
                "display_value": display,
                "start_landmark": start,
                "end_landmark": end,
                "start_xyz": start_xyz,
                "end_xyz": end_xyz,
                "ring": ring,
                "confidence": round(float(conf), 3),
                "status": status,
                "color": color_for(status),
                "repeatability_spread_mm": round(spread, 2),
                "reason": reason,
            }
        )

    reported = [m for m in items if m["value_mm"] is not None]
    overall = float(np.mean([m["confidence"] for m in reported])) if reported else 0.0
    data = {"side": side_name, "measurements": items, "view_quality": {"top": round(top_q, 3), "side": round(side_q, 3)}}
    return AgentOutput(AGENT, bool(reported), overall, data)
