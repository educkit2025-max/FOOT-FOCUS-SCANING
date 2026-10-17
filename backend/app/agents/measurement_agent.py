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
    "arch_height": "Measured from the 3D foot",
    "heel_girth": "Measured from the 3D foot",
    "ankle_girth": "Measured from the 3D foot",
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


def _display(key: str, value: float | None) -> str:
    if value is None:
        return "—"
    if key == "foot_volume":
        return f"{int(round(value / 1000.0))} cm³"
    return f"{int(round(value / 10.0))} cm"


def _xyz(x: float, y: float, z: float) -> list[float]:
    return [round(float(x), 1), round(float(y), 1), round(float(z), 1)]


def _slice(verts: np.ndarray, y: float, band: float) -> np.ndarray:
    return verts[np.abs(verts[:, 1] - y) <= band]


def _width_at(verts: np.ndarray, y: float, band: float = 8.0) -> tuple[float, list, list] | None:
    sl = _slice(verts, y, band)
    if len(sl) < 4:
        return None
    xa, xb = float(sl[:, 0].min()), float(sl[:, 0].max())
    z = float(np.clip(np.percentile(sl[:, 2], 40), 4.0, 20.0))
    return xb - xa, _xyz(xa, y, z), _xyz(xb, y, z)


def _height_at(verts: np.ndarray, y: float, band: float = 10.0) -> tuple[float, list, list] | None:
    sl = _slice(verts, y, band)
    if len(sl) < 4:
        return None
    z = float(sl[:, 2].max())
    x = float(np.median(sl[:, 0]))
    return z, _xyz(x, y, 0.0), _xyz(x, y, z)


def _girth_at(verts: np.ndarray, y: float, band: float = 7.0) -> tuple[float, list] | None:
    from scipy.spatial import ConvexHull

    sl = _slice(verts, y, band)
    if len(sl) < 8:
        return None
    pts = sl[:, [0, 2]]
    try:
        hull = ConvexHull(pts)
    except Exception:  # noqa: BLE001
        return None
    ring = pts[hull.vertices]
    closed = np.vstack([ring, ring[0]])
    g = float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())
    pts3 = [_xyz(p[0], y, p[1]) for p in ring]
    return g, pts3


def _volume_mm3(verts: np.ndarray, faces: np.ndarray) -> float:
    vol = 0.0
    for a, b, c in faces:
        vol += float(np.dot(verts[a], np.cross(verts[b], verts[c])))
    return abs(vol) / 6.0


def fill_from_mesh(items: list[dict], mesh: dict) -> None:
    """Fill any missing measurement with a size taken from the colour 3D foot, in millimetres."""
    V = np.asarray(mesh.get("vertices") or [], np.float64)
    F = np.asarray(mesh.get("faces") or [], np.int64)
    if len(V) < 50:
        return
    y0, y1 = np.percentile(V[:, 1], [1.0, 99.0])
    L = float(y1 - y0)
    if L < 80:
        return
    x_mid = float(np.median(V[V[:, 2] < 20.0, 0])) if np.any(V[:, 2] < 20.0) else float(np.median(V[:, 0]))
    z_max = float(V[:, 2].max())
    y_len = np.linspace(y0 + 8.0, y1 - 8.0, 24)
    widths = [(_width_at(V, y) or (0.0, None, None))[0] for y in y_len]
    y_maxw = float(y_len[int(np.argmax(widths))]) if widths else y0 + 0.72 * L

    stations = {
        "max_foot_width": y_maxw,
        "ball_width": y0 + 0.72 * L,
        "forefoot_width": y0 + 0.78 * L,
        "heel_width": y0 + 0.18 * L,
        "midfoot_width": y0 + 0.50 * L,
        "toe_box_width": y0 + 0.85 * L,
        "toe_span": y0 + 0.93 * L,
        "instep_height": y0 + 0.50 * L,
        "arch_height": y0 + 0.40 * L,
        "hallux_height": y0 + 0.92 * L,
        "ball_girth": y0 + 0.72 * L,
        "forefoot_girth": y0 + 0.70 * L,
        "instep_girth": y0 + 0.50 * L,
        "heel_girth": y0 + 0.18 * L,
        "ankle_girth": y0 + 0.12 * L,
    }

    for m in items:
        if m.get("value_mm") is not None:
            m["display_value"] = _display(m["key"], m["value_mm"])
            continue
        key = m["key"]
        got = None
        if key == "foot_length":
            got = (L, _xyz(x_mid, y0, 2.0), _xyz(x_mid, y1, 2.0), None)
        elif key in ("arch_length", "heel_to_ball_length"):
            yb = stations["ball_width"]
            got = (yb - y0, _xyz(x_mid, y0, 2.0), _xyz(x_mid, yb, 2.0), None)
        elif key in ("max_foot_width", "ball_width", "forefoot_width", "heel_width", "midfoot_width", "toe_box_width", "toe_span"):
            w = _width_at(V, stations[key])
            if w:
                got = (w[0], w[1], w[2], None)
        elif key in ("instep_height", "hallux_height"):
            h = _height_at(V, stations[key])
            if h:
                got = (h[0], h[1], h[2], None)
        elif key == "arch_height":
            sl = _slice(V, stations[key], 12.0)
            if len(sl) >= 8:
                inner = sl[sl[:, 0] < x_mid] if np.any(sl[:, 0] < x_mid) else sl
                z = float(np.percentile(inner[:, 2], 25))
                got = (max(z, 4.0), _xyz(x_mid, stations[key], 0.0), _xyz(x_mid, stations[key], max(z, 4.0)), None)
        elif key in ("ball_girth", "forefoot_girth", "instep_girth", "heel_girth"):
            g = _girth_at(V, stations[key])
            if g:
                got = (g[0], None, None, g[1])
        elif key == "ankle_girth":
            high = V[V[:, 2] >= 0.65 * z_max]
            g = _girth_at(high if len(high) >= 8 else V, stations[key], band=12.0)
            if g:
                got = (g[0], None, None, g[1])
        elif key == "foot_volume" and len(F):
            got = (_volume_mm3(V, F), None, None, None)
        if not got:
            continue
        value, start, end, ring = got
        m["value_mm"] = round(float(value), 1)
        m["display_value"] = _display(key, value)
        if start:
            m["start_xyz"] = start
        if end:
            m["end_xyz"] = end
        if ring:
            m["ring"] = ring
        if m["confidence"] <= 0:
            m["confidence"] = 0.6
        m["status"] = status_for(m["confidence"])
        m["color"] = color_for(m["status"])
        m["reason"] = None

    for m in items:
        if m.get("value_mm") is not None:
            m["display_value"] = _display(m["key"], m["value_mm"])


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
        items.append(
            {
                "key": key,
                "number": number,
                "measurement": label,
                "view": source,
                "value_mm": round(float(value), 1) if value is not None else None,
                "display_value": _display(key, value),
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
