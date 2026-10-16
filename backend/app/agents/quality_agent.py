"""Agent 7 — Quality control: accept or reject each view and the final profile.

A rejection names only the views that must be recaptured, with the reason for each.
"""

from __future__ import annotations

from app.agents.base import AgentOutput
from app.agents.imaging import SHARP_MIN

AGENT = "quality_control"

ESSENTIAL = {"foot_length": "final", "max_foot_width": "final", "instep_height": "estimated"}
ESSENTIAL_VIEW = {"foot_length": "top", "max_foot_width": "top", "instep_height": "medial"}
RANK = {"additional_scan_required": 0, "estimated": 1, "final": 2}


def _add(retakes: dict, view: str, reason: str) -> None:
    retakes.setdefault(view, [])
    if reason not in retakes[view]:
        retakes[view].append(reason)


def check_view(view_id: str, kind: str, record: dict | None, retakes: dict, warnings: list) -> None:
    if record is None:
        _add(retakes, view_id, "View missing")
        return
    if record.get("error"):
        _add(retakes, view_id, record.get("message") or record["error"])
        return
    if record.get("sharpness", 100.0) < SHARP_MIN:
        _add(retakes, view_id, "Image blurred — hold the phone still")
    tilt = float(record.get("tilt_deg", 0.0))
    if tilt > 15.0:
        _add(retakes, view_id, f"Camera tilt: {tilt:.0f}° — reduce angle")
    elif tilt > 10.0:
        warnings.append(f"{view_id}: camera tilt {tilt:.0f}° (acceptable)")
    if kind == "top":
        if not record.get("scale_valid", False):
            _add(retakes, view_id, "A4 paper not fully visible")
        if record.get("edge_contact", {}).get("toe_edge") and record.get("toe_contact_mm", 0) > 8:
            _add(retakes, view_id, "Toes outside the paper")
        if record.get("side_mismatch"):
            _add(retakes, view_id, record["side_mismatch"])
        if record.get("flags", {}).get("leg_intrusion_suspected"):
            warnings.append(f"{view_id}: leg may overlap the heel outline")
        spread = record.get("length_spread_mm", 0.0)
        if spread > 4.0:
            _add(retakes, view_id, f"Length varied {spread:.1f} mm between photos — keep the foot still")
    else:
        if record.get("wrong_direction"):
            _add(retakes, view_id, "Wrong side — show the inner side of the foot")
        if abs(record.get("roll_deg", 0.0)) > 8.0:
            _add(retakes, view_id, "Camera not level — hold the phone straight")


def plausibility(side: str, raw: dict, retakes: dict, top_view: str, side_view: str) -> None:
    L = raw.get("foot_length")
    if not L:
        return
    if not 150.0 <= L <= 340.0:
        _add(retakes, top_view, f"Foot length {L / 10:.1f} cm is outside the expected range")
        return
    w = raw.get("max_foot_width")
    if w and not 0.30 <= w / L <= 0.48:
        _add(retakes, top_view, "Width/length proportion implausible — check that only the foot is on the paper")
    hw = raw.get("heel_width")
    if w and hw and not 0.45 <= hw / w <= 0.90:
        _add(retakes, top_view, "Heel outline unclear — keep the heel fully on the paper")
    ih = raw.get("instep_height")
    if ih and not 0.15 <= ih / L <= 0.36:
        _add(retakes, side_view, "Instep profile implausible — retake the inner-side view")


def _evaluate_side(side: str, top_id: str, med_id: str, views: dict, info: dict, retakes: dict, warnings: list) -> None:
    check_view(top_id, "top", views.get(top_id), retakes, warnings)
    check_view(med_id, "side", views.get(med_id), retakes, warnings)
    plausibility(side, info.get("raw", {}), retakes, top_id, med_id)
    by_key = {m["key"]: m for m in info.get("measurements", [])}
    for key, needed in ESSENTIAL.items():
        m = by_key.get(key)
        if m is None or m["value_mm"] is None or RANK[m["status"]] < RANK[needed]:
            view_id = top_id if ESSENTIAL_VIEW[key] == "top" else med_id
            if (views.get(view_id) or {}).get("error"):
                continue
            label = m["measurement"] if m else key
            conf = f" ({m['confidence'] * 100:.0f}%)" if m and m["value_mm"] is not None else ""
            _add(retakes, view_id, f"{label} confidence too low{conf}")


def run_single(side: str, top: dict | None, side_view: dict | None, info: dict) -> AgentOutput:
    """One foot from a rotating video: 'top' are the overhead frames, 'side' the profile frames."""
    retakes: dict[str, list[str]] = {}
    warnings: list[str] = []
    _evaluate_side(side, "top", "side", {"top": top, "side": side_view}, info, retakes, warnings)
    data = {"accepted": not retakes, "retake": [{"view": v, "reasons": r} for v, r in retakes.items()], "warnings": warnings}
    return AgentOutput(AGENT, True, 1.0 if not retakes else 0.5, data, messages=warnings)


def run(views: dict, per_side: dict, comparison: dict | None) -> AgentOutput:
    """views: view_id → record; per_side: side → {"measurements": [...], "raw": {...}}."""
    retakes: dict[str, list[str]] = {}
    warnings: list[str] = []
    for side in ("left", "right"):
        _evaluate_side(side, f"{side}_top", f"{side}_medial", views, per_side.get(side) or {}, retakes, warnings)

    if comparison:
        d = comparison.get("length_difference_mm")
        if d is not None and abs(d) > 15.0:
            warnings.append(f"Left and right length differ by {abs(d) / 10:.1f} cm — please confirm both top views")

    accepted = not retakes
    data = {
        "accepted": accepted,
        "retake": [{"view": v, "reasons": r} for v, r in retakes.items()],
        "warnings": warnings,
    }
    return AgentOutput(AGENT, True, 1.0 if accepted else 0.5, data, messages=warnings)
