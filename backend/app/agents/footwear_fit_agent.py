"""Agent 10 — Footwear fit: sizing from the larger foot's calibrated measurements."""

from __future__ import annotations

import math

from app.agents.base import AgentOutput, failed

AGENT = "footwear_fit"
TOE_ALLOWANCE_CM = 1.5


def eu_size(length_mm: float) -> float:
    """Paris point (2/3 cm) on foot length plus toe allowance, rounded up to the next half size."""
    raw = 1.5 * (length_mm / 10.0 + TOE_ALLOWANCE_CM)
    return math.ceil(raw * 2.0) / 2.0


def mondopoint(length_mm: float) -> int:
    return int(5 * round(length_mm / 5.0))


def width_category(width_mm: float, length_mm: float) -> str:
    r = width_mm / length_mm
    if r < 0.355:
        return "narrow"
    if r < 0.395:
        return "regular"
    if r < 0.425:
        return "wide"
    return "extra wide"


def instep_category(instep_mm: float | None, length_mm: float) -> str | None:
    if instep_mm is None:
        return None
    r = instep_mm / length_mm
    if r < 0.22:
        return "low"
    if r < 0.27:
        return "medium"
    return "high"


def heel_category(heel_mm: float | None, ball_mm: float | None) -> str | None:
    if not heel_mm or not ball_mm:
        return None
    r = heel_mm / ball_mm
    if r < 0.62:
        return "narrow heel"
    if r < 0.72:
        return "regular heel"
    return "broad heel"


def run(side: str, measurements: list[dict]) -> AgentOutput:
    by = {m["key"]: m for m in measurements}
    length = by.get("foot_length", {})
    width = by.get("max_foot_width", {})
    if length.get("value_mm") is None or width.get("value_mm") is None:
        return failed(AGENT, "SIZING_INPUTS_MISSING", ["Foot length and width are required for sizing"])
    L, W = length["value_mm"], width["value_mm"]
    instep = by.get("instep_height", {}).get("value_mm")
    provisional = length["status"] != "final" or width["status"] != "final"
    notes = []
    if provisional:
        notes.append("Provisional — length or width is estimated, rescan for a final size")
    data = {
        "sizing_foot": side,
        "eu_size": eu_size(L),
        "mondopoint_mm": mondopoint(L),
        "width_category": width_category(W, L),
        "instep_category": instep_category(instep, L),
        "heel_category": heel_category(by.get("heel_width", {}).get("value_mm"), by.get("ball_width", {}).get("value_mm")),
        "toe_allowance_cm": TOE_ALLOWANCE_CM,
        "provisional": provisional,
        "notes": notes,
    }
    conf = min(length["confidence"], width["confidence"])
    return AgentOutput(AGENT, True, conf, data, messages=notes)
