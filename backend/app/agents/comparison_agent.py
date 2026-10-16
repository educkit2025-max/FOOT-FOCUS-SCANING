"""Agent 8 — Left/right comparison. Each foot is measured independently; nothing is mirrored."""

from __future__ import annotations

from app.agents.base import AgentOutput, failed

AGENT = "comparison"


def run(left: list[dict], right: list[dict]) -> AgentOutput:
    lmap = {m["key"]: m for m in left}
    rmap = {m["key"]: m for m in right}
    rows = []
    for key, lm in lmap.items():
        rm = rmap.get(key)
        if rm is None:
            continue
        lv, rv = lm["value_mm"], rm["value_mm"]
        diff = pct = None
        if lv is not None and rv is not None:
            diff = rv - lv
            mean = (lv + rv) / 2.0
            pct = abs(diff) / mean * 100.0 if mean else None
        rows.append(
            {
                "key": key,
                "measurement": lm["measurement"],
                "left_mm": lv,
                "right_mm": rv,
                "difference_mm": round(diff, 1) if diff is not None else None,
                "difference_percent": round(pct, 1) if pct is not None else None,
                "larger": None if diff is None or abs(diff) < 0.5 else ("right" if diff > 0 else "left"),
            }
        )
    length = next((r for r in rows if r["key"] == "foot_length"), None)
    if not length or length["difference_mm"] is None:
        return failed(AGENT, "LENGTH_MISSING", ["Both foot lengths are needed for comparison"], {"rows": rows})
    larger = "right" if length["difference_mm"] >= 0 else "left"
    data = {
        "rows": rows,
        "length_difference_mm": length["difference_mm"],
        "larger_foot": larger,
        "sizing_basis": larger,
        "equal_within_tolerance": abs(length["difference_mm"]) < 2.0,
    }
    return AgentOutput(AGENT, True, 1.0, data)
