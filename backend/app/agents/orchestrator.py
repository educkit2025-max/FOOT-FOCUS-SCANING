"""Runs the agents in order for one foot profile and records real stage progress."""

from __future__ import annotations

import logging
import statistics
import traceback
from pathlib import Path

from app.agents import (
    calibration_agent,
    comparison_agent,
    footwear_fit_agent,
    landmark_agent,
    measurement_agent,
    quality_agent,
    reconstruction_agent,
    segmentation_agent,
    visualization_agent,
)
from app.agents.base import AgentOutput
from app.agents.capture_guide_agent import gravity_angles
from app.agents.imaging import intrinsics_for, load_image_path, sharpness_score
from app.services import profile_storage as store

log = logging.getLogger(__name__)

STAGES = [
    ("analyzing_images", "Analyzing images"),
    ("calibration", "Calibrating with A4 paper"),
    ("segmentation", "Separating feet from background"),
    ("landmarks", "Locating foot landmarks"),
    ("reconstruction", "Building 3D foot shape"),
    ("measurement", "Measuring both feet"),
    ("comparison", "Comparing left and right"),
    ("quality_control", "Checking quality"),
    ("visualization", "Preparing 3D view"),
    ("footwear_fit", "Calculating shoe fit"),
]
SIDES = ("left", "right")


class _Progress:
    def __init__(self, profile_id: str) -> None:
        self.pid = profile_id
        self.states = {k: "pending" for k, _ in STAGES}

    def enter(self, stage: str) -> None:
        for k, v in self.states.items():
            if v == "running":
                self.states[k] = "done"
        self.states[stage] = "running"
        self._write("processing", stage)

    def finish(self, state: str, error: str | None = None) -> None:
        for k, v in self.states.items():
            if v == "running":
                self.states[k] = "done" if error is None else "failed"
        self._write(state, None, error)

    def _write(self, state: str, stage: str | None, error: str | None = None) -> None:
        done = sum(1 for v in self.states.values() if v == "done")
        store.write_status(
            self.pid,
            {
                "state": state,
                "stage": stage,
                "progress": round(done / len(STAGES), 3),
                "stages": [{"id": k, "label": label, "state": self.states[k]} for k, label in STAGES],
                "error": error,
            },
        )


def _gravity(meta: dict | None):
    if not meta:
        return None
    g = [meta.get("gravity_x"), meta.get("gravity_y"), meta.get("gravity_z")]
    return tuple(float(v) for v in g) if all(v is not None for v in g) else None


def top_sample(path, side: str, meta: dict, debug_path: Path | None = None) -> dict:
    img = load_image_path(path)
    intr = intrinsics_for(img.shape, meta.get("focal_mm"), meta.get("sensor_long_mm"))
    rec: dict = {"capture": path.name, "sharpness": round(sharpness_score(img), 1), "intrinsics": intr.source}
    cal = calibration_agent.run(img, intr)
    if not cal.ok:
        return {**rec, "error": cal.error, "message": "; ".join(cal.messages)}
    rec.update(
        {
            "tilt_deg": cal.data["tilt_deg"],
            "camera_height_mm": cal.data["camera_height_mm"],
            "calibration_confidence": cal.data["calibration_confidence"],
            "scale_valid": cal.data["scale_valid"],
            "paper_cut_off": cal.data["paper_cut_off"],
            "calibration_messages": cal.messages,
        }
    )
    paper = cal.artifacts["paper"]
    rect = calibration_agent.rectify(img, paper, px_per_mm=2.0, margin_mm=30.0)
    seg = segmentation_agent.segment_top(rect)
    if not seg.ok:
        return {**rec, "error": seg.error, "message": "; ".join(seg.messages)}
    rec.update(
        {
            "segmentation_confidence": seg.data["segmentation_confidence"],
            "edge_contact": seg.data["edge_contact"],
            "toe_contact_mm": seg.data["toe_contact_mm"],
            "segmentation_messages": seg.messages,
        }
    )
    if debug_path is not None:
        visualization_agent.draw_top_debug(rect.image, seg.artifacts["mask"], debug_path)
    lm = landmark_agent.locate_top(seg.artifacts["mask"], seg.artifacts["contour"], rect, paper, side)
    if not lm.ok:
        return {**rec, "error": lm.error, "message": "; ".join(lm.messages)}
    rec.update({k: lm.data[k] for k in ("detected_side", "side_confidence", "landmarks_mm", "raw_mm", "outline_uv", "flags")})
    rec["landmark_confidence"] = lm.data["landmark_confidence"]
    rec["landmark_messages"] = lm.messages
    return rec


def side_sample(path, side: str, meta: dict, top_length: float | None) -> dict:
    img = load_image_path(path)
    rec: dict = {"capture": path.name, "sharpness": round(sharpness_score(img), 1)}
    grav = gravity_angles(_gravity(meta))
    rec["tilt_deg"] = round(abs(grav["elevation_deg"]), 1) if grav else 0.0
    rec["tilt_source"] = "device_gravity" if grav else "not_reported"
    seg = segmentation_agent.segment_side(img)
    if not seg.ok:
        return {**rec, "error": seg.error, "message": "; ".join(seg.messages)}
    rec["segmentation_confidence"] = seg.data["segmentation_confidence"]
    lm = landmark_agent.locate_side(seg.artifacts["mask"], side, top_length)
    if not lm.ok:
        return {**rec, "error": lm.error, "message": "; ".join(lm.messages)}
    rec.update({k: lm.data[k] for k in ("roll_deg", "wrong_direction", "raw_mm", "landmarks_mm", "profile_uh")})
    rec["landmark_confidence"] = lm.data["landmark_confidence"]
    return rec


def aggregate(samples: list[dict], length_key: str) -> dict | None:
    """Median of every raw value across the burst; the representative sample is the median-length one."""
    good = [s for s in samples if not s.get("error")]
    if not good:
        return dict(samples[0]) if samples else None
    good.sort(key=lambda s: s["raw_mm"].get(length_key) or 0.0)
    rep = dict(good[len(good) // 2])
    raw = {}
    for key in rep["raw_mm"]:
        vals = [s["raw_mm"].get(key) for s in good if s["raw_mm"].get(key) is not None]
        raw[key] = round(statistics.median(vals), 2) if vals else None
    rep["raw_mm"] = raw
    lengths = [s["raw_mm"].get(length_key) for s in good if s["raw_mm"].get(length_key) is not None]
    rep["length_spread_mm"] = round(max(lengths) - min(lengths), 2) if len(lengths) > 1 else 0.0
    rep["samples_used"] = len(good)
    rep["samples_total"] = len(samples)
    return rep


def light_record(record: dict | None) -> dict | None:
    if record is None:
        return None
    return {k: v for k, v in record.items() if k not in ("outline_uv", "profile_uh", "landmarks_mm")}


def process_profile(profile_id: str) -> None:
    prog = _Progress(profile_id)
    try:
        _run(profile_id, prog)
    except Exception as exc:  # noqa: BLE001
        log.exception("profile %s failed", profile_id)
        store.write_result(profile_id, {"profile_id": profile_id, "status": "failed", "error": str(exc), "trace": traceback.format_exc()})
        prog.finish("failed", str(exc))


def _run(pid: str, prog: _Progress) -> None:
    agents: list[AgentOutput] = []
    prog.enter("analyzing_images")
    metas = {v: store.view_meta(pid, v) or {} for v in store.ALL_VIEWS}

    prog.enter("calibration")
    top_samples = {s: [] for s in SIDES}
    for side in SIDES:
        view = f"{side}_top"
        for i, p in enumerate(store.view_captures(pid, view)):
            debug = store.debug_dir(pid) / f"{view}_rectified.jpg" if i == 0 else None
            top_samples[side].append(top_sample(p, side, metas[view], debug))

    prog.enter("segmentation")
    top = {s: aggregate(top_samples[s], "foot_length") for s in SIDES}
    for side in SIDES:
        rec = top[side]
        if rec and not rec.get("error"):
            votes = [s.get("detected_side") for s in top_samples[side] if not s.get("error") and s.get("side_confidence", 0) >= 0.5]
            if votes and votes.count(side) < len(votes) / 2:
                rec["side_mismatch"] = f"This looks like the {votes[0]} foot — capture the {side} foot"

    prog.enter("landmarks")
    side_samples = {s: [] for s in SIDES}
    for side in SIDES:
        view = f"{side}_medial"
        L = (top[side] or {}).get("raw_mm", {}).get("foot_length") if top[side] else None
        for p in store.view_captures(pid, view):
            side_samples[side].append(side_sample(p, side, metas[view], L))
    med = {s: aggregate(side_samples[s], "instep_height") for s in SIDES}

    prog.enter("reconstruction")
    meshes: dict[str, dict | None] = {}
    for side in SIDES:
        t, m = top[side], med[side]
        if t and not t.get("error") and m and not m.get("error"):
            rec = reconstruction_agent.run(t, m, side)
            agents.append(rec)
            if rec.ok:
                meshes[side] = rec.data
                store.write_mesh(pid, side, rec.data, reconstruction_agent.to_obj(rec.data))
                continue
        meshes[side] = None

    prog.enter("measurement")
    per_side = {}
    for side in SIDES:
        t = top[side] if top[side] and not top[side].get("error") else None
        m = med[side] if med[side] and not med[side].get("error") else None
        out = measurement_agent.run(
            side,
            t,
            [s for s in top_samples[side] if not s.get("error")],
            m,
            [s for s in side_samples[side] if not s.get("error")],
            meshes[side],
        )
        agents.append(out)
        raw = {**((t or {}).get("raw_mm", {})), **((m or {}).get("raw_mm", {}))}
        per_side[side] = {"measurements": out.data["measurements"], "raw": raw, "view_quality": out.data["view_quality"]}

    prog.enter("comparison")
    comp = comparison_agent.run(per_side["left"]["measurements"], per_side["right"]["measurements"])
    agents.append(comp)

    prog.enter("quality_control")
    views = {f"{s}_top": top[s] for s in SIDES} | {f"{s}_medial": med[s] for s in SIDES}
    qc = quality_agent.run(views, per_side, comp.data if comp.ok else None)
    agents.append(qc)

    prog.enter("visualization")
    vis = {}
    for side in SIDES:
        t = top[side] if top[side] and not top[side].get("error") else None
        if t is None:
            vis[side] = None
            continue
        m = med[side] if med[side] and not med[side].get("error") else None
        v = visualization_agent.run(side, t, m, per_side[side]["measurements"], meshes[side])
        agents.append(v)
        vis[side] = v.data

    prog.enter("footwear_fit")
    sizing_side = comp.data.get("larger_foot") if comp.ok else None
    if sizing_side is None:
        measured = [s for s in SIDES if any(m["key"] == "foot_length" and m["value_mm"] is not None for m in per_side[s]["measurements"])]
        sizing_side = measured[0] if len(measured) == 1 else None
    fit = footwear_fit_agent.run(sizing_side, per_side[sizing_side]["measurements"]) if sizing_side else None
    if fit:
        agents.append(fit)

    def val(side, key):
        return next((m["value_mm"] for m in per_side[side]["measurements"] if m["key"] == key), None)

    profile = {"units": "mm"}
    for side in SIDES:
        for m in per_side[side]["measurements"]:
            profile[f"{side}_{m['key']}"] = m["value_mm"]
    reported = [
        m["confidence"]
        for s in SIDES
        for m in per_side[s]["measurements"]
        if m["key"] in quality_agent.ESSENTIAL and m["value_mm"] is not None
    ]
    profile["larger_foot"] = sizing_side
    profile["scan_confidence"] = round(min(reported), 3) if reported else 0.0
    profile["measurement_confidences"] = {s: {m["key"]: m["confidence"] for m in per_side[s]["measurements"]} for s in SIDES}

    def cm(v):
        return round(v / 10.0, 1) if v is not None else None

    result = {
        "profile_id": pid,
        "status": "complete" if qc.data["accepted"] else "needs_retake",
        "accepted": qc.data["accepted"],
        "retake": qc.data["retake"],
        "warnings": qc.data["warnings"],
        "summary": {
            "left_length_cm": cm(val("left", "foot_length")),
            "left_width_cm": cm(val("left", "max_foot_width")),
            "right_length_cm": cm(val("right", "foot_length")),
            "right_width_cm": cm(val("right", "max_foot_width")),
            "larger_foot": sizing_side,
            "sizing_basis": sizing_side,
            "scan_confidence": profile["scan_confidence"],
        },
        "foot_profile": profile,
        "feet": {
            s: {
                "measurements": per_side[s]["measurements"],
                "view_quality": per_side[s]["view_quality"],
                "visualization": vis[s],
                "mesh_available": meshes[s] is not None,
                "volume_cm3": meshes[s]["volume_cm3"] if meshes[s] else None,
            }
            for s in SIDES
        },
        "comparison": comp.data,
        "footwear": fit.data if fit and fit.ok else None,
        "views": {k: light_record(v) for k, v in views.items()},
        "full_3d": reconstruction_agent.full_3d_status(),
        "agents": [a.summary() for a in agents],
    }
    store.write_result(pid, result)
    prog.finish("complete" if qc.data["accepted"] else "needs_retake")
