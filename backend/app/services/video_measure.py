"""Rotating-video scan of one foot: best-frame picking and agent-based measurement.

Step 1 (`analyze_frames`) runs right after frame selection and picks the 5 best frames from
different positions. Step 2 (`measure_scan`) runs the agents on the selected frames.
"""

from __future__ import annotations

import json
import logging
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.agents import (
    calibration_agent,
    focus_agent,
    footwear_fit_agent,
    measurement_agent,
    orchestrator,
    quality_agent,
    reconstruction_agent,
    segmentation_agent,
    visualization_agent,
)
from app.agents.imaging import downscale, intrinsics_for, sharpness_score
from app.models.schemas import ScanStatus
from app.services import storage

log = logging.getLogger(__name__)

CANDIDATE_FPS = 8.0
GUIDED_SECONDS = 30.0
# (id, label, start s, end s) of the on-screen guidance schedule during the 30 s scan.
POSITIONS = [
    ("top", "Top view", 0.0, 5.0),
    ("inner", "Inner side", 5.0, 10.0),
    ("heel", "Heel", 10.0, 16.0),
    ("outer", "Outer side", 16.0, 22.0),
    ("front", "Front / top", 22.0, 30.0),
]
MAX_TOP_FRAMES = 24
MAX_SIDE_FRAMES = 8
TOP_TILT_MAX = 20.0

STAGES = [
    ("analyzing_frames", "Analyzing selected frames"),
    ("calibration", "Calibrating with A4 paper"),
    ("segmentation", "Separating foot from background"),
    ("landmarks", "Locating foot landmarks"),
    ("reconstruction", "Building 3D foot shape"),
    ("colour_3d", "Building colour 3D foot"),
    ("measurement", "Measuring the foot"),
    ("quality_control", "Checking quality"),
    ("visualization", "Preparing 3D view"),
    ("footwear_fit", "Calculating shoe fit"),
]


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _position_for(t: float, duration: float) -> str:
    scaled = t * GUIDED_SECONDS / max(duration, 1.0)
    for pid, _, lo, hi in POSITIONS:
        if lo <= scaled < hi:
            return pid
    return POSITIONS[-1][0]


def analyze_frames(scan_id: str, report: dict) -> list[dict]:
    """Score every selected frame (sharpness, foot presence, A4 pose) and pick the best per position."""
    root = storage.scan_dir(scan_id)
    duration = report.get("original_candidate_count", 0) / CANDIDATE_FPS
    records = []
    for f in report.get("frames", []):
        path = root / "selected" / f["filename"]
        img = cv2.imread(str(path))
        if img is None:
            continue
        t = f.get("source_index", f["index"]) / CANDIDATE_FPS
        small, _ = downscale(img, 320)
        presence = float((segmentation_agent.foot_seed_mask(small) > 0).mean())
        cal = calibration_agent.run(img, intrinsics_for(img.shape))
        records.append(
            {
                "filename": f["filename"],
                "thumbnail": f["thumbnail"],
                "time_s": round(t, 2),
                "position": _position_for(t, duration),
                "sharpness": round(sharpness_score(img), 1),
                "foot_fraction": round(presence, 3),
                "a4_found": cal.ok,
                "a4_valid": bool(cal.ok and cal.data["scale_valid"]),
                "tilt_deg": cal.data.get("tilt_deg") if cal.ok else None,
            }
        )
    _write_json(root / "frame_analysis.json", {"duration_s": round(duration, 1), "frames": records})

    best_dir = root / "best"
    best_dir.mkdir(exist_ok=True)
    max_sharp = max((r["sharpness"] for r in records), default=1.0) or 1.0
    used: set[str] = set()
    best = []

    def score(r: dict, pos: str) -> float:
        s = r["sharpness"] / max_sharp
        s *= 1.0 if 0.03 <= r["foot_fraction"] <= 0.6 else 0.3
        if pos in ("top", "front") and r["a4_valid"] and (r["tilt_deg"] or 90) <= TOP_TILT_MAX:
            s *= 1.4
        return s

    for k, (pos, label, _, _) in enumerate(POSITIONS):
        pool = [r for r in records if r["position"] == pos and r["filename"] not in used]
        if not pool:
            pool = [r for r in records if r["filename"] not in used]
        if not pool:
            break
        r = max(pool, key=lambda x: score(x, pos))
        used.add(r["filename"])
        img = cv2.imread(str(root / "selected" / r["filename"]))
        preview, _ = downscale(img, 720)
        name = f"best_{k}.jpg"
        cv2.imwrite(str(best_dir / name), preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
        best.append(
            {
                "rank": k + 1,
                "position": pos,
                "label": label,
                "image_url": f"/api/v1/scans/{scan_id}/best/{name}",
                "time_s": r["time_s"],
                "sharpness": r["sharpness"],
                "a4_detected": r["a4_valid"],
                "tilt_deg": r["tilt_deg"],
                "quality_score": int(round(100 * min(1.0, score(r, pos)))),
            }
        )
    _write_json(root / "best_frames.json", best)
    return best


class _Stages:
    def __init__(self, scan_id: str) -> None:
        self.scan_id = scan_id
        self.states = {k: "pending" for k, _ in STAGES}
        self.current = STAGES[0][0]

    def enter(self, stage: str) -> None:
        for k, v in self.states.items():
            if v == "running":
                self.states[k] = "done"
        self.states[stage] = "running"
        self.current = stage
        self.detail(0.0, dict(STAGES)[stage])

    def detail(self, fraction: float, message: str) -> None:
        """Progress inside the running stage, for the long colour-3D step."""
        done = sum(v == "done" for v in self.states.values())
        storage.set_status(
            self.scan_id,
            ScanStatus.processing,
            progress=int(100 * (done + min(max(fraction, 0.0), 0.99)) / len(STAGES)),
            stage=self.current,
            message=message,
            extra={"stages": self.payload()},
        )

    def close(self, failed: bool = False) -> None:
        for k, v in self.states.items():
            if v == "running":
                self.states[k] = "failed" if failed else "done"

    def payload(self) -> list[dict]:
        return [{"id": k, "label": label, "state": self.states[k]} for k, label in STAGES]


def _trim(samples: list[dict], key: str) -> list[dict]:
    """Drop the extreme 10% on each side so a few mis-segmented frames cannot skew the median."""
    if len(samples) < 5:
        return samples
    vals = np.array([s["raw_mm"].get(key) or 0.0 for s in samples])
    lo, hi = np.percentile(vals, [10, 90])
    return [s for s, v in zip(samples, vals) if lo <= v <= hi]


def _failure_record(attempts: list[dict], had_candidates: bool, kind: str) -> dict:
    if kind == "top" and not had_candidates:
        return {
            "error": "CALIBRATION_NOT_FOUND",
            "message": "A4 sheet not detected from above — put the A4 on a darker floor and keep all 4 corners visible",
        }
    if kind == "side" and not had_candidates:
        return {"error": "SIDE_NOT_FOUND", "message": "No clear side view — move slowly around the inner and outer side"}
    errors = Counter(a.get("message") or a.get("error") for a in attempts if a.get("error"))
    msg = errors.most_common(1)[0][0] if errors else "Foot not measurable in these frames"
    return {"error": "NOT_MEASURABLE", "message": msg}


def measure_scan(scan_id: str) -> dict:
    stages = _Stages(scan_id)
    try:
        result = _measure(scan_id, stages)
        stages.close()
        storage.set_status(
            scan_id,
            ScanStatus.complete,
            progress=100,
            stage="complete",
            message="Foot profile ready" if result["accepted"] else "Rescan recommended",
            extra={"stages": stages.payload(), "measurement_summary": result["summary"]},
        )
        return result
    except Exception as exc:  # noqa: BLE001
        log.exception("measurement failed for %s", scan_id)
        stages.close(failed=True)
        _write_json(storage.scan_dir(scan_id) / "result_v2.json", {"scan_id": scan_id, "status": "failed", "error": str(exc), "trace": traceback.format_exc()})
        storage.set_status(
            scan_id,
            ScanStatus.failed,
            progress=100,
            stage="failed",
            message=str(exc),
            error_code="MEASUREMENT_FAILED",
            extra={"stages": stages.payload()},
        )
        raise


def _measure(scan_id: str, stages: _Stages) -> dict:
    root = storage.scan_dir(scan_id)
    meta = storage.load_meta(scan_id)
    foot = meta.get("foot") or "right"
    device = meta.get("device") or {}
    cam_meta = {"focal_mm": device.get("focal_length_mm"), "sensor_long_mm": device.get("sensor_long_mm")}

    stages.enter("analyzing_frames")
    analysis = _read_json(root / "frame_analysis.json")
    if analysis is None:
        report = _read_json(root / "frame_selection.json") or {}
        analyze_frames(scan_id, report)
        analysis = _read_json(root / "frame_analysis.json") or {"frames": []}
    frames = analysis["frames"]

    stages.enter("calibration")
    top_cands = [f for f in frames if f["a4_valid"] and (f["tilt_deg"] or 90.0) <= TOP_TILT_MAX]
    top_cands = sorted(top_cands, key=lambda f: f["sharpness"], reverse=True)[:MAX_TOP_FRAMES]

    stages.enter("segmentation")
    top_attempts = []
    for i, f in enumerate(top_cands):
        debug = root / "debug_overlays" / "top_rectified.jpg" if i == 0 else None
        top_attempts.append(orchestrator.top_sample(root / "selected" / f["filename"], foot, cam_meta, debug))
    top_ok = [s for s in top_attempts if not s.get("error")]
    level = [s for s in top_ok if s.get("tilt_deg", 90.0) <= 15.0]
    if len(level) >= 3:
        top_ok = level
    top_ok = _trim(top_ok, "foot_length")
    top = orchestrator.aggregate(top_ok, "foot_length") if top_ok else _failure_record(top_attempts, bool(top_cands), "top")

    stages.enter("landmarks")
    L = top["raw_mm"]["foot_length"] if not top.get("error") else None
    side_cands = [
        f
        for f in frames
        if f["position"] in ("inner", "outer")
        and not (f["a4_valid"] and (f["tilt_deg"] or 90.0) <= 45.0)
        and 0.03 <= f["foot_fraction"] <= 0.6
    ]
    side_cands = sorted(side_cands, key=lambda f: f["sharpness"], reverse=True)[:MAX_SIDE_FRAMES]
    side_attempts = [orchestrator.side_sample(root / "selected" / f["filename"], foot, {}, L) for f in side_cands] if L else []
    side_ok = []
    for s in side_attempts:
        if s.get("error"):
            continue
        ih = s["raw_mm"].get("instep_height")
        if ih is None or not 0.15 * L <= ih <= 0.36 * L:
            continue
        # A rotating video sees both sides; profile height does not depend on which one.
        s["wrong_direction"] = False
        side_ok.append(s)
    side_ok = _trim(side_ok, "instep_height")
    if side_ok:
        side = orchestrator.aggregate(side_ok, "instep_height")
    elif L is None:
        side = {"error": "NO_SCALE", "message": "Side profile needs a calibrated top view first"}
    else:
        side = _failure_record(side_attempts, bool(side_cands), "side")

    stages.enter("reconstruction")
    top_rec = top if not top.get("error") else None
    side_rec = side if not side.get("error") else None
    mesh = None
    if top_rec and side_rec:
        rec = reconstruction_agent.run(top_rec, side_rec, foot)
        if rec.ok:
            mesh = rec.data
            _write_json(root / "mesh_hull.json", mesh)

    stages.enter("colour_3d")
    display_mesh = mesh
    footprint = mesh
    if footprint is None and top_rec and top_rec.get("outline_uv"):
        # Without the hull, line the colour mesh up with the A4 top outline the measurement lines use.
        rows = top_rec["outline_uv"]
        footprint = {"vertices": [[r[1], r[0], 0.0] for r in rows] + [[r[2], r[0], 0.0] for r in rows]}
    colour = focus_agent.run(
        root,
        frames,
        foot,
        L,
        footprint,
        cam_meta["focal_mm"],
        cam_meta["sensor_long_mm"],
        progress=lambda i, n: stages.detail(0.7 * i / n, f"Building colour 3D foot — frame {i} of {n}"),
    )
    if colour.ok:
        display_mesh = colour.data
        (root / "mesh.obj").write_text(focus_agent.to_obj(display_mesh), encoding="utf-8")
    elif mesh:
        (root / "mesh.obj").write_text(reconstruction_agent.to_obj(mesh), encoding="utf-8")
    if display_mesh:
        _write_json(root / "mesh.json", display_mesh)

    stages.enter("measurement")
    meas = measurement_agent.run(foot, top_rec, top_ok, side_rec, side_ok, mesh)
    items = meas.data["measurements"]
    if top_rec is None:
        for m in items:
            if m["value_mm"] is None and m["view"] in ("top", "side", "girth", "3d"):
                m["reason"] = top["message"]

    stages.enter("quality_control")
    raw = {**((top_rec or {}).get("raw_mm", {})), **((side_rec or {}).get("raw_mm", {}))}
    qc = quality_agent.run_single(foot, top, side, {"measurements": items, "raw": raw})

    stages.enter("visualization")
    vis = visualization_agent.run(foot, top_rec, side_rec, items, mesh).data if top_rec else None

    stages.enter("footwear_fit")
    fit = footwear_fit_agent.run(foot, items)

    def value(key):
        return next((m for m in items if m["key"] == key), {})

    length, width = value("foot_length"), value("max_foot_width")
    essentials = [m["confidence"] for m in items if m["key"] in quality_agent.ESSENTIAL and m["value_mm"] is not None]
    result = {
        "scan_id": scan_id,
        "foot": foot,
        "status": "complete" if qc.data["accepted"] else "needs_rescan",
        "accepted": qc.data["accepted"],
        "retake": qc.data["retake"],
        "warnings": qc.data["warnings"],
        "summary": {
            "length_cm": round(length["value_mm"] / 10, 1) if length.get("value_mm") else None,
            "width_cm": round(width["value_mm"] / 10, 1) if width.get("value_mm") else None,
            "length_status": length.get("status"),
            "width_status": width.get("status"),
            "scan_confidence": round(min(essentials), 3) if essentials else 0.0,
        },
        "foot_result": {
            "measurements": items,
            "visualization": vis,
            "mesh_available": display_mesh is not None,
            "mesh_method": display_mesh["method"] if display_mesh else None,
            "mesh_scale": display_mesh.get("scale_source", "a4_calibrated") if display_mesh else None,
            "mesh_messages": colour.messages,
            "volume_cm3": mesh["volume_cm3"] if mesh else None,
        },
        "footwear": fit.data if fit.ok else None,
        "frames_used": {"top": len(top_ok), "top_tried": len(top_cands), "side": len(side_ok), "side_tried": len(side_cands)},
        "views": {"top": orchestrator.light_record(top), "side": orchestrator.light_record(side)},
        "best_frames": _read_json(root / "best_frames.json") or [],
        "full_3d": reconstruction_agent.full_3d_status(),
    }
    _write_json(root / "result_v2.json", result)
    return result
