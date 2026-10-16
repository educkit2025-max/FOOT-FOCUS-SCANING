"""Agent 5b — colour 3D foot surface (FOCUS-SfM).

Builds the skin-coloured multi-view surface shown on the result screen. Measurements are not taken from
it: they stay with the A4-calibrated agents. The surface is scaled to the calibrated foot length and placed
in the same heel→toe frame so the measurement lines sit on it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from app.agents.base import AgentOutput, failed

AGENT = "colour_3d"
MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "focus_toc_predictor.pth"
MODEL_URL = "https://drive.google.com/file/d/1aU1Bf_pE7WjtWAX85ru9htQGygmTDIUV"
DEFAULT_FOCAL_FRACTION = 0.77


def status() -> dict:
    deps = {name: importlib.util.find_spec(name) is not None for name in ("torch", "geffnet", "pycolmap", "pymeshlab", "trimesh", "scipy")}
    deps["toc_model"] = MODEL_PATH.exists()
    return {"available": all(deps.values()), "method": "focus_sfm", "dependencies": deps, "model_url": MODEL_URL}


def _focal_px(focal_mm: float | None, sensor_long_mm: float | None) -> float:
    from app.agents.focus.sfm import FRAME_H

    if focal_mm and sensor_long_mm:
        return float(focal_mm) / float(sensor_long_mm) * FRAME_H
    return DEFAULT_FOCAL_FRACTION * FRAME_H


def run(scan_root: Path, frame_records: list[dict], side: str, length_mm: float | None, hull_mesh: dict | None,
        focal_mm: float | None = None, sensor_long_mm: float | None = None, progress=None) -> AgentOutput:
    st = status()
    if not st["available"]:
        missing = [k for k, ok in st["dependencies"].items() if not ok]
        return failed(AGENT, "FOCUS_NOT_INSTALLED", [f"Colour 3D needs: {', '.join(missing)}"])

    from app.agents.focus import sfm

    paths = sfm.pick_frames(frame_records, scan_root)
    if len(paths) < 6:
        return failed(AGENT, "TOO_FEW_FRAMES", [f"Colour 3D needs at least 6 frames around the foot, got {len(paths)}"])
    ref = np.asarray(hull_mesh["vertices"], np.float64) if hull_mesh else None
    try:
        res = sfm.run(paths, scan_root / "focus", MODEL_PATH, _focal_px(focal_mm, sensor_long_mm), length_mm, ref, progress)
    except Exception as exc:  # noqa: BLE001
        return failed(AGENT, "FOCUS_FAILED", [f"Colour 3D could not be built: {exc}"])

    stats = res["stats"]
    calibrated = res["scale_source"] == "a4_foot_length"
    coverage = stats["views_calibrated"] / max(stats["frames"], 1)
    msgs = [] if calibrated else ["3D shape is not to scale — A4 sheet was not measured from above"]
    if stats["views_calibrated"] < 10:
        msgs.append(
            f"3D shape is rough — only {stats['views_calibrated']} of {stats['views_with_foot']} views could be placed; "
            "move the phone slowly and keep the whole foot in view"
        )
    if stats["footedness_predicted"] != side:
        msgs.append(f"The 3D model thinks this is a {stats['footedness_predicted']} foot — check the selected foot")
    data = {
        "side": side,
        "method": "focus_sfm",
        "units": "mm",
        "axes": {"x": "user right", "y": "heel to toe", "z": "up from floor"},
        "vertices": [[round(float(c), 2) for c in v] for v in res["vertices"]],
        "faces": np.asarray(res["faces"]).tolist(),
        "colors": np.asarray(res["colors"]).tolist(),
        "scale_source": res["scale_source"],
        "mesh_length_mm": round(res["mesh_length_mm"], 1),
        "stats": stats,
        "reconstruction_confidence": round(min(1.0, coverage) * (1.0 if calibrated else 0.5), 2),
    }
    return AgentOutput(AGENT, True, data["reconstruction_confidence"], data, messages=msgs)


def to_obj(mesh: dict) -> str:
    lines = ["# NIKONYX colour foot mesh (FOCUS-SfM), millimetres", f"# side: {mesh.get('side')}", f"# scale: {mesh.get('scale_source')}"]
    lines += [f"v {x} {y} {z} {r / 255:.3f} {g / 255:.3f} {b / 255:.3f}" for (x, y, z), (r, g, b) in zip(mesh["vertices"], mesh["colors"])]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh["faces"]]
    return "\n".join(lines) + "\n"
