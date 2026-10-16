"""Agent 5 — 3D reconstruction.

Builds a watertight foot surface (mm) from the calibrated top outline and the medial height profile
(two-view silhouette hull with superellipse cross-sections). Girths and volume are measured on it; the
coloured surface shown to the user comes from `focus_agent` when FOCUS is installed.
"""

from __future__ import annotations

import numpy as np

from app.agents.base import AgentOutput, failed

AGENT = "reconstruction"
STATIONS = 49
SEGMENTS = 32
TOP_EXPONENT = 2.4
SOLE_EXPONENT = 4.0
SOLE_FRACTION = 0.38


def full_3d_status() -> dict:
    from app.agents import focus_agent

    return focus_agent.status()


def _interp_outline(outline_uv: list, stations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(outline_uv, np.float64)
    vmin = np.interp(stations, arr[:, 0], arr[:, 1])
    vmax = np.interp(stations, arr[:, 0], arr[:, 2])
    return vmin, vmax


def _interp_height(profile_uh: list, stations: np.ndarray) -> np.ndarray:
    arr = np.asarray(profile_uh, np.float64)
    return np.interp(stations, arr[:, 0], arr[:, 1])


def _superellipse(a: float, b_top: float, b_bot: float, n: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    x = a * np.sign(c) * np.abs(c) ** (2.0 / TOP_EXPONENT)
    top = s >= 0
    z = np.where(
        top,
        b_top * np.abs(s) ** (2.0 / TOP_EXPONENT),
        -b_bot * np.abs(s) ** (2.0 / SOLE_EXPONENT),
    )
    x = np.where(top, x, a * np.sign(c) * np.abs(c) ** (2.0 / SOLE_EXPONENT))
    return np.column_stack([x, z])


def _ring_perimeter(ring: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(np.vstack([ring, ring[:1]]), axis=0), axis=1).sum())


def _section(vmin: float, vmax: float, height: float, n: int = SEGMENTS) -> np.ndarray:
    a = max((vmax - vmin) / 2.0, 0.5)
    vc = (vmax + vmin) / 2.0
    zc = max(height, 1.0) * SOLE_FRACTION
    ring = _superellipse(a, max(height, 1.0) - zc, zc, n)
    ring[:, 0] += vc
    ring[:, 1] += zc
    return ring


def _mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def run(top: dict, side: dict | None, side_name: str) -> AgentOutput:
    outline = top.get("outline_uv") or []
    L = float(top.get("raw_mm", {}).get("foot_length") or 0.0)
    if len(outline) < 10 or L <= 0:
        return failed(AGENT, "TOP_OUTLINE_MISSING", ["Top view outline missing"])
    if not side or not side.get("profile_uh"):
        return failed(AGENT, "SIDE_PROFILE_MISSING", ["Inner-side view needed for 3D shape"])

    stations = np.linspace(0.0, L, STATIONS)
    vmin, vmax = _interp_outline(outline, stations)
    heights = _interp_height(side["profile_uh"], stations)
    # Ends of the foot taper to closed caps.
    taper = np.clip(np.minimum(stations, L - stations) / (0.03 * L), 0.15, 1.0)
    heights = heights * np.sqrt(taper)

    rings = []
    verts = []
    for i, u in enumerate(stations):
        ring = _section(vmin[i], vmax[i], heights[i])
        rings.append(ring)
        for x, z in ring:
            verts.append([x, u, z])
    verts = np.asarray(verts, np.float64)

    faces = []
    for i in range(STATIONS - 1):
        for j in range(SEGMENTS):
            a = i * SEGMENTS + j
            b = i * SEGMENTS + (j + 1) % SEGMENTS
            c = (i + 1) * SEGMENTS + j
            d = (i + 1) * SEGMENTS + (j + 1) % SEGMENTS
            faces += [[a, c, b], [b, c, d]]
    heel_c = len(verts)
    toe_c = heel_c + 1
    verts = np.vstack([verts, [rings[0].mean(axis=0)[0], 0.0, rings[0].mean(axis=0)[1]], [rings[-1].mean(axis=0)[0], L, rings[-1].mean(axis=0)[1]]])
    last = (STATIONS - 1) * SEGMENTS
    for j in range(SEGMENTS):
        faces.append([heel_c, j, (j + 1) % SEGMENTS])
        faces.append([toe_c, last + (j + 1) % SEGMENTS, last + j])
    faces = np.asarray(faces, np.int64)

    vol = _mesh_volume(verts, faces)
    if vol < 0:
        faces = faces[:, [0, 2, 1]]
        vol = -vol

    lm = top.get("landmarks_mm", {})

    def girth_at(u: float) -> tuple[float, list]:
        u = float(np.clip(u, 0.0, L))
        vmn, vmx = _interp_outline(outline, np.array([u]))
        hh = float(_interp_height(side["profile_uh"], np.array([u]))[0])
        ring = _section(float(vmn[0]), float(vmx[0]), hh, 64)
        pts = [[round(float(x), 1), round(u, 1), round(float(z), 1)] for x, z in ring]
        return _ring_perimeter(ring), pts

    ball_u = float(lm.get("ball_line_midpoint", [0.72 * L])[0])
    fore_u = float(lm.get("forefoot_a", [0.70 * L])[0])
    girths = {}
    for name, u in (("ball_girth", ball_u), ("forefoot_girth", fore_u), ("instep_girth", 0.55 * L)):
        g, ring = girth_at(u)
        girths[name] = {"value_mm": round(g, 1), "u_mm": round(u, 1), "ring": ring}

    status = full_3d_status()
    data = {
        "side": side_name,
        "method": "two_view_silhouette_hull",
        "units": "mm",
        "axes": {"x": "user right", "y": "heel to toe", "z": "up from floor"},
        "vertices": [[round(float(c), 2) for c in v] for v in verts],
        "faces": faces.tolist(),
        "stations": STATIONS,
        "segments": SEGMENTS,
        "volume_cm3": round(vol / 1000.0, 1),
        "girths": girths,
        "full_3d": status,
        "reconstruction_confidence": 0.7,
    }
    msgs = [] if status["available"] else ["Colour 3D (FOCUS) not installed — showing the calibrated two-view shape"]
    return AgentOutput(AGENT, True, 0.7, data, messages=msgs)


def to_obj(mesh: dict) -> str:
    lines = ["# NIKONYX foot mesh, millimetres", f"# side: {mesh.get('side')}", f"# method: {mesh.get('method')}"]
    lines += [f"v {x} {y} {z}" for x, y, z in mesh["vertices"]]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh["faces"]]
    return "\n".join(lines) + "\n"
