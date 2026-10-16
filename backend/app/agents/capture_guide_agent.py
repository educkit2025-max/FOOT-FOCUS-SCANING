"""Agent 1 — Capture guide: live per-frame assessment with GREEN / AMBER / RED guidance."""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.agents import calibration_agent, landmark_agent, segmentation_agent
from app.agents.base import AgentOutput
from app.agents.imaging import SHARP_MIN, SHARP_OK, downscale, intrinsics_for, sharpness_score

AGENT = "capture_guide"
GREEN, AMBER, RED = "GREEN", "AMBER", "RED"
SEVERITY = {GREEN: 0, AMBER: 1, RED: 2}

VIEW_INFO = {
    "left_top": ("left", "top"),
    "right_top": ("right", "top"),
    "left_medial": ("left", "side"),
    "right_medial": ("right", "side"),
    "left_oblique": ("left", "oblique"),
    "right_oblique": ("right", "oblique"),
    "left_rear": ("left", "rear"),
    "right_rear": ("right", "rear"),
    "left_lateral": ("left", "lateral"),
    "right_lateral": ("right", "lateral"),
}


def view_kind(view: str) -> tuple[str, str]:
    return VIEW_INFO.get(view, ("right", "top"))


class _Checks:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, check: str, state: str, message: str) -> None:
        self.items.append({"check": check, "state": state, "message": message})

    def worst(self) -> dict | None:
        bad = [c for c in self.items if c["state"] != GREEN]
        if not bad:
            return None
        return max(bad, key=lambda c: SEVERITY[c["state"]])


def _norm(pts: np.ndarray, w: int, h: int) -> list[list[float]]:
    return [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in np.asarray(pts).reshape(-1, 2)]


def gravity_angles(gravity: tuple[float, float, float] | None) -> dict | None:
    """Android TYPE_GRAVITY in device axes. Back camera looks along −z."""
    if not gravity:
        return None
    gx, gy, gz = gravity
    g = math.sqrt(gx * gx + gy * gy + gz * gz)
    if g < 1e-3:
        return None
    return {
        "down_tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, gz / g)))),
        "elevation_deg": math.degrees(math.asin(max(-1.0, min(1.0, gz / g)))),
        "roll_deg": math.degrees(math.atan2(gx, gy)),
    }


def _lighting(image: np.ndarray, checks: _Checks) -> float:
    v = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2]
    mean = float(v.mean())
    clipped = float((v > 250).mean())
    if mean < 60:
        checks.add("lighting", RED, "Too dark — add more light")
        return 0.3
    if clipped > 0.25:
        checks.add("lighting", AMBER, "Too bright — avoid direct light on the paper")
        return 0.7
    if mean < 90:
        checks.add("lighting", AMBER, "Lighting is low — move to a brighter area")
        return 0.75
    checks.add("lighting", GREEN, "Lighting good")
    return 1.0


def _motion(gray_small: np.ndarray, prev_gray: np.ndarray | None, checks: _Checks) -> float:
    if prev_gray is None or prev_gray.shape != gray_small.shape:
        checks.add("motion", GREEN, "Hold steady")
        return 1.0
    diff = float(np.mean(cv2.absdiff(gray_small, prev_gray)))
    if diff > 14:
        checks.add("motion", RED, "Too much motion — hold the phone still")
        return 0.4
    if diff > 7:
        checks.add("motion", AMBER, "Slight motion — hold steady")
        return 0.8
    checks.add("motion", GREEN, "Steady")
    return 1.0


def _sharp(image: np.ndarray, checks: _Checks) -> float:
    s = sharpness_score(image)
    if s < SHARP_MIN:
        checks.add("focus", RED, "Image blurred — tap to focus and hold still")
        return 0.4
    if s < SHARP_OK:
        checks.add("focus", AMBER, "Image slightly soft — hold still")
        return 0.85
    checks.add("focus", GREEN, "Sharp")
    return 1.0


def _assess_top(img, side, intr, grav, checks: _Checks, overlays: dict) -> dict:
    h, w = img.shape[:2]
    out: dict = {"foot_detected": False, "camera_tilt_deg": None, "camera_height_mm": None}
    cal = calibration_agent.run(img, intr)
    if not cal.ok:
        if grav:
            out["camera_tilt_deg"] = round(grav["down_tilt_deg"], 1)
        checks.add("a4", RED, "A4 paper not visible — show the whole sheet")
        return out
    paper = cal.artifacts["paper"]
    overlays["paper"] = _norm(paper.corners_img, w, h)
    heel_line = paper.plane_to_image([[0.0, 12.0], [paper.paper_w_mm, 12.0]])
    overlays["heel_line"] = _norm(heel_line, w, h)
    out["camera_tilt_deg"] = round(cal.data["tilt_deg"], 1)
    out["camera_height_mm"] = cal.data["camera_height_mm"]

    if cal.data["paper_cut_off"]:
        checks.add("a4", RED, "Keep the whole A4 sheet in view — move higher")
    elif cal.data["paper_area_fraction"] < 0.12:
        checks.add("a4", AMBER, "Move closer — the paper is small in the frame")
    else:
        checks.add("a4", GREEN, "A4 detected")

    tilt = cal.data["tilt_deg"]
    if tilt > 15:
        checks.add("tilt", RED, f"Camera tilt: {tilt:.0f}° — reduce angle")
    elif tilt > 10:
        checks.add("tilt", AMBER, f"Camera tilt: {tilt:.0f}° — hold the phone flatter")
    else:
        checks.add("tilt", GREEN, f"Camera tilt: {tilt:.0f}°")

    rect = calibration_agent.rectify(img, paper, px_per_mm=1.0, margin_mm=10.0)
    seg = segmentation_agent.segment_top(rect)
    if not seg.ok:
        checks.add("foot", RED, "Place your foot on the paper")
        return out
    out["foot_detected"] = True
    cnt = seg.artifacts["contour"].reshape(-1, 2).astype(np.float64)
    step = max(1, len(cnt) // 80)
    img_pts = paper.plane_to_image(rect.rect_to_plane(cnt[::step]))
    overlays["foot_contour"] = _norm(img_pts, w, h)

    ec = seg.data["edge_contact"]
    if ec["toe_edge"] and seg.data["toe_contact_mm"] > 8:
        checks.add("foot", RED, "Toes outside the paper — move your foot back")
    elif ec["heel_edge"]:
        checks.add("foot", RED, "Place your heel on the paper line")
    elif ec["left_edge"] or ec["right_edge"]:
        checks.add("foot", AMBER, "Centre your foot on the paper")
    else:
        checks.add("foot", GREEN, "Foot inside the paper")

    lm = landmark_agent.locate_top(seg.artifacts["mask"], seg.artifacts["contour"], rect, paper, side)
    if lm.ok:
        if lm.data["side_confidence"] >= 0.5 and lm.data["detected_side"] != side:
            checks.add("side", RED, f"Wrong foot — show your {side} foot")
        else:
            checks.add("side", GREEN, f"{side.capitalize()} foot")
        out["preview_length_mm"] = lm.data["raw_mm"]["foot_length"]
        out["preview_width_mm"] = lm.data["raw_mm"]["max_foot_width"]
    return out


def _assess_side(img, side, kind, grav, checks: _Checks, overlays: dict) -> dict:
    h, w = img.shape[:2]
    out: dict = {"foot_detected": False, "camera_tilt_deg": None, "camera_height_mm": None}
    if grav:
        elev = grav["elevation_deg"]
        out["camera_tilt_deg"] = round(abs(elev), 1)
        if abs(elev) > 15:
            checks.add("tilt", RED, f"Camera tilt: {abs(elev):.0f}° — hold the phone upright at foot level")
        elif abs(elev) > 10:
            checks.add("tilt", AMBER, f"Camera tilt: {abs(elev):.0f}° — level the phone")
        else:
            checks.add("tilt", GREEN, "Camera level")

    small, scale = downscale(img, 480)
    seg = segmentation_agent.segment_side(small, work_side=320)
    if not seg.ok:
        checks.add("foot", RED, "Foot not detected — show the inner side of the foot")
        return out
    out["foot_detected"] = True
    mask = seg.artifacts["mask"]
    s_total = seg.data["work_scale"] * scale
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64) / s_total
    overlays["foot_contour"] = _norm(cnt[:: max(1, len(cnt) // 80)], w, h)

    t = seg.data["touches_frame"]
    if t["left"] or t["right"]:
        checks.add("frame", RED, "Foot outside frame — step back so heel and toes are visible")
    elif t["bottom"]:
        checks.add("frame", RED, "Show the floor under the foot")
    else:
        checks.add("frame", GREEN, "Whole foot in frame")

    # Nominal scale: only orientation and level checks are read from this call.
    lm = landmark_agent.locate_side(mask, side, 250.0)
    if lm.ok:
        roll = lm.data["roll_deg"]
        if abs(roll) > 8:
            checks.add("level", AMBER, "Camera not level — rotate the phone slightly")
        else:
            checks.add("level", GREEN, "Ground line level")
        if kind == "side":
            if lm.data["wrong_direction"]:
                checks.add("side", RED, "Wrong side — show the inner side of the foot")
            else:
                checks.add("side", GREEN, "Inner side visible")
        checks.add("heel", GREEN, "Heel on the ground")
    else:
        checks.add("heel", AMBER, "Keep the heel on the ground and the floor visible")
    return out


def assess(
    image: np.ndarray,
    view: str,
    focal_mm: float | None = None,
    sensor_long_mm: float | None = None,
    prev_gray: np.ndarray | None = None,
    gravity: tuple[float, float, float] | None = None,
) -> tuple[AgentOutput, np.ndarray]:
    img, _ = downscale(image, 960)
    side, kind = view_kind(view)
    intr = intrinsics_for(img.shape, focal_mm, sensor_long_mm)
    grav = gravity_angles(gravity)
    checks = _Checks()
    overlays: dict = {}

    light_q = _lighting(img, checks)
    gray_small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (160, 120))
    motion_q = _motion(gray_small, prev_gray, checks)
    sharp_q = _sharp(img, checks)

    if kind == "top":
        geo = _assess_top(img, side, intr, grav, checks, overlays)
    else:
        geo = _assess_side(img, side, kind, grav, checks, overlays)

    worst = checks.worst()
    state = GREEN if worst is None else worst["state"]
    message = "Hold still — ready to capture" if worst is None else worst["message"]
    sev_penalty = sum(0.25 if c["state"] == RED else 0.08 if c["state"] == AMBER else 0.0 for c in checks.items)
    score = max(0.0, min(1.0, light_q * motion_q * sharp_q - sev_penalty))

    data = {
        "view": view,
        "capture_valid": state == GREEN,
        "foot_detected": geo.get("foot_detected", False),
        "camera_tilt_deg": geo.get("camera_tilt_deg"),
        "camera_height_mm": geo.get("camera_height_mm"),
        "motion_quality": round(motion_q, 2),
        "lighting_quality": round(light_q, 2),
        "guidance_state": state,
        "message": message,
        "checks": checks.items,
        "overlays": overlays,
        "quality_score": int(round(score * 100)),
        "preview_length_mm": geo.get("preview_length_mm"),
        "preview_width_mm": geo.get("preview_width_mm"),
    }
    return AgentOutput(AGENT, True, score, data), gray_small
