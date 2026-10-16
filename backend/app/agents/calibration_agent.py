"""Agent 2 — Calibration: A4 detection, corners, perspective, scale, plane and camera pose."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.agents.base import AgentOutput, failed
from app.agents.imaging import Intrinsics, downscale

AGENT = "calibration"
A4_SHORT_MM = 210.0
A4_LONG_MM = 297.0


@dataclass
class PaperGeometry:
    """Plane frame: origin at the heel-side (image-bottom) left corner, x to the right, y toward the toes, mm."""

    corners_img: np.ndarray
    paper_w_mm: float
    paper_h_mm: float
    H_img2plane: np.ndarray
    H_plane2img: np.ndarray
    camera_center_mm: np.ndarray
    tilt_deg: float

    def plane_to_image(self, pts) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(arr, self.H_plane2img).reshape(-1, 2)

    def image_to_plane(self, pts) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(arr, self.H_img2plane).reshape(-1, 2)


@dataclass
class RectifiedView:
    """Top-down image of the paper plane at a fixed px/mm with a margin around the sheet."""

    image: np.ndarray
    valid: np.ndarray
    px_per_mm: float
    margin_mm: float
    paper_w_mm: float
    paper_h_mm: float

    def rect_to_plane(self, pts) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        x = arr[:, 0] / self.px_per_mm - self.margin_mm
        y = (self.paper_h_mm + self.margin_mm) - arr[:, 1] / self.px_per_mm
        return np.column_stack([x, y])

    def paper_box_px(self, inset_mm: float = 0.0) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) of the sheet in rectified pixels; y1 is the heel edge."""
        s, m = self.px_per_mm, self.margin_mm
        x0 = int(round((m + inset_mm) * s))
        x1 = int(round((m + self.paper_w_mm - inset_mm) * s))
        y0 = int(round((m + inset_mm) * s))
        y1 = int(round((m + self.paper_h_mm - inset_mm) * s))
        return x0, y0, x1, y1


def _paper_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = (gray > max(otsu, 90)) & (hsv[..., 1] < 70)
    mask = bright.astype(np.uint8) * 255
    k = max(3, int(round(min(img.shape[:2]) * 0.01)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def _approx_quad(hull: np.ndarray) -> np.ndarray | None:
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.1, 19):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
        if len(approx) < 4:
            break
    return None


def find_paper_quad(img: np.ndarray) -> np.ndarray | None:
    mask = _paper_mask(img)
    area_img = float(mask.shape[0] * mask.shape[1])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(cnt)
        if area < 0.04 * area_img:
            break
        if area > 0.97 * area_img:
            continue
        # The convex hull fills the notch the foot cuts into the sheet.
        hull = cv2.convexHull(cnt)
        hull_area = max(cv2.contourArea(hull), 1.0)
        quad = _approx_quad(hull)
        if quad is None:
            box = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
            if hull_area / max(cv2.contourArea(box), 1.0) < 0.9:
                continue
            quad = box
        if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
            continue
        if 0.85 <= cv2.contourArea(quad) / hull_area <= 1.15:
            return quad
    return None


def _order_heel_first(quad: np.ndarray) -> np.ndarray:
    """Return corners as BL, BR, TR, TL where the bottom edge (heel edge) is lowest in the image."""
    c = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    q = quad[np.argsort(ang)]  # clockwise on screen (image y points down)
    mids_y = [(q[i][1] + q[(i + 1) % 4][1]) / 2.0 for i in range(4)]
    i = int(np.argmax(mids_y))
    br, bl, tl, tr = q[i], q[(i + 1) % 4], q[(i + 2) % 4], q[(i + 3) % 4]
    return np.array([bl, br, tr, tl], dtype=np.float64)


def _refine_corners(gray: np.ndarray, corners: np.ndarray, scale: float) -> np.ndarray:
    win = int(np.clip(round(4.0 / scale), 5, 15))
    pts = corners.reshape(-1, 1, 2).astype(np.float32).copy()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    try:
        refined = cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), criteria).reshape(-1, 2).astype(np.float64)
    except cv2.error:
        return corners
    moved = np.linalg.norm(refined - corners, axis=1)
    return np.where((moved < 2.5 / scale)[:, None], refined, corners)


def _pose(H_plane2img: np.ndarray, intr: Intrinsics) -> tuple[np.ndarray, float, float, float]:
    """Camera centre in plane mm, tilt from the plane normal, and two shape-consistency metrics."""
    M = np.linalg.inv(intr.matrix()) @ H_plane2img
    n1 = float(np.linalg.norm(M[:, 0]))
    n2 = float(np.linalg.norm(M[:, 1]))
    lam = 2.0 / (n1 + n2)
    if lam * M[2, 2] < 0:
        lam = -lam
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    r3 = np.cross(r1, r2)
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    cam = -R.T @ t
    tilt = float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0.0, 1.0))))
    aspect_dev = abs(n1 / n2 - 1.0)
    ortho = abs(float(M[:, 0] @ M[:, 1])) / (n1 * n2)
    return cam, tilt, aspect_dev, ortho


def run(image: np.ndarray, intr: Intrinsics) -> AgentOutput:
    small, scale = downscale(image, 1280)
    quad = find_paper_quad(small)
    if quad is None:
        return failed(AGENT, "CALIBRATION_NOT_FOUND", ["A4 paper not detected — keep the whole sheet in view"])

    corners = _order_heel_first(quad) / scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = _refine_corners(gray, corners, scale)

    bottom = float(np.linalg.norm(corners[1] - corners[0]))
    left = float(np.linalg.norm(corners[3] - corners[0]))
    portrait = bottom <= left
    pw, ph = (A4_SHORT_MM, A4_LONG_MM) if portrait else (A4_LONG_MM, A4_SHORT_MM)
    plane = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=np.float32)
    H_i2p = cv2.getPerspectiveTransform(corners.astype(np.float32), plane)
    H_p2i = np.linalg.inv(H_i2p)
    cam, tilt, aspect_dev, ortho = _pose(H_p2i, intr)
    height = abs(float(cam[2]))

    h_img, w_img = image.shape[:2]
    messages: list[str] = []
    conf = 1.0 - 3.0 * aspect_dev - 3.0 * ortho
    margin = float(
        min(np.min(corners[:, 0]), np.min(corners[:, 1]), np.min(w_img - corners[:, 0]), np.min(h_img - corners[:, 1]))
    ) / max(h_img, w_img)
    cut_off = margin < 0.01
    if cut_off:
        conf *= 0.35
        messages.append("A4 paper not fully visible")
    area_frac = float(cv2.contourArea(corners.astype(np.float32))) / float(w_img * h_img)
    if area_frac < 0.08:
        conf *= 0.85
        messages.append("Move closer to the paper")
    if not 180.0 <= height <= 1100.0:
        conf *= 0.8
        messages.append("Unusual camera distance")
    conf = float(np.clip(conf, 0.0, 0.99))

    cx, cy = w_img / 2.0, h_img / 2.0
    probe = cv2.perspectiveTransform(np.array([[[cx, cy]], [[cx + 1, cy]], [[cx, cy + 1]]], np.float64), H_i2p).reshape(-1, 2)
    mm_per_px = float((np.linalg.norm(probe[1] - probe[0]) + np.linalg.norm(probe[2] - probe[0])) / 2.0)

    paper = PaperGeometry(
        corners_img=corners,
        paper_w_mm=pw,
        paper_h_mm=ph,
        H_img2plane=H_i2p,
        H_plane2img=H_p2i,
        camera_center_mm=cam,
        tilt_deg=tilt,
    )
    data = {
        "reference": "A4",
        "width_mm": A4_SHORT_MM,
        "height_mm": A4_LONG_MM,
        "orientation": "portrait" if portrait else "landscape",
        "scale_valid": conf >= 0.6 and not cut_off,
        "calibration_confidence": round(conf, 3),
        "tilt_deg": round(tilt, 2),
        "camera_height_mm": round(height, 1),
        "camera_position_mm": [round(float(v), 1) for v in cam],
        "mm_per_px_at_center": round(mm_per_px, 4),
        "corners_px": [[round(float(x), 1), round(float(y), 1)] for x, y in corners],
        "paper_area_fraction": round(area_frac, 3),
        "paper_cut_off": cut_off,
        "aspect_deviation": round(aspect_dev, 4),
        "orthogonality": round(ortho, 4),
        "intrinsics_source": intr.source,
    }
    return AgentOutput(agent=AGENT, ok=True, confidence=conf, data=data, messages=messages, artifacts={"paper": paper})


def rectify(image: np.ndarray, paper: PaperGeometry, px_per_mm: float = 2.0, margin_mm: float = 30.0) -> RectifiedView:
    s, m = px_per_mm, margin_mm
    T = np.array([[s, 0.0, s * m], [0.0, -s, s * (paper.paper_h_mm + m)], [0.0, 0.0, 1.0]])
    size = (int(round(s * (paper.paper_w_mm + 2 * m))), int(round(s * (paper.paper_h_mm + 2 * m))))
    warp = T @ paper.H_img2plane
    rect = cv2.warpPerspective(image, warp, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ones = np.full(image.shape[:2], 255, np.uint8)
    valid = cv2.warpPerspective(ones, warp, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return RectifiedView(rect, valid, s, m, paper.paper_w_mm, paper.paper_h_mm)
