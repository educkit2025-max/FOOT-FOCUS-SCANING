"""Live capture guidance: the phone posts preview frames and receives GREEN / AMBER / RED feedback."""

from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.agents import capture_guide_agent
from app.agents.imaging import load_image_bytes

router = APIRouter(prefix="/api/v1/guide", tags=["guide"])

_MAX_SESSIONS = 32
_prev_frames: OrderedDict[str, tuple[float, np.ndarray]] = OrderedDict()


@router.post("/frame")
async def guide_frame(
    file: UploadFile = File(...),
    view: str = Form(...),
    session_id: str = Form("default"),
    focal_mm: float | None = Form(None),
    sensor_long_mm: float | None = Form(None),
    gravity_x: float | None = Form(None),
    gravity_y: float | None = Form(None),
    gravity_z: float | None = Form(None),
) -> dict:
    data = await file.read()
    try:
        image = load_image_bytes(data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"unreadable image: {exc}") from exc

    key = f"{session_id}:{view}"
    prev = _prev_frames.get(key)
    prev_gray = prev[1] if prev and time.time() - prev[0] < 3.0 else None
    gravity = (gravity_x, gravity_y, gravity_z) if None not in (gravity_x, gravity_y, gravity_z) else None
    started = time.perf_counter()
    out, gray = await run_in_threadpool(capture_guide_agent.assess, image, view, focal_mm, sensor_long_mm, prev_gray, gravity)

    _prev_frames[key] = (time.time(), gray)
    _prev_frames.move_to_end(key)
    while len(_prev_frames) > _MAX_SESSIONS:
        _prev_frames.popitem(last=False)
    return {**out.data, "latency_ms": int((time.perf_counter() - started) * 1000)}
