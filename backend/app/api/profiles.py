"""Multi-view 3D foot profile endpoints (NIKONYX)."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.agents import capture_guide_agent
from app.agents.imaging import load_image_bytes
from app.agents.orchestrator import STAGES, process_profile
from app.services import profile_storage as store

router = APIRouter(prefix="/api/v1/profiles", tags=["profiles"])


class CreateProfileRequest(BaseModel):
    device_model: str | None = None
    app_version: str | None = None


def _require(profile_id: str) -> None:
    if not store.exists(profile_id):
        raise HTTPException(404, "profile not found")


def _views_summary(profile_id: str) -> list[dict]:
    out = []
    for view in store.ALL_VIEWS:
        meta = store.view_meta(profile_id, view)
        out.append(
            {
                "view": view,
                "required": view in store.REQUIRED_VIEWS,
                "uploaded": meta is not None,
                "captures": (meta or {}).get("captures", 0),
                "quality_score": (meta or {}).get("quality_score"),
                "guidance_state": (meta or {}).get("guidance_state"),
                "message": (meta or {}).get("message"),
            }
        )
    return out


@router.post("")
async def create_profile(body: CreateProfileRequest | None = None) -> dict:
    pid = store.create_profile((body or CreateProfileRequest()).model_dump())
    return {"profile_id": pid, "required_views": store.REQUIRED_VIEWS, "optional_views": store.OPTIONAL_VIEWS}


@router.get("/{profile_id}")
async def get_profile(profile_id: str) -> dict:
    _require(profile_id)
    return {
        "profile": store.read_profile(profile_id),
        "views": _views_summary(profile_id),
        "status": store.read_status(profile_id),
        "ready_to_build": all(v in store.uploaded_views(profile_id) for v in store.REQUIRED_VIEWS),
    }


@router.post("/{profile_id}/views/{view}")
async def upload_view(
    profile_id: str,
    view: str,
    files: list[UploadFile] = File(...),
    focal_mm: float | None = Form(None),
    sensor_long_mm: float | None = Form(None),
    gravity_x: float | None = Form(None),
    gravity_y: float | None = Form(None),
    gravity_z: float | None = Form(None),
) -> dict:
    _require(profile_id)
    if view not in store.ALL_VIEWS:
        raise HTTPException(400, f"unknown view {view}")
    images = [await f.read() for f in files]
    images = [b for b in images if b]
    if not images:
        raise HTTPException(400, "no image data")
    try:
        first = load_image_bytes(images[0])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"unreadable image: {exc}") from exc
    gravity = (gravity_x, gravity_y, gravity_z) if None not in (gravity_x, gravity_y, gravity_z) else None
    assessment, _ = await run_in_threadpool(capture_guide_agent.assess, first, view, focal_mm, sensor_long_mm, None, gravity)
    a = assessment.data
    meta = {
        "focal_mm": focal_mm,
        "sensor_long_mm": sensor_long_mm,
        "gravity_x": gravity_x,
        "gravity_y": gravity_y,
        "gravity_z": gravity_z,
        "quality_score": a["quality_score"],
        "guidance_state": a["guidance_state"],
        "message": a["message"],
        "checks": a["checks"],
    }
    info = await run_in_threadpool(store.save_view, profile_id, view, images, meta)
    return {"view": view, "captures": info["captures"], "assessment": a}


@router.get("/{profile_id}/views/{view}/thumbnail")
async def view_thumbnail(profile_id: str, view: str) -> FileResponse:
    _require(profile_id)
    path = store.thumbnail_path(profile_id, view)
    if not path.exists():
        raise HTTPException(404, "thumbnail not found")
    return FileResponse(path, media_type="image/jpeg")


@router.post("/{profile_id}/process")
async def process(profile_id: str, background: BackgroundTasks) -> dict:
    _require(profile_id)
    missing = [v for v in store.REQUIRED_VIEWS if v not in store.uploaded_views(profile_id)]
    if missing:
        raise HTTPException(409, {"error": "REQUIRED_VIEWS_MISSING", "missing": missing})
    status = store.read_status(profile_id) or {}
    if status.get("state") == "processing":
        return {"profile_id": profile_id, "state": "processing"}
    store.write_status(
        profile_id,
        {"state": "processing", "stage": STAGES[0][0], "progress": 0.0, "stages": [{"id": k, "label": l, "state": "pending"} for k, l in STAGES]},
    )
    background.add_task(process_profile, profile_id)
    return {"profile_id": profile_id, "state": "processing"}


@router.get("/{profile_id}/status")
async def status(profile_id: str) -> dict:
    _require(profile_id)
    return store.read_status(profile_id) or {}


@router.get("/{profile_id}/result")
async def result(profile_id: str) -> dict:
    _require(profile_id)
    res = store.read_result(profile_id)
    if res is None:
        raise HTTPException(404, "result not ready")
    return res


@router.get("/{profile_id}/mesh/{side}.{ext}")
async def mesh(profile_id: str, side: str, ext: str) -> FileResponse:
    _require(profile_id)
    if side not in ("left", "right") or ext not in ("json", "obj"):
        raise HTTPException(400, "side must be left/right and ext json/obj")
    path = store.mesh_path(profile_id, side, ext)
    if not path.exists():
        raise HTTPException(404, "mesh not available")
    return FileResponse(path, media_type="application/json" if ext == "json" else "text/plain")


@router.delete("/{profile_id}")
async def delete(profile_id: str) -> dict:
    _require(profile_id)
    store.delete_profile(profile_id)
    return {"deleted": profile_id}
