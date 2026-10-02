"""Scan REST endpoints (Milestone 1)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.models.schemas import (
    CreateScanRequest,
    CreateScanResponse,
    FramesResponse,
    FrameInfo,
    ProcessRequest,
    ScanStatus,
    StatusResponse,
)
from app.services import storage
from app.services.pipeline import process_scan_frames

router = APIRouter(prefix="/api/v1/scans", tags=["scans"])


def _status_from_meta(meta: dict) -> StatusResponse:
    return StatusResponse(
        scan_id=meta["scan_id"],
        status=ScanStatus(meta["status"]),
        progress_percent=meta.get("progress_percent", 0),
        stage=meta.get("stage", ""),
        message=meta.get("message", ""),
        error_code=meta.get("error_code"),
        foot=meta.get("foot"),
        created_at=meta.get("created_at"),
        updated_at=meta.get("updated_at"),
        capture=meta.get("capture") or {},
        frame_summary=meta.get("frame_summary") or {},
    )


@router.post("", response_model=CreateScanResponse)
async def create_scan(body: CreateScanRequest) -> CreateScanResponse:
    meta = storage.create_scan(
        {
            "foot": body.foot.value,
            "device": body.device.model_dump(),
            "capture": body.capture.model_dump(),
            "notes": body.notes,
        }
    )
    return CreateScanResponse(
        scan_id=meta["scan_id"],
        status=ScanStatus.created,
        created_at=meta["created_at"],
    )


@router.get("")
async def list_scans() -> list[StatusResponse]:
    return [_status_from_meta(m) for m in storage.list_scans()]


@router.get("/{scan_id}/status", response_model=StatusResponse)
async def get_status(scan_id: str) -> StatusResponse:
    try:
        meta = storage.load_meta(scan_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _status_from_meta(meta)


@router.post("/{scan_id}/video")
async def upload_video(scan_id: str, file: UploadFile = File(...)) -> StatusResponse:
    try:
        meta = storage.load_meta(scan_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    suffix = Path(file.filename or "scan.mp4").suffix or ".mp4"
    dest = storage.scan_dir(scan_id) / "video" / f"capture{suffix}"

    storage.set_status(
        scan_id,
        ScanStatus.uploading,
        progress=5,
        stage="uploading",
        message="Receiving video",
    )

    size = 0
    max_bytes = settings.max_upload_mb * 1024 * 1024
    with dest.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                dest.unlink(missing_ok=True)
                storage.set_status(
                    scan_id,
                    ScanStatus.failed,
                    progress=100,
                    stage="failed",
                    message="Upload exceeds size limit",
                    error_code="UPLOAD_TOO_LARGE",
                )
                raise HTTPException(status_code=413, detail="File too large")
            out.write(chunk)

    meta = storage.set_status(
        scan_id,
        ScanStatus.ready,
        progress=15,
        stage="ready",
        message="Video uploaded",
        extra={
            "paths": {**meta.get("paths", {}), "video": str(dest)},
            "capture": {
                **(meta.get("capture") or {}),
                "uploaded_bytes": size,
                "video_filename": dest.name,
            },
        },
    )
    return _status_from_meta(meta)


@router.post("/{scan_id}/process", response_model=StatusResponse)
async def process_scan(
    scan_id: str,
    body: ProcessRequest,
    background_tasks: BackgroundTasks,
) -> StatusResponse:
    try:
        meta = storage.load_meta(scan_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not meta.get("paths", {}).get("video"):
        raise HTTPException(status_code=400, detail="Upload a video first")

    storage.set_status(
        scan_id,
        ScanStatus.processing,
        progress=20,
        stage="queued",
        message="Frame selection queued",
    )

    def _job() -> None:
        process_scan_frames(scan_id, body.target_frames)

    background_tasks.add_task(_job)
    return _status_from_meta(storage.load_meta(scan_id))


@router.get("/{scan_id}/frames", response_model=FramesResponse)
async def get_frames(scan_id: str) -> FramesResponse:
    try:
        meta = storage.load_meta(scan_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    report_path = storage.scan_dir(scan_id) / "frame_selection.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Frames not ready yet")

    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    frames = [
        FrameInfo(
            index=f["index"],
            filename=f["filename"],
            thumbnail_url=f"/api/v1/scans/{scan_id}/thumbnails/{f['thumbnail']}",
            sharpness=f["sharpness"],
            selected=True,
            discard_reason=None,
        )
        for f in report.get("frames", [])
    ]
    return FramesResponse(
        scan_id=scan_id,
        original_frame_count=report.get("original_candidate_count", 0),
        selected_frame_count=report.get("selected_frame_count", 0),
        frames=frames,
    )


@router.get("/{scan_id}/thumbnails/{name}")
async def get_thumbnail(scan_id: str, name: str) -> FileResponse:
    path = storage.scan_dir(scan_id) / "thumbnails" / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(path, media_type="image/jpeg")


@router.delete("/{scan_id}")
async def delete_scan(scan_id: str) -> dict:
    try:
        storage.load_meta(scan_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    storage.delete_scan(scan_id)
    return {"deleted": scan_id}


@router.get("/{scan_id}/result")
async def get_result(scan_id: str) -> dict:
    """Milestone 1 placeholder: frame selection summary only."""
    try:
        meta = storage.load_meta(scan_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if meta.get("status") != ScanStatus.frames_ready.value:
        raise HTTPException(status_code=409, detail="Full result not available yet (Milestone 1 = frames only)")
    frames = await get_frames(scan_id)
    return {
        "scan_id": scan_id,
        "foot": meta.get("foot"),
        "device": meta.get("device"),
        "capture": {
            **(meta.get("capture") or {}),
            "original_frame_count": frames.original_frame_count,
            "selected_frame_count": frames.selected_frame_count,
        },
        "milestone": 1,
        "measurements": None,
        "mesh": None,
        "frames": [f.model_dump() for f in frames.frames],
        "quality": {
            "note": "Reconstruction and metric confidence arrive in later milestones"
        },
        "warnings": [],
    }
