"""Background processing orchestration for Milestone 1."""

from __future__ import annotations

from pathlib import Path

from app.models.schemas import ScanStatus
from app.services import storage
from app.services.frame_selection import select_frames


def process_scan_frames(scan_id: str, target_frames: int) -> dict:
    meta = storage.load_meta(scan_id)
    video_rel = meta.get("paths", {}).get("video")
    if not video_rel:
        raise FileNotFoundError("Video not uploaded for this scan")

    video_path = Path(video_rel)
    root = storage.scan_dir(scan_id)

    storage.set_status(
        scan_id,
        ScanStatus.processing,
        progress=10,
        stage="extracting_frames",
        message="Extracting candidate frames",
        error_code=None,
    )

    try:
        storage.set_status(
            scan_id,
            ScanStatus.processing,
            progress=35,
            stage="quality_filter",
            message="Scoring sharpness and viewpoint novelty",
        )
        report = select_frames(
            video_path=video_path,
            frames_dir=root / "frames",
            selected_dir=root / "selected",
            thumbnails_dir=root / "thumbnails",
            target_frames=target_frames,
        )
        storage.set_status(
            scan_id,
            ScanStatus.frames_ready,
            progress=100,
            stage="frames_ready",
            message="Selected frames ready for review",
            extra={"frame_summary": {
                "original_frame_count": report["original_candidate_count"],
                "selected_frame_count": report["selected_frame_count"],
                "target_frames": target_frames,
            }},
        )
        return report
    except Exception as exc:  # noqa: BLE001 - surface to API status
        storage.set_status(
            scan_id,
            ScanStatus.failed,
            progress=100,
            stage="failed",
            message=str(exc),
            error_code="FRAME_SELECTION_FAILED",
        )
        raise
