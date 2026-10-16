"""Background step 1 of a video scan: select 60–80 frames and pick the 5 best positions."""

from __future__ import annotations

from pathlib import Path

from app.models.schemas import ScanStatus
from app.services import storage
from app.services.frame_selection import select_frames
from app.services.video_measure import analyze_frames


def process_scan_frames(scan_id: str, target_frames: int) -> dict:
    meta = storage.load_meta(scan_id)
    video_rel = meta.get("paths", {}).get("video")
    if not video_rel:
        raise FileNotFoundError("Video not uploaded for this scan")

    video_path = Path(video_rel)
    root = storage.scan_dir(scan_id)

    try:
        storage.set_status(
            scan_id,
            ScanStatus.processing,
            progress=10,
            stage="extracting_frames",
            message="Extracting frames from the 30 s video",
            error_code=None,
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
            ScanStatus.processing,
            progress=60,
            stage="best_frames",
            message=f"Kept {report['selected_frame_count']} sharp frames — choosing the best 5 positions",
        )
        best = analyze_frames(scan_id, report)
        storage.set_status(
            scan_id,
            ScanStatus.frames_ready,
            progress=100,
            stage="frames_ready",
            message=f"{report['selected_frame_count']} frames selected",
            extra={
                "frame_summary": {
                    "original_frame_count": report["original_candidate_count"],
                    "selected_frame_count": report["selected_frame_count"],
                    "target_frames": target_frames,
                    "a4_frames": sum(1 for b in best if b["a4_detected"]),
                },
            },
        )
        return {"frames": report, "best": best}
    except Exception as exc:  # noqa: BLE001
        storage.set_status(
            scan_id,
            ScanStatus.failed,
            progress=100,
            stage="failed",
            message=str(exc),
            error_code="PROCESSING_FAILED",
        )
        raise
