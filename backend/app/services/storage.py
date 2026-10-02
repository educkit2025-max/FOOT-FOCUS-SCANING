"""Filesystem storage for scan packages."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.models.schemas import ScanStatus


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_scan_id() -> str:
    return uuid.uuid4().hex


def scan_dir(scan_id: str) -> Path:
    path = settings.data_dir / scan_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def meta_path(scan_id: str) -> Path:
    return scan_dir(scan_id) / "meta.json"


def load_meta(scan_id: str) -> dict[str, Any]:
    path = meta_path(scan_id)
    if not path.exists():
        raise FileNotFoundError(f"Scan not found: {scan_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_meta(scan_id: str, meta: dict[str, Any]) -> None:
    meta["updated_at"] = _now()
    meta_path(scan_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def create_scan(payload: dict[str, Any]) -> dict[str, Any]:
    scan_id = new_scan_id()
    root = scan_dir(scan_id)
    (root / "video").mkdir(exist_ok=True)
    (root / "frames").mkdir(exist_ok=True)
    (root / "thumbnails").mkdir(exist_ok=True)
    (root / "selected").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)

    meta = {
        "scan_id": scan_id,
        "status": ScanStatus.created.value,
        "progress_percent": 0,
        "stage": "created",
        "message": "Scan created",
        "error_code": None,
        "created_at": _now(),
        "updated_at": _now(),
        **payload,
        "paths": {
            "video": None,
            "frames": str(root / "frames"),
            "thumbnails": str(root / "thumbnails"),
            "selected": str(root / "selected"),
        },
        "frame_summary": {},
    }
    save_meta(scan_id, meta)
    return meta


def set_status(
    scan_id: str,
    status: ScanStatus,
    *,
    progress: int | None = None,
    stage: str | None = None,
    message: str | None = None,
    error_code: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = load_meta(scan_id)
    meta["status"] = status.value
    if progress is not None:
        meta["progress_percent"] = progress
    if stage is not None:
        meta["stage"] = stage
    if message is not None:
        meta["message"] = message
    if error_code is not None:
        meta["error_code"] = error_code
    if extra:
        meta.update(extra)
    save_meta(scan_id, meta)
    return meta


def delete_scan(scan_id: str) -> None:
    root = settings.data_dir / scan_id
    if root.exists():
        shutil.rmtree(root)


def list_scans() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted(settings.data_dir.iterdir()):
        meta_file = path / "meta.json"
        if meta_file.exists():
            items.append(json.loads(meta_file.read_text(encoding="utf-8")))
    return items
