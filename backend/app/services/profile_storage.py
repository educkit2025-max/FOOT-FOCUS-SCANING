"""On-disk layout for multi-view foot profiles.

backend/data/profiles/{profile_id}/
    profile.json, status.json, result.json, mesh_{side}.json, mesh_{side}.obj
    views/{view}/capture_{i}.jpg, view.json, thumbnail.jpg
    debug/{view}_rectified.jpg
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from app.agents.imaging import downscale, load_image_bytes
from app.config import settings

REQUIRED_VIEWS = ["left_top", "left_medial", "right_top", "right_medial"]
OPTIONAL_VIEWS = ["left_oblique", "right_oblique", "left_rear", "right_rear"]
ALL_VIEWS = REQUIRED_VIEWS + OPTIONAL_VIEWS
MAX_CAPTURES = 5


def profiles_root() -> Path:
    root = settings.data_dir.parent / "profiles"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def profile_dir(profile_id: str) -> Path:
    if len(profile_id) != 32 or not all(c in "0123456789abcdef" for c in profile_id):
        raise ValueError("invalid profile id")
    return profiles_root() / profile_id


def exists(profile_id: str) -> bool:
    try:
        return profile_dir(profile_id).exists()
    except ValueError:
        return False


def create_profile(meta: dict) -> str:
    pid = uuid.uuid4().hex
    d = profile_dir(pid)
    (d / "views").mkdir(parents=True)
    _write_json(d / "profile.json", {"profile_id": pid, "created_at": _now(), **meta})
    write_status(pid, {"state": "capturing", "stage": None, "progress": 0.0, "stages": []})
    return pid


def view_dir(profile_id: str, view: str) -> Path:
    if view not in ALL_VIEWS:
        raise ValueError(f"unknown view {view}")
    return profile_dir(profile_id) / "views" / view


def save_view(profile_id: str, view: str, images: list[bytes], meta: dict) -> dict:
    d = view_dir(profile_id, view)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for i, data in enumerate(images[:MAX_CAPTURES]):
        (d / f"capture_{i}.jpg").write_bytes(data)
    thumb, _ = downscale(load_image_bytes(images[0]), 360)
    cv2.imwrite(str(d / "thumbnail.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 82])
    info = {"view": view, "captures": min(len(images), MAX_CAPTURES), "uploaded_at": _now(), **meta}
    _write_json(d / "view.json", info)
    return info


def view_captures(profile_id: str, view: str) -> list[Path]:
    d = view_dir(profile_id, view)
    return sorted(d.glob("capture_*.jpg")) if d.exists() else []


def view_meta(profile_id: str, view: str) -> dict | None:
    return _read_json(view_dir(profile_id, view) / "view.json")


def uploaded_views(profile_id: str) -> list[str]:
    return [v for v in ALL_VIEWS if view_captures(profile_id, v)]


def thumbnail_path(profile_id: str, view: str) -> Path:
    return view_dir(profile_id, view) / "thumbnail.jpg"


def debug_dir(profile_id: str) -> Path:
    d = profile_dir(profile_id) / "debug"
    d.mkdir(exist_ok=True)
    return d


def read_profile(profile_id: str) -> dict | None:
    return _read_json(profile_dir(profile_id) / "profile.json")


def write_status(profile_id: str, status: dict) -> None:
    _write_json(profile_dir(profile_id) / "status.json", {**status, "updated_at": _now()})


def read_status(profile_id: str) -> dict | None:
    return _read_json(profile_dir(profile_id) / "status.json")


def write_result(profile_id: str, result: dict) -> None:
    _write_json(profile_dir(profile_id) / "result.json", result)


def read_result(profile_id: str) -> dict | None:
    return _read_json(profile_dir(profile_id) / "result.json")


def write_mesh(profile_id: str, side: str, mesh: dict, obj_text: str) -> None:
    d = profile_dir(profile_id)
    _write_json(d / f"mesh_{side}.json", mesh)
    (d / f"mesh_{side}.obj").write_text(obj_text, encoding="utf-8")


def mesh_path(profile_id: str, side: str, ext: str) -> Path:
    return profile_dir(profile_id) / f"mesh_{side}.{ext}"


def delete_profile(profile_id: str) -> None:
    shutil.rmtree(profile_dir(profile_id), ignore_errors=True)
