"""API request/response models for Milestone 1."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FootSide(str, Enum):
    left = "left"
    right = "right"


class ScanStatus(str, Enum):
    created = "created"
    uploading = "uploading"
    ready = "ready"
    processing = "processing"
    frames_ready = "frames_ready"
    complete = "complete"
    failed = "failed"


class DeviceInfo(BaseModel):
    manufacturer: str | None = None
    model: str | None = None
    camera_id: str | None = None
    resolution: str | None = None
    focal_length_mm: float | None = None
    sensor_long_mm: float | None = None


class CaptureMeta(BaseModel):
    duration_seconds: float | None = None
    target_fps: int | None = 30
    weight_bearing: bool = True


class CreateScanRequest(BaseModel):
    foot: FootSide
    device: DeviceInfo = Field(default_factory=DeviceInfo)
    capture: CaptureMeta = Field(default_factory=CaptureMeta)
    notes: str | None = None


class CreateScanResponse(BaseModel):
    scan_id: str
    status: ScanStatus
    created_at: datetime


class ProcessRequest(BaseModel):
    target_frames: int = Field(default=70, ge=5, le=120)


class FrameInfo(BaseModel):
    index: int
    filename: str
    thumbnail_url: str
    sharpness: float
    selected: bool
    discard_reason: str | None = None


class FramesResponse(BaseModel):
    scan_id: str
    original_frame_count: int
    selected_frame_count: int
    frames: list[FrameInfo]
    best_frames: list[dict[str, Any]] = Field(default_factory=list)


class StatusResponse(BaseModel):
    scan_id: str
    status: ScanStatus
    progress_percent: int = 0
    stage: str = "idle"
    message: str = ""
    error_code: str | None = None
    foot: FootSide | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    capture: dict[str, Any] = Field(default_factory=dict)
    frame_summary: dict[str, Any] = Field(default_factory=dict)
    measurements: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)
    stages: list[dict[str, Any]] = Field(default_factory=list)


class MeasurementValue(BaseModel):
    value: float | None = None
    confidence: float = 0.0
    unit: str = "cm"


class ResultResponse(BaseModel):
    scan_id: str
    foot: FootSide | None = None
    status: ScanStatus
    milestone: int = 4
    measurements: dict[str, Any]
    quality: dict[str, Any]
    frame_summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    mesh: dict[str, Any] | None = None
    notes: list[str] = Field(default_factory=list)
