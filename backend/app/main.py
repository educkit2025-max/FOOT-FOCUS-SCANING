"""Foot scanner backend — Milestone 1 FastAPI entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.scans import router as scans_router
from app.config import settings

app = FastAPI(
    title="Foot Focus Scanner API",
    version="0.1.0",
    description="Standalone foot scanning backend (Milestone 1: video upload + frame selection).",
)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")

origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scans_router)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "milestone": 1,
        "data_dir": str(settings.data_dir),
    }


@app.get("/system")
async def system_info() -> dict:
    import platform
    import shutil

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "ffprobe": bool(shutil.which("ffprobe")),
        "gpu_note": "FOCUS/CUDA checks added in Milestone 2",
    }
