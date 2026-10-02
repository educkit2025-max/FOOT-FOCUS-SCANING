# Backend (Milestone 1)

Local Python API for the standalone foot scanner.

## Setup (Windows)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Requires **FFmpeg** on PATH.

## Run

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open docs: http://127.0.0.1:8000/docs

For Pixel 3a on the same Wi‑Fi, use your PC LAN IP, e.g. `http://192.168.x.x:8000`.

## Milestone 1 flow

1. `POST /api/v1/scans` — create scan (`foot`: left/right)
2. `POST /api/v1/scans/{id}/video` — upload mp4/webm
3. `POST /api/v1/scans/{id}/process` — extract + select frames
4. `GET /api/v1/scans/{id}/status` — poll progress
5. `GET /api/v1/scans/{id}/frames` — selected thumbnails

Scan packages are stored under `backend/data/scans/`.
