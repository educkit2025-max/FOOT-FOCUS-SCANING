# Implementation Plan

## Milestones

| # | Goal | Exit criteria |
|---|------|----------------|
| **1** | Android video → PC API → selected frames | Pixel 3a upload works; thumbnails returned |
| **2** | FOCUS reconstruction | Recognizable mesh + valid poses from phone video |
| **3** | Metric scale from calibration mat | Known object within tolerance in mm |
| **4** | Core measurements | length, max/heel width, instep height + overlays |
| **5** | Girths + volume | ball/instep girth, volume_cm³ |
| **6** | Confidence + rescan | per-measurement scores; clear reject/rescan |
| **7** | Polished UX + GLB viewer | full screen flow + 3D result |
| **8** | Formal validation | 3× repeatability + ground-truth table; then Nikonyx |

## Milestone 1 scope (implementing now)

### Backend
- FastAPI app with scan create / video upload / process / status / frames
- Disk storage under `backend/data/scans/{scan_id}/`
- FFmpeg frame extract + simple quality filter (Laplacian blur) + uniform/novelty subsample (~30 frames default)
- JPEG thumbnails for Android review

### Android
- Compose app targeting Pixel 3a (`minSdk 26`, `targetSdk 34`)
- Screens: Home → Choose foot → Prep → Record → Upload/Progress → Frame review
- CameraX video (prefer 1080p30, main rear camera, zoom locked to 1×)
- Configurable backend base URL (default LAN IP)

### Explicitly deferred
- FOCUS inference, COLMAP, measurements, GLB viewer polish, cloud deploy

## GitHub

Local work only unless you explicitly request a push to `educkit2025-max/FOOT-FOCUS-SCANING`.
