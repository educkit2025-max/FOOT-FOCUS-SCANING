# How to test (Milestone 1) — Pixel 3a

## Frame rates (important)

| Stage | Rate | Meaning |
|-------|------|---------|
| Phone recording | **~30 frames/second** | Continuous **video**, not still photos. Target 1080p. |
| Scan length | **30 seconds** | ~900 video frames recorded |
| PC candidate extract | **8 frames/second** | Backend samples candidates from the video (~240) |
| Final selected set | **~70 frames** (target 60–80) | Best sharp/non-duplicate views for reconstruction later |

So: the phone records video at ~30 fps; the PC keeps about **60–80 useful stills** from the whole scan.

### Where images are stored (on the PC)

After upload + process, files are under:

`C:\Users\LENOVO\Desktop\FOOT-FOCUS-SCANING\backend\data\scans\{scan_id}\`

| Folder | Contents |
|--------|----------|
| `video/` | Original phone video |
| `frames/` | All candidate stills extracted from video |
| `selected/` | Final kept images (~60–80) |
| `thumbnails/` | Small JPEG previews shown in the app |

## Test steps

### 1) Start PC backend
```powershell
cd C:\Users\LENOVO\Desktop\FOOT-FOCUS-SCANING\backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open http://127.0.0.1:8000/docs to confirm it is up.

### 2) Find your PC Wi‑Fi IP
```powershell
ipconfig
```
Use the IPv4 of your Wi‑Fi adapter (example `192.168.1.42`).

### 3) Run the Android app
1. Open `android/` in Android Studio  
2. Connect **Pixel 3a** (USB debugging on)  
3. Run the app  
4. On Home, set backend URL to `http://YOUR_PC_IP:8000`  
5. New scan → Left/Right → Prep → Record **30s** (foot still, phone moves) → Upload  
6. Success = ~60–80 selected frame thumbnails appear  
7. Open `backend\data\scans\{scan_id}\selected\` on the PC to see the kept images

### 4) Optional API-only test (no phone)
Upload any short foot video to `/api/v1/scans` → `/video` → `/process` → `/frames` via the docs page.

## Branding
Splash + loading use the **Nikonyx** logo. This remains a standalone test module until you approve Nikonyx integration.
