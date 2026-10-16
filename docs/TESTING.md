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

## Length & width in cm (current)

After upload the backend now:

1. Selects ~70 frames  
2. Detects **A4 paper** (210×297 mm) for metric scale  
3. Segments the foot  
4. Computes **length_cm** and **width_cm** (median across frames)  
5. Returns confidence + notes  

**Required for accurate cm values:** keep a white **A4 sheet** under/around the foot during the 30s scan.

Results appear on the phone **Result** screen and in:

`backend/data/scans/{scan_id}/measurements.json`


## Colour 3D foot (FOCUS)

Measurements always come from the A4-calibrated agents. The 3D foot on the result page is built with
FOCUS-SfM ([OllieBoyne/FOCUS](https://github.com/OllieBoyne/FOCUS)) from 20 frames spread over the
30 s video, then scaled to the A4-measured foot length:

1. The FOCUS network predicts, per frame, which point of a template foot each pixel shows (TOC),
   the surface normal and the foot mask.
2. Matching TOC values link the frames; COLMAP (`pycolmap`) solves the camera positions.
3. Matched points are triangulated, filtered and meshed (screened Poisson) with the skin colour
   from the frames.

Setup (once) — see `backend/requirements-focus.txt`. The model file (277 MB) goes to
`backend/data/models/focus_toc_predictor.pth`. Without it the app shows the calibrated two-view shape.

On this PC (i5-7200U, no GPU) the colour 3D step takes about 3–4 minutes per scan.

Licence: FOCUS is MIT, but its network code comes from DSINE (Imperial College), which allows only
non-commercial, internal or academic research use. Commercial use of NIKONYX needs a licence from
researchcontracts.engineering@imperial.ac.uk, or a retrained/replaced network.

## Branding
Splash + loading use the **Nikonyx** logo. This remains a standalone test module until you approve Nikonyx integration.
