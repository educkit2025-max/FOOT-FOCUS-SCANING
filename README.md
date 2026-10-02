# FOOT-FOCUS-SCANING

Standalone **Android 3D foot scanning & measurement** prototype for accuracy testing before Nikonyx integration.

Test device focus: **Google Pixel 3a**.

## Capture rule

Keep the **foot still** (weight-bearing). Move the **phone** around the foot for ~15–20 seconds. Do not rotate the foot mid-scan.

## Layout

| Path | Purpose |
|------|---------|
| `android/` | Kotlin / Compose / CameraX scanning app (Milestone 1) |
| `backend/` | FastAPI PC engine — upload, frame selection (Milestone 1+) |
| `FOCUS/` | Upstream multi-view reconstruction (MIT, Boyne & Cipolla) |
| `docs/` | Architecture, reuse map, plan |
| `calibration/` | Printable calibration mat (Milestone 3) |
| `validation/` | Ground-truth protocol (later) |

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [FOCUS reuse](docs/FOCUS_REUSE.md)
- [Implementation plan](docs/IMPLEMENTATION_PLAN.md)

## Milestone 1 quick start

### PC backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Android

Open `android/` in Android Studio → run on Pixel 3a → set backend URL to `http://<PC-LAN-IP>:8000`.

## FOCUS attribution

This repository includes code from **FOCUS — Multi-View Foot Reconstruction from Synthetically Trained Dense Correspondences** (3DV 2025), MIT License. See `LICENSE` and cite:

```
@inproceedings{boyne2025focus,
  title={FOCUS - Multi-View Foot Reconstruction from Synthetically Trained Dense Correspondences},
  author={Boyne, Oliver and Cipolla, Roberto},
  booktitle={2025 International Conference on 3D Vision (3DV)},
  year={2025}
}
```

Original FOCUS README content is preserved below for upstream usage.

---

<p align="center">
  <a href="http://ollieboyne.github.io/FOCUS">
        <img width=70% src="https://www.ollieboyne.com/FOCUS/images/logos/focus_v1.png">
  </a>
</p>

This repository contains the code for foot reconstruction using dense correspondences, as shown in the FOCUS paper.

## FOCUS set-up

1) Install FOCUS and dependencies: `pip install --editable .`
2) Download required data (TOC model, etc. — see original Downloads table in project docs).
3) If using uncalibrated images, install [COLMAP](https://colmap.github.io/install.html).

FOCUS can be run using `FOCUS/run_focus.py` with `--video_path` / `--source_folder` and `--method sfm|o`.
