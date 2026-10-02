# FOCUS Reuse Map

MIT licensed (Ollie Boyne, 2024). Retain `LICENSE` and citation in all derivatives.

## What FOCUS already provides

| Component | Path | Role in our pipeline |
|-----------|------|----------------------|
| Video → frames | `FOCUS/run_focus.py` `_frames_from_video`, `FOCUS/utils/video.py` | Baseline uniform frame sampling (default **20**). We wrap with quality-aware selection. |
| TOC / dense correspondence | `FOCUS/toc_prediction/` | Dense foot correspondences for matching + COLMAP custom matches. |
| Masks / normals helpers | `FOCUS/utils/mask.py`, `normals.py`, `image.py` | Reuse preprocessing (480×640 aspect resize). |
| COLMAP + TOC matches | `FOCUS/calibration/` | Camera pose estimation when intrinsics unknown. |
| Matching / triangulation | `FOCUS/matching/`, `FOCUS/utils/triangulation.py` | Multi-view geometry. |
| Fusion mesh (FOCUS-SfM) | `FOCUS/fusion/` | Primary reconstruction path (`--method sfm`). |
| Optim mesh (FOCUS-O) | `FOCUS/optim/` | Alternate path (`--method o`) for benchmarking. |
| Remeshing / outliers | `FOCUS/utils/remeshing.py`, `o3d_outlier_removal.py` | Post-process hooks. |
| Eval vs Foot3D | `FOCUS/eval/` | Research validation only — not production metrics. |
| Rendering | `FOCUS/vis/` | Debug visualization. |

## What FOCUS does **not** provide (we must build)

- Guided Android capture UX (CameraX, coverage ring, live QA)
- Printable ArUco/AprilTag calibration mat + detector
- Metric scale recovery (mm) and scale validation
- Anatomical coordinate system + landmarks
- Geometric measurements (length, widths, girths, volume)
- Per-measurement confidence / rescan logic
- FastAPI job API, GLB export for mobile viewer
- Pixel 3a–specific capture constraints (fixed 1× rear, no zoom)

## Reuse decision

- **Keep** FOCUS as an embedded engine under `FOCUS/` (do not rewrite SfM/fusion in Milestone 1–2).
- **Call** it from `backend/` workers after frame selection + calibration modules exist.
- **Extend** frame extraction: quality filter + viewpoint novelty before invoking FOCUS.
- **Benchmark** `--num_frames` at 20 / 30 / 40 / 60 on Pixel 3a scans.
