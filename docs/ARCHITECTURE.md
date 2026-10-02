# Standalone Foot Scanner Architecture

**Product:** Independent Android test app + local Python engine.  
**Not** integrated into Nikonyx until Milestone 8 validation passes.  
**Test device:** Google Pixel 3a (single rear camera — ideal for fixed 1× capture).

## Capture logic (critical)

| Correct | Incorrect |
|---------|-----------|
| Phone moves around a **stationary**, weight-bearing bare foot | Rotating / lifting the foot mid-scan |
| ~15–20 s video, 1080p @ 30 fps when possible | Processing every frame blindly |
| Calibration mat visible for metric scale | Guessing size from appearance / LLM |

Demo reference (`docs/demo_frames/`): FOCUS splash shows multi-view photos → segmented foot → 3D mesh. Same principle: many viewpoints of a rigid foot.

## Layers

```
┌─────────────────────────────────────────────┐
│  Android (Kotlin / Compose / CameraX)       │
│  guide → record → QA → upload → view result │
└─────────────────────┬───────────────────────┘
                      │ Wi‑Fi (dev) / HTTPS later
┌─────────────────────▼───────────────────────┐
│  FastAPI on PC                              │
│  /scans CRUD + video + status + result      │
└─────────────────────┬───────────────────────┘
                      │ background job
┌─────────────────────▼───────────────────────┐
│  Pipeline workers                           │
│  frames → quality → FOCUS → scale → measure │
└─────────────────────────────────────────────┘
```

## Coordinate convention (target)

- **Y** = heel → toe (heel near Y=0)
- **X** = medial ↔ lateral
- **Z** = floor normal (up)
- Units after scale: **millimetres**

## Calibration policy

1. **Primary:** printable mat (ArUco/AprilTag + known border + heel line)  
2. **Fallback:** plain A4 (210×297 mm)  
3. Intrinsics: Camera2 metadata when available → COLMAP refine otherwise

Target scan: **~30 seconds**, then keep about **60–80** selected frames.
