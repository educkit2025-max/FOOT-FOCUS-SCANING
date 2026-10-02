# Android Foot Scanner (Milestone 1)

Standalone Kotlin/Compose app for **Google Pixel 3a** testing.

## Open in Android Studio

1. Open folder `android/`
2. Let Gradle sync
3. Connect Pixel 3a (USB debugging on)
4. Set PC backend URL on the Home screen to `http://<your-pc-lan-ip>:8000`
5. Run `app`

## Milestone 1 behavior

- Choose left/right foot
- Preparation checklist (weight-bearing, foot still, phone moves)
- Record ~15–20 s rear-camera HD video (zoom locked to 1×)
- Upload to local FastAPI
- Show selected frame thumbnails returned by the PC

## Notes

- `usesCleartextTraffic=true` is enabled for LAN testing only
- Not integrated with Nikonyx
