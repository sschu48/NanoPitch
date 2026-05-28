# Project 2 Final MVP

This branch turns Project 2 into one polished free-take analysis flow. The
system records one audio take, treats that WAV as the shared input artifact,
and produces one detection report across the available axes.

## Current status

| Area | Status | Where to look |
|---|---|---|
| Browser app | Free-take detector dashboard | `coach/web/` |
| Pitch | Detected from recorded take | NanoPitch WASM |
| Tempo | Detected from recorded take | fused novelty + autocorrelation |
| Dynamics | Detected from recorded take | RMS dBFS range |
| Technique | Optional local API | `server/technique/api.py` |

This MVP is intentionally detection-first. We do not have a song score,
reference performance, authored dynamics target, or validated good/bad
technique target for arbitrary free takes, so the report uses measured
properties instead of grades.

## Report contract

Every axis reports the same shape:

```json
{
  "axis": "pitch",
  "mode": "detection",
  "available": true,
  "headline": "Pitch contour detected",
  "feedback": "Detection-first summary text.",
  "metrics": {},
  "timeline": []
}
```

The browser combines the axis results into one `free_take_detection` report.

## How to run the browser app

```bash
# from repo root
python3 -m http.server 8080
```

Open:

```text
http://127.0.0.1:8080/coach/web/
```

The app loads `../../deployment/web/model.json` automatically. Record a take,
stop recording, and the app analyzes the recorded 16 kHz WAV.

## How to run the technique API

In another terminal, install the demo requirements if needed:

```bash
python3 -m pip install -r server/technique/gt_singer_grader/requirements-demo.txt
```

Then start the API:

```bash
python3 server/technique/api.py --port 8765
```

Health check:

```text
http://127.0.0.1:8765/health
```

If the technique API is offline, the browser still returns pitch, tempo, and
dynamics. The technique card reports that the service is unavailable.

## Axis details

- **Pitch:** voiced percent, median f0, median note, pitch range, and
  short-term stability.
- **Tempo:** onset count, estimated BPM, BPM confidence, and onset spacing.
- **Dynamics:** average level, peak level, p5/p95 loudness, and range used.
- **Technique:** detected family, confidence, voiced percent, technique
  strengths, and family probabilities from the GT Singer model.

## What is intentionally not done yet

- This is not song grading yet. Pitch and tempo need a score/MIDI target.
- Dynamics needs authored markings or a reference take before it can be judged.
- Technique is PyTorch-backed through a local service, not browser-native.
- The original detector probe remains useful for low-level debugging, but the
  delivery app is now `coach/web/`.

## Next steps

1. Add song/reference mode so pitch and tempo can be scored against targets.
2. Add reference-take comparison for dynamics.
3. Decide whether technique stays as a Python service or moves to ONNX/WASM.
4. Add saved report export once the report schema stabilizes.
