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

Do not start this app from `deployment/web/` if you want the submission UI.
Serving `deployment/web/` directly opens the lower-level live detector/debug
page, not the Project 2 coach dashboard.

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

## Validation

The follow-up validation harness in `validation/` checks the three browser
axes that do not require the technique service. It generates synthetic
known-answer WAV files, runs the same NanoPitch WASM/model path used by the
browser app, builds the same report shape from `coach/web/analyzer.js`, and
compares the resulting metrics against tolerances.

Run it from the repo root:

```bash
node validation/run_validation.js
```

The harness writes generated WAV fixtures and a visual HTML report to:

```text
validation/audio/
validation/results/index.html
```

Current baseline results:

| Fixture | Axis checked | Result |
|---|---|---|
| `pitch_a4_harmonic` | Pitch | 436.5 Hz median f0 vs. 440 Hz target, -13.8 cents |
| `tempo_120bpm_pulses` | Tempo | 120 BPM estimated vs. 120 BPM target, 11 onsets |
| `dynamics_constant` | Dynamics | 0.7 dB range, expected low contrast |
| `dynamics_soft_loud_soft` | Dynamics | 13.8 dB range, expected clear contrast |

All four synthetic fixtures currently pass. This validates basic detector
plumbing and catches regressions, but it does not replace future testing on
controlled human vocal clips.

## What is intentionally not done yet

- This is not song grading yet. Pitch and tempo need a score/MIDI target.
- Dynamics needs authored markings or a reference take before it can be judged.
- Technique is PyTorch-backed through a local service, not browser-native.
- The original detector probe remains useful for low-level debugging, but the
  delivery app is now `coach/web/`.

## Repository boundaries after merge

`main` should become the stable Project 2 MVP baseline after this PR merges.
Future work should branch from `main` instead of continuing from the old split
demo branches.

Recommended branch pattern:

```text
feature/*     browser UI, report, recording, and analysis improvements
technique/*   GT Singer model training, evaluation, export, and checkpoint work
```

Keep these boundaries clean:

- Browser/product work lives under `coach/web/`.
- NanoPitch WASM/model deployment assets stay under `deployment/web/`.
- Technique service and model code live under `server/technique/`.
- Training data, local worktrees, virtual environments, and experiment runs are
  not submission artifacts.

If the technique model improves later, make that a focused PR with:

- the model/code change,
- the updated packaged checkpoint or metadata,
- before/after validation metrics,
- a quick browser/API compatibility check.

## Next steps

1. Add song/reference mode so pitch and tempo can be scored against targets.
2. Add reference-take comparison for dynamics.
3. Decide whether technique stays as a Python service or moves to ONNX/WASM.
4. Add saved report export once the report schema stabilizes.
