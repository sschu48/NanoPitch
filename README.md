# NanoPitch Coach

NanoPitch Coach is a browser-based singing-coach demo. It records one free
take, analyzes that same WAV, and returns a four-axis feedback report:

- pitch
- tempo
- dynamics
- vocal technique

The project is detection-first. It gives objective post-take feedback about
what happened in the recording; it does not grade against a MIDI score,
reference singer, or teacher-authored lesson target.

## Problem We Tried To Solve

Beginner singers often do not know what changed between takes. Our goal was to
give immediate, objective feedback after a short recording so a singer can see:

- whether they produced stable voiced pitch
- whether their attacks imply a consistent tempo
- whether they used dynamic contrast
- what vocal technique the model detects

This is not meant to replace a vocal coach. It is a lightweight feedback tool
that makes one recorded take easier to inspect.

## Run The Project

From the repo root, start the browser app:

```bash
python3 -m http.server 8080
```

Open:

```text
http://127.0.0.1:8080/coach/web/
```

In a second terminal, start the technique service:

```bash
make technique-api
```

With both commands running, the app records one take and shows all four axes.
If the technique API is not running, the app still shows pitch, tempo, and
dynamics, and marks technique as unavailable.

## Pipeline

```text
Browser microphone
  -> record one 16 kHz WAV
  -> run NanoPitch WebAssembly pitch/VAD model in the browser
  -> compute pitch, tempo, and dynamics summaries in JavaScript
  -> send the same WAV to the local PyTorch technique API
  -> return one unified four-axis report
```

## What Each Axis Reports

| Axis | What it reports | Implementation |
|---|---|---|
| Pitch | voiced percent, median f0, median note, range, stability | NanoPitch WASM pitch/VAD model |
| Tempo | onset count, estimated BPM, onset spacing | browser DSP novelty + autocorrelation |
| Dynamics | average/peak loudness and dynamic range | RMS dBFS contour |
| Technique | detected technique family, confidence, frame-level technique scores | local PyTorch API |

## Data And Models

| Component | Data / model source | How we used it |
|---|---|---|
| Pitch/VAD | GTSinger vocal data | training audio for the browser pitch/VAD model |
| Pitch labels | RMVPE-derived f0 labels | label generation, not live inference |
| Noise robustness | FSDNoisy18k environmental noise | random-SNR augmentation during pitch/VAD training |
| Technique | GT Singer English technique/control data | trained the packaged technique checkpoint |
| Technique expansion plan | VocalSet, DAMP-style recordings, app recordings | documented and scaffolded, but not used in the submitted checkpoint |

The pitch/VAD model is our NanoPitch causal conv + GRU model exported to
`deployment/web/model.json` for browser inference.

The technique model is a custom GT Singer technique classifier in
`server/technique/gt_singer_grader/`. It is not an off-the-shelf pretrained
technique model. The submitted checkpoint is packaged at:

```text
server/technique/gt_singer_grader/models/technique_demo_best.pth
```

## Libraries Used

- Browser app: JavaScript, Web Audio API, Canvas, WebAssembly
- Pitch/VAD deployment: NanoPitch WASM runtime in `deployment/web/`
- Technique model/API: Python, PyTorch, NumPy, SciPy
- Training workflow: TensorBoard and tqdm
- Validation: Node.js, Python `unittest`, GitHub Actions

## Development Approach

We built the project in stages:

1. Record a clean browser WAV from the microphone.
2. Run the existing NanoPitch model in-browser for frame-level VAD and pitch.
3. Build the first three report axes locally: pitch, tempo, and dynamics.
4. Add the fourth axis by sending the same WAV to a local technique API.
5. Keep the report shape consistent so all axes render as the same kind of
   feedback card.
6. Add no-data validation checks, manifest tooling, and documentation so the
   full project is reproducible from GitHub.

## Accuracy And Testing

### Browser Axis Validation

Run:

```bash
node validation/run_validation.js
```

Current browser-path validation passes all 4 synthetic fixtures:

| Fixture | Expected | Actual |
|---|---|---|
| `pitch_a4_harmonic` | 440 Hz | 436.5 Hz median f0 |
| `tempo_120bpm_pulses` | 120 BPM | 120 BPM |
| `dynamics_constant` | low contrast | 0.7 dB range |
| `dynamics_soft_loud_soft` | clear contrast | 13.8 dB range |

### Pitch/VAD Model

We added log-domain random-SNR noise augmentation in `training/train.py`. On
the 600-clip NanoPitch held-out test set:

| Metric | Baseline | Augmented model |
|---|---:|---:|
| Overall VAD accuracy | 86.8% | 94.9% |
| Offline raw pitch accuracy | 91.8% | 90.7% |
| Realtime raw pitch accuracy | 89.8% | 87.8% |
| -5 dB median pitch error | 73.2 cents | 47.1 cents |

The main gain was better singing-vs-silence detection under noise. The tradeoff
was a small drop in raw pitch accuracy.

### Technique Model

The submitted technique checkpoint was trained on GT Singer English data:

- best epoch: 76
- validation split: 10%
- clip accuracy: 65.8%
- technique macro F1: 0.497

The technique axis is fully integrated and runnable, but it is the least mature
part of the project. The main issue was data quality and time: we did not have
enough high-quality, labeled app-style WAV recordings to train and validate a
reliable user-recording technique model. Public datasets help, but they do not
perfectly match our target app recordings or technique taxonomy.

What we added instead:

- GT Singer baseline training/inference workflow
- app-recording label templates
- manifest builders and validators
- dataset strategy for VocalSet, DAMP-style recordings, and future app data
- packaging and verification tooling for future improved checkpoints

So the submitted technique model should be treated as a working demo-grade
fourth axis, not a production vocal-technique judge.

## Submitted GitHub Contents

Everything needed for the class project is on `main`:

| Path | Purpose |
|---|---|
| `coach/web/` | browser UI and four-axis report |
| `deployment/web/` | NanoPitch WASM runtime and browser model |
| `server/technique/` | local technique API |
| `server/technique/gt_singer_grader/` | technique model, checkpoint, training/evaluation workflow |
| `training/` | NanoPitch pitch/VAD training code |
| `validation/` | synthetic validation harness |
| `.github/workflows/` | light CI check |
| `RESULTS.md` | detailed pitch/VAD augmentation results |

Large raw datasets are not committed. The repo includes code, packaged browser
assets, the packaged technique demo checkpoint, documentation, and validation
tooling.

## Demo Script

1. Run `python3 -m http.server 8080`.
2. Run `make technique-api` in a second terminal.
3. Open `http://127.0.0.1:8080/coach/web/`.
4. Record a 10-15 second singing take.
5. Show that the same recording drives pitch, tempo, dynamics, and technique.
6. Explain the key limitation: technique detection works end-to-end, but needs
   more labeled app-style WAVs before it can be called accurate.
