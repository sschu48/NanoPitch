# NanoPitch Coach

NanoPitch Coach is a browser-based singing analysis MVP. It records one free
take, analyzes that same WAV, and returns objective feedback for pitch, tempo,
dynamics, and vocal technique.

## 30-Second Summary

- **What we built:** a local singing-coach demo that turns one recorded take
  into a four-axis feedback report.
- **Coaching problem:** give a beginner immediate objective feedback after
  singing, before attempting full song/reference grading.
- **How it works:** browser mic -> 16 kHz WAV -> NanoPitch WASM model -> local
  analyzer -> pitch/tempo/dynamics cards; the same WAV is sent to a local
  PyTorch technique API for the fourth card.
- **What it "grades":** this MVP is detection-first. It reports measured pitch,
  rhythm, loudness, and technique signals. It does not yet score against a MIDI
  melody, reference singer, or teacher-authored target.
- **Main validation:** 4/4 synthetic browser-path tests pass; NanoPitch VAD
  improved from 86.8% to 94.9% after random-SNR augmentation.

## Run The Demo

From the repo root:

```bash
python3 -m http.server 8080
```

Open:

```text
http://127.0.0.1:8080/coach/web/
```

In a second terminal, start the technique service for the fourth axis:

```bash
make technique-api
```

With the technique service running, the same recorded WAV produces four report
cards: pitch, tempo, dynamics, and technique. If the technique API is not
running or times out, the browser app still completes the report and marks the
technique axis unavailable.

For the final class demo, run both commands so all four axes are active.

## What The App Reports

| Axis | What we measure | Implementation |
|---|---|---|
| Pitch | voiced percent, median f0, median note, range, short-term stability | NanoPitch WASM pitch/VAD model |
| Tempo | onset count, estimated BPM, onset spacing | browser DSP novelty + autocorrelation |
| Dynamics | average/peak loudness and range used | RMS dBFS contour |
| Technique | detected technique family and confidence | local GT Singer PyTorch API |

## Pipeline

```text
Browser microphone
  -> recorded 16 kHz free-take WAV
  -> NanoPitch WebAssembly inference
  -> per-frame VAD, f0, posteriorgram, mel, RMS
  -> local report builder for pitch, tempo, dynamics
  -> POST /analyze to local technique API
  -> unified four-axis free_take_detection report
```

## Data, Models, And Libraries

| Category | What we used |
|---|---|
| Training data | GTSinger vocal features, FSDNoisy18k environmental noise |
| Pitch labels | RMVPE-derived f0 labels from the NanoPitch feature set |
| Pitch/VAD model | NanoPitch causal conv + GRU model exported to `deployment/web/model.json` |
| Technique model | GT Singer technique classifier in `server/technique/gt_singer_grader/` |
| Browser stack | plain JavaScript, Web Audio API, Canvas/SVG/HTML, WebAssembly |
| Python stack | PyTorch, NumPy, SciPy, TensorBoard, tqdm |
| Validation stack | Node.js using the same `coach/web/analyzer.js` and WASM model path |

Large raw datasets are not committed. The repo includes download scripts,
evaluation summaries, browser deployment assets, and the packaged demo
technique checkpoint.

## Accuracy And Testing

### NanoPitch Evaluation

We implemented log-domain random-SNR noise augmentation in
`training/train.py`. On the 600-clip NanoPitch held-out test set:

| Metric | Baseline | Augmented model |
|---|---:|---:|
| Overall VAD accuracy | 86.8% | 94.9% |
| Offline raw pitch accuracy | 91.8% | 90.7% |
| Realtime raw pitch accuracy | 89.8% | 87.8% |
| -5 dB median pitch error | 73.2 cents | 47.1 cents |

Interpretation: the model got much better at detecting singing vs. silence in
noise, with a small pitch-accuracy tradeoff.

### Browser Validation

Run:

```bash
node validation/run_validation.js
```

Current result: all 4 synthetic fixtures pass.

| Fixture | Expected | Actual |
|---|---|---|
| `pitch_a4_harmonic` | 440 Hz | 436.5 Hz median f0 |
| `tempo_120bpm_pulses` | 120 BPM | 120 BPM |
| `dynamics_constant` | low contrast | 0.7 dB range |
| `dynamics_soft_loud_soft` | clear contrast | 13.8 dB range |

### Technique Model

The submitted technique axis uses a packaged GT Singer English checkpoint at
`server/technique/gt_singer_grader/models/technique_demo_best.pth`. It reports
detected technique family, confidence, dominant frame-level technique scores,
and a coarse timeline. The checkpoint metadata reports 65.8% clip accuracy and
0.497 technique macro F1 on its held-out validation split.

This is the final fourth axis for the class project, but it is still
detection-first. We do not claim that it is a production vocal-coaching judge:
it has not yet been validated on a large set of real NanoPitch app recordings.

## Repo Map

| Path | Purpose |
|---|---|
| `coach/web/` | main Project 2 browser UI |
| `deployment/web/` | NanoPitch WASM runtime and browser model |
| `server/technique/` | local technique API and GT Singer technique-model workflow |
| `training/` | NanoPitch model, training, and evaluation code |
| `validation/` | synthetic browser-path validation harness |
| `RESULTS.md` | detailed NanoPitch augmentation results |
| `coach/COACH.md` | longer roadmap for future song/reference grading |

Ignored local folders such as `.venv/`, `data/`, `NanoPitch-technique/`,
`validation/audio/`, and `validation/results/` are not submission artifacts.

## Thursday Demo Script

1. Start `python3 -m http.server 8080`.
2. Open `http://127.0.0.1:8080/coach/web/`.
3. Record a 10-15 second take with a sustained note, clear attacks, and one
   louder section.
4. Show that the same recorded WAV drives every report card.
5. Point out median note/f0, voiced percent, BPM/onsets, and dynamics range.
6. Show the technique card from the local technique API: detected family,
   confidence, dominant technique scores, and target match if a focus technique
   is selected.
7. Close with the key limitation: this is detection feedback, not full song
   grading yet. Full grading needs a MIDI score, reference take, or authored
   target.
