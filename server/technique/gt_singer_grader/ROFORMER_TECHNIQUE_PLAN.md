# RoFormer-Style Technique Model Plan

## Objective

Retrain the singing-technique detector with a stronger non-causal
time-frequency encoder while preserving the existing NanoPitch technique API,
label taxonomy, manifest tooling, and evaluation workflow.

The goal is not to use BS-RoFormer for source separation. The app input is
already isolated singing. The goal is to adapt the useful part of that model
family: longer-context band/time-frequency encoding.

## Problem Statement

The current technique checkpoint uses a compact causal Conv1d plus GRU encoder
over log-mel features. That is efficient and useful as a demo baseline, but it
is likely underpowered for technique recognition because vocal techniques are
not usually single-frame events.

Technique cues often require:

- longer phrase context for vibrato and glissando
- harmonic and register context for falsetto, mixed voice, and pharyngeal tone
- high-frequency breath/noise structure for breathy singing
- relationships between frequency bands, not only compressed frame history
- non-causal context, since the app analyzes completed short recordings

## Target Model Shape

Keep the current output contract and replace only the encoder.

```text
raw singing WAV
  -> mono 16 kHz
  -> STFT magnitude or richer mel spectrogram
  -> band or mel-band projection
  -> small non-causal RoFormer/Transformer encoder
       - temporal attention over longer windows
       - band/frequency attention or band-mixed blocks
       - rotary or relative positional encoding
  -> voiced-frame-aware pooling
  -> heads:
       clip family classifier
       frame/window technique classifier
       VAD/voiced classifier
```

The first candidate should be a small classifier inspired by BS-RoFormer, not a
full source-separation model. There should be no mask decoder and no waveform
reconstruction objective in the first pass.

## Non-Goals

- Do not add a vocal/accompaniment separator; the expected input is solo
  singing.
- Do not classify "good singing" or vocal quality yet.
- Do not ship a new checkpoint based only on GT Singer validation.
- Do not replace the existing API response shape or front-end contract.
- Do not use a large separation-scale model unless the small classifier has a
  clear failure mode that needs more capacity.

## Objective Goals

1. Establish a reproducible Conv-GRU baseline on the exact split used for the
   new architecture comparison.
2. Implement a `band_roformer` or `tf_roformer` model variant that trains from
   the same manifests as `TechniqueGraderModel`.
3. Use longer non-causal listening windows, starting with 6-10 second clips.
4. Preserve the existing output heads so evaluation and API integration remain
   comparable.
5. Compare the new encoder against the baseline on the same validation
   manifest and operating-point gates.
6. Validate on app-style singing recordings before packaging any replacement
   checkpoint.

## Acceptance Gates

A RoFormer-style candidate is only worth promoting if it improves technique
detection without hiding regressions behind threshold changes.

Minimum promotion evidence:

- same train/validation split or documented split fingerprint
- complete `run_config.json`, `metrics_history.jsonl`, `best_metrics.json`,
  evaluation directory, and comparison report
- macro F1 improves over the Conv-GRU baseline
- top-2 family accuracy improves or stays flat
- control false-positive rate does not increase materially
- non-technique false-positive rate remains within the configured gate
- calibration does not get worse enough to make confidence misleading
- app-recording validation is ready and candidate beats or matches baseline on
  the app evaluation manifest

Suggested initial targets:

- macro F1 delta: at least `+0.03`
- top-2 accuracy delta: at least `+0.00`
- control false-positive-rate delta: no worse than `+0.02`
- non-technique false-positive rate: `<= 0.25`
- expected calibration error delta: no worse than `+0.03`

These are starting gates. Tighten them once the app-recording validation set is
large enough to make small differences meaningful.

## Experiment Roadmap

### Phase 0: Baseline Lock

- Run or identify the current Conv-GRU baseline evaluation.
- Record exact manifest hashes, split grouping, checkpoint path, and operating
  point.
- Treat this as the comparison anchor.

### Phase 1: Encoder Prototype

- Add a new model class without changing the current model.
- Keep the same output keys: `vad_logits`, `technique_logits`, `clip_logits`.
- Add shape/unit tests that check short, padded, and max-length clips.
- Add a training flag such as `--architecture conv_gru|band_roformer`.

### Phase 2: Feature Decision

Start with the lowest-risk feature path:

- Option A: 128-bin log-mel spectrogram for simpler compatibility.
- Option B: STFT magnitude split into frequency bands for closer BS-RoFormer
  behavior.

Choose Option A first if implementation speed matters. Move to Option B only if
mel features appear to be the limiting factor.

### Phase 3: GT Singer Architecture Comparison

- Train the new encoder on GT Singer with the same split strategy.
- Compare against the Conv-GRU baseline on the same validation manifest.
- Inspect per-family confusion, especially falsetto vs mixed voice and control
  false positives.

### Phase 4: App-Domain Validation

- Evaluate both baseline and candidate on the same app-recording evaluation
  manifest.
- Reject the candidate if it improves clean GT Singer metrics but worsens
  app-recording false positives.
- If app data is limited, use this phase to define the next collection plan.

### Phase 5: Packaging

- Package only after comparison and app-validation gates pass.
- Keep the old Conv-GRU checkpoint available as rollback.
- Update metadata to identify the architecture and training evidence.

## Implementation Tasks

- Add `BandRoFormerTechniqueModel` or equivalent model class.
- Add architecture selection in `train.py`.
- Save `architecture` and model hyperparameters in `model_kwargs`.
- Update `infer.py` to instantiate by architecture while preserving old
  checkpoint compatibility.
- Add tests for model loading, output shapes, and backwards compatibility.
- Add run-plan documentation in `EXPERIMENTS.md` once the first command is
  known.
- Compare candidates with the existing `evaluate.py`, `verify_evaluation.py`,
  and `compare_runs.py` tooling.

## Key Risks

- Data may be the limiting factor, not architecture.
- A larger encoder can overfit GT Singer and fail on app recordings.
- More context can improve technique detection but worsen latency and compute.
- STFT-band modeling may add complexity without improving the current labels.
- Technique labels may not be clean enough to reward a more expressive model.

The first implementation should therefore be small, measurable, and reversible.
