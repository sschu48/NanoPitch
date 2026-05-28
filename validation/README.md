# Validation Harness

This folder contains a lightweight validation harness for the three browser
axes that do not depend on the technique service:

- pitch
- tempo
- dynamics

The harness generates known-answer synthetic WAV files, runs the same
NanoPitch WASM/model path used by the browser app, builds the same local report
shape from `coach/web/analyzer.js`, and compares metrics against tolerances.

Run from the repo root:

```bash
node validation/run_validation.js
```

Generated artifacts:

```text
validation/audio/      generated WAV fixtures
validation/results/    JSON summary and visual HTML report
```

Open the visual report after a run:

```text
validation/results/index.html
```

The visual report includes:

- audio controls for each generated fixture,
- a pitch/loudness/onset chart,
- the detector metrics returned by the same report code used by the app,
- pass/fail rows for each expected property.

These tests are not meant to prove subjective singing quality. They are meant
to catch obvious detector regressions:

- a known 440 Hz tone should be detected near 440 Hz,
- regular 120 BPM pulses should produce onsets and a tempo estimate near 120,
- quiet/loud/quiet audio should show more dynamic range than constant audio.

## Current synthetic fixtures

| Fixture | Purpose | Expected |
|---|---|---|
| `pitch_a4_harmonic` | Pitch sanity check | Median f0 near 440 Hz, voiced percent high, low dynamic range |
| `tempo_120bpm_pulses` | Tempo/onset sanity check | Estimated BPM near 120, onset count in expected range |
| `dynamics_constant` | Dynamics low-range check | Loudness range stays low |
| `dynamics_soft_loud_soft` | Dynamics contrast check | Loudness range is clearly higher than the constant fixture |

## Latest baseline result

The initial baseline run on the merged Project 2 MVP passed all four fixtures:

```text
PASS pitch_a4_harmonic
  pitch median f0: expected 440 Hz +/- 75 cents, actual 436.5 Hz (-13.8 cents)
  pitch voiced percent: expected >= 45%, actual 61.9%
  dynamic range maximum: expected <= 7 dB, actual 0.6 dB

PASS tempo_120bpm_pulses
  tempo estimate: expected 120 BPM +/- 10, actual 120 BPM
  onset count: expected 7..14, actual 11

PASS dynamics_constant
  dynamic range maximum: expected <= 5 dB, actual 0.7 dB

PASS dynamics_soft_loud_soft
  dynamic range minimum: expected >= 9 dB, actual 13.8 dB
```

## What this does not validate yet

- Human singing accuracy. The fixtures are synthetic by design, so they catch
  regressions but do not replace real vocal test clips.
- Technique detection. Brady's PyTorch service is intentionally excluded from
  this harness for now.
- Song grading. The MVP does not compare against a melody, score, or reference
  take yet.

## Next validation step

Add a small set of controlled human WAV clips with metadata describing the
expected pitch center, approximate tempo, and loudness pattern. Those clips can
use the same comparison/report machinery as the synthetic fixtures.
