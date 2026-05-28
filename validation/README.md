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

These tests are not meant to prove subjective singing quality. They are meant
to catch obvious detector regressions:

- a known 440 Hz tone should be detected near 440 Hz,
- regular 120 BPM pulses should produce onsets and a tempo estimate near 120,
- quiet/loud/quiet audio should show more dynamic range than constant audio.
