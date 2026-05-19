# NanoPitch Coach - Quick Start

Browser-based singing coach built on top of the NanoPitch model. Pick a
song, sing along to the metronome, get a per-note quality report.

The coach is designed around four independent axes (pitch, tempo,
technique, dynamics), each detector-first: pitch and tempo are *graded*
against the score; technique and dynamics are *detected and reported*
without judging quality. Reports become coaching messages later (v5)
when reference recordings supply per-note targets.

See [COACH.md](COACH.md) for the four-axis framing, phased roadmap,
and scoring math.

For the current shareable demo split, see
[`../PROJECT2_SUBMISSION.md`](../PROJECT2_SUBMISSION.md).
In short:

- `detector-loudness-clarity` has the live browser detector probe for pitch,
  tempo, and loudness.
- `origin/brady_dev` has Brady's separate GT Singer technique prototype.
- Dynamics grading and technique integration are intentionally not merged into
  the live browser detector yet.

## Run the coach scaffold

The coach is a static web app — no build step, no backend.

```bash
# from repo root
cd coach/web
python3 -m http.server 8080
# open http://localhost:8080 in Chrome or Safari
```

The page expects to find:
- `../../deployment/web/nanopitch.js` and `nanopitch.wasm` (committed)
- `../../training/runs/v1_aug/model.json` (regenerate if missing — see below)

## Run the live detector probe

The current detector-first status is easiest to see in the live detector page:

```bash
# from repo root
cd deployment/web
python3 -m http.server 8080
# open http://127.0.0.1:8080
```

Then drop `deployment/web/model.json` onto the page and start the microphone.

This probe covers:

- Pitch: NanoPitch f0/VAD
- Tempo: live onset/BPM detector prototype
- Loudness: live RMS detector prototype

It does not yet implement a finished Dynamics grading axis. It also shows a
clarity diagnostic, but clarity is not the planned Technique axis. The
Technique prototype currently lives separately on `origin/brady_dev`.

## Regenerating the model JSON

If `training/runs/v1_aug/model.json` is missing (it's gitignored — too
large to track), regenerate it from the checkpoint:

```bash
# from repo root, with .venv active
python deployment/export_weights.py \
    training/runs/v1_aug/checkpoints/best.pth \
    -o training/runs/v1_aug/model.json
```

You'll need `training/runs/v1_aug/checkpoints/best.pth`, which means you
need to have run training. See the project 1 README for that flow.

## Microphone permissions

The browser will prompt for mic access on first record. Chrome and
Safari both support `getUserMedia` over `localhost` without HTTPS.

## Status

The song-aware coach UI is still scaffold/integration work. The shareable
detector status is split across the live NanoPitch/tempo/loudness browser
probe on this branch and Brady's separate GT Singer technique prototype on
`origin/brady_dev`.
