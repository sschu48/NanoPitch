# NanoPitch Coach — Quick Start

Browser-based singing coach built on top of the NanoPitch model. Pick a
song, sing along to the metronome, get a per-note quality report.

The coach is designed around four independent axes (pitch, tempo,
technique, dynamics), each detector-first: pitch and tempo are *graded*
against the score; technique and dynamics are *detected and reported*
without judging quality. Reports become coaching messages later (v5)
when reference recordings supply per-note targets.

See [COACH.md](COACH.md) for the four-axis framing, phased roadmap,
and scoring math.

## Run locally

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

v1 scaffolding only. UI structure and stub functions are in place;
recording, scoring, and report rendering are not yet implemented.
