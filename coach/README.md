# NanoPitch Coach - Quick Start

Browser-based Project 2 MVP built on top of the NanoPitch model. Record one
free take, analyze that same take, and view one detection report across pitch,
tempo, dynamics, and optional technique.

The current app is detection-first, not grading-first. It measures what
happened in the take because we do not yet have a song score, reference
performance, or validated quality target for arbitrary free-take grading.

See [COACH.md](COACH.md) for the longer four-axis roadmap.

## Run the browser app

The browser app is static. It loads NanoPitch WASM/model assets from
`deployment/web`.

```bash
# from repo root
python3 -m http.server 8080
# open http://127.0.0.1:8080/coach/web/
```

## Run the optional technique API

```bash
# from repo root
python3 server/technique/api.py --port 8765
```

If `torch` is missing, install the demo requirements first:

```bash
python3 -m pip install -r server/technique/gt_singer_grader/requirements-demo.txt
```

The browser still works without the technique API; only the technique axis is
marked unavailable.

## Microphone permissions

The browser will prompt for mic access on first record. Chrome and
Safari both support `getUserMedia` over `localhost` without HTTPS.

## Status

`coach/web` is now the primary Project 2 delivery app. `deployment/web` remains
the lower-level live detector/debug page.
