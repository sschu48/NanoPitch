# Project 2 v1 Submission Snapshot

This PR is a snapshot of our current Project 2 progress. It is not meant to
claim that the full singing coach is finished. The goal is to make the current
state easy to review: what runs today, what is separate, and what comes next.

## Current status

| Area | Status | Where to look |
|---|---|---|
| NanoPitch pitch/VAD | Live in browser | `deployment/web/index.html` |
| Tempo | Live prototype | `deployment/web/index.html` |
| Loudness | Live prototype | `deployment/web/index.html` |
| Dynamics | Not implemented as a grading axis yet | Future work from the loudness signal |
| Technique | Separate PyTorch/GT Singer prototype | Brady's `origin/brady_dev` branch |
| Coach UI | Scaffolded, not fully integrated | `coach/web/` |

The live browser page currently shows NanoPitch pitch tracking, onset/tempo
estimation, and loudness level. It also includes a clarity diagnostic, but
clarity is not the planned Technique axis.

Technique is intentionally separate right now. Brady's model lives on
`origin/brady_dev`; this PR does not edit or merge into Brady's branch.

## How to run the live browser detector

From this branch:

```bash
cd deployment/web
python3 -m http.server 8080
```

Open:

```text
http://127.0.0.1:8080
```

Then drag `deployment/web/model.json` onto the page and click
`Start Microphone`.

This demo is the best current view of the live detector work:

- NanoPitch pitch and voice activity detection through WASM.
- Tempo/onset prototype from live audio novelty and BPM estimation.
- Loudness prototype from live RMS level.

## How to review the coach scaffold

The coach app is a static browser scaffold for the eventual song-aware
experience:

```bash
cd coach/web
python3 -m http.server 8080
```

Open:

```text
http://127.0.0.1:8080
```

The coach scaffold shows the intended product direction: preset songs,
recording flow, and report structure. The live detector outputs are not fully
wired into this coach flow yet.

## How to review Brady's technique prototype

Brady's technique model is a separate PyTorch/upload demo on `origin/brady_dev`.
To inspect it without modifying his branch, use a separate worktree:

```bash
git fetch origin
git worktree add ../NanoPitch-technique origin/brady_dev
cd ../NanoPitch-technique
pip install -r gt_singer_grader/requirements-demo.txt
python -m gt_singer_grader.demo \
  --checkpoint ./gt_singer_grader/models/technique_demo_best.pth \
  --port 8765 \
  --open-browser
```

Open:

```text
http://127.0.0.1:8765
```

The technique prototype should be treated as research progress, not as a
browser-integrated coach axis yet. It classifies GT Singer technique families
and still needs validation before being used as a general singing-quality
grader.

## What is intentionally not done yet

- Dynamics grading is not implemented. We have live loudness measurement, but
  not a finished "dynamics" score.
- Technique is not browser-native and is not merged into the live detector.
- Tempo is detected live, but it is not yet scored against song JSON note
  starts.
- The coach UI exists as a scaffold; the detector page is still the best
  working demo for the live axes.

## Next steps

1. Wire NanoPitch pitch output into the coach report.
2. Compare detected tempo/onsets against `start_beat` values in song JSON.
3. Convert loudness into a real dynamics grade only after deciding the scoring
   rule.
4. Decide whether Technique should become browser-native via export or remain
   a Python-backed service.
5. Combine the axes into one report once the detectors are stable.
