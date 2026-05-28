# Technique Axis Developer Setup

This branch is set up so a developer can work on the technique-model pipeline
without downloading datasets or training models first.

## What This Branch Contains

- GT Singer technique-model code and packaged demo checkpoint.
- Detection-first API output: dominant techniques, uncertainty, and coarse
  technique timeline.
- Normalized JSONL manifest schema and manifest builders.
- Dataset registry and model-development roadmap.
- Lightweight no-data checks that avoid PyTorch and dataset downloads.

## Prerequisites

- Python 3.11 or compatible Python 3.x
- Node.js, for browser JavaScript syntax checks
- `make`

No Python package install is required for the light checks.

Optional later:

- PyTorch and NumPy for inference/training:

```bash
python3 -m pip install -r server/technique/gt_singer_grader/requirements-demo.txt
```

- Full NanoPitch training dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## First Command

From the repo root:

```bash
make technique-check-light
```

This runs:

- Python syntax checks for the technique API and grader modules
- JavaScript syntax checks for the coach browser app
- stdlib-only unit tests for manifest tooling
- app-recordings CSV -> normalized JSONL manifest smoke test
- JSONL manifest validation

## Files To Read First

1. `MODEL_DEVELOPMENT.md` for the goal, data strategy, training stages, and
   evaluation gates.
2. `dataset_registry.json` for candidate datasets and their role in the plan.
3. `manifest.py` for the normalized manifest contract.
4. `build_manifest.py` for currently supported manifest builders.
5. `train.py` for the current GT Singer training loop.
6. `feedback.py` and `server/technique/api.py` for inference-time output shape.

## Working Without Datasets

Use the app-recording label template to test manifest plumbing:

```bash
cd server/technique
python3 -m gt_singer_grader.build_manifest app-recordings \
  --csv gt_singer_grader/app_recordings_labels_template.csv \
  --output /tmp/nanopitch-technique/app_recordings_manifest.jsonl

python3 -m gt_singer_grader.manifest /tmp/nanopitch-technique/app_recordings_manifest.jsonl
```

The CSV columns are:

```text
audio_path,recording_id,singer_id,song_id,families,techniques,split_group,label_source,notes
```

## Running The Existing Demo Later

The browser coach app does not require training data:

```bash
python3 -m http.server 8080
```

Open:

```text
http://127.0.0.1:8080/coach/web/
```

The optional technique API requires PyTorch:

```bash
python3 server/technique/api.py --port 8765
```

If the API is offline, the browser still reports pitch, tempo, and dynamics.

## Training Later

After downloading GT Singer, train a first honest baseline with speaker-held-out
validation and user-recording augmentation:

```bash
cd server/technique
python3 -m gt_singer_grader.train \
  --dataset-root ./gt_singer_grader/data/GTSinger \
  --output-dir ./gt_singer_grader/runs/gtsinger_speaker_aug_v1 \
  --split-group speaker \
  --user-audio-augmentation \
  --epochs 50 \
  --batch-size 8
```

Do not treat random clip-level validation as product evidence. For user-uploaded
recordings, speaker-held-out and app-recording validation are required.

## Current Known Gaps

- Raw GT Singer data is not committed and not downloaded by this branch.
- PyTorch is optional for light dev and is not required for the first checks.
- Real app-style labeled recordings still need to be collected.
- The current checkpoint is demo-grade, trained on clean GT Singer English data.
- Future work should add dataset-specific builders for VocalSet/CVT/SVQTD only
  after license/taxonomy review.
