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

- Technique-model training dependencies:

```bash
python3 -m pip install -r server/technique/gt_singer_grader/requirements-training.txt
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
- dataset registry strategy audit
- app-recordings CSV -> normalized JSONL manifest smoke test
- JSONL manifest validation

## Files To Read First

1. `MODEL_DEVELOPMENT.md` for the goal, data strategy, training stages, and
   evaluation gates.
2. `dataset_registry.json` for candidate datasets and their role in the plan.
3. `EXPERIMENTS.md` for the first baseline run, follow-on dataset plan, and
   run-tracking expectations.
4. `APP_RECORDING_LABELING.md` for target-domain reviewer labels.
5. `manifest.py` for the normalized manifest contract.
6. `build_manifest.py` for currently supported manifest builders.
7. `train.py` for the current GT Singer training loop.
8. `feedback.py` and `server/technique/api.py` for inference-time output shape.

Audit dataset expansion decisions with:

```bash
cd server/technique
python3 -m gt_singer_grader.dataset_strategy --strict
```

## Working Without Datasets

Use the app-recording review fixture to test manifest plumbing:

```bash
cd server/technique
python3 -m gt_singer_grader.build_manifest app-recordings \
  --csv gt_singer_grader/tests/fixtures/app_recordings_review_smoke.csv \
  --output /tmp/nanopitch-technique/app_recordings_manifest.jsonl

python3 -m gt_singer_grader.manifest /tmp/nanopitch-technique/app_recordings_manifest.jsonl
```

For real app recordings, start from `app_recordings_review_template.csv` or run
`python3 -m gt_singer_grader.prepare_app_recordings` against collected WAVs to
create `gt_singer_grader/data/app_recordings/review_labels.csv`.

## Running The Existing Demo Later

The browser coach app does not require training data:

```bash
python3 -m http.server 8080
```

Open:

```text
http://127.0.0.1:8080/coach/web/
```

The technique API requires PyTorch:

```bash
make technique-api
```

Run it from the repo root in a second terminal while `python3 -m http.server
8080` serves the browser app. With the API running, the browser report contains
all four project axes: pitch, tempo, dynamics, and technique. If the API is
offline, the browser still reports pitch, tempo, and dynamics and marks the
technique axis unavailable.

## Training Later

After downloading GT Singer, train a first honest baseline with speaker-held-out
validation and user-recording augmentation:

```bash
cd server/technique
python3 -m pip install -r gt_singer_grader/requirements-training.txt
python3 -m gt_singer_grader.download_dataset \
  --output-dir ./gt_singer_grader/data/GTSinger \
  --language English
```

```bash
cd server/technique
python3 -m gt_singer_grader.preflight
```

The preflight report includes a `next_steps` list with required install and
download commands that still apply to the local machine. It also reports
app-label collection and release-readiness warnings, such as stale packaged
checkpoint metadata, under `optional_next_steps`.

To summarize the whole experiment sequence from the current artifact state:

```bash
cd server/technique
python3 -m gt_singer_grader.experiment_status
```

Use this after each train/evaluate/compare step to see the next command and the
remaining evidence gap.

```bash
cd server/technique
python3 -m gt_singer_grader.train \
  --dataset-root ./gt_singer_grader/data/GTSinger \
  --output-dir ./gt_singer_grader/runs/gtsinger_song_aug_v1 \
  --training-plan ./gt_singer_grader/runs/gtsinger_song_aug_v1/training_plan.json \
  --split-group song \
  --user-audio-augmentation \
  --epochs 50 \
  --batch-size 8 \
  --quiet
```

Do not treat random clip-level validation as product evidence. For user-uploaded
recordings, speaker-held-out and app-recording validation are required.

## Current Known Gaps

- Raw GT Singer data is not committed and not downloaded by this branch.
- PyTorch is optional for light dev and is not required for the first checks.
- Real app-style labeled recordings still need to be collected.
- The current checkpoint is demo-grade, trained on clean GT Singer English data.
- VocalSet has a conservative manifest builder, but raw VocalSet data is not
  present locally and the supplemental run should wait until the GT Singer
  baseline has metrics.
- CVT/SVQTD remain later candidates after license/taxonomy review.
