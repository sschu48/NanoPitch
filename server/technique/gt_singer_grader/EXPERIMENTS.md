# Technique Model Experiments

Use this file as the human-readable index for technique-model training runs.
Each run directory should also contain machine-readable artifacts written by
`train.py`.

## Run Artifacts

Every serious training run should keep:

- `run_config.json`: command arguments, model config, split strategy, split
  summaries, Git revision, Python version, platform metadata, training-plan
  hash, and manifest artifact hashes
- `metrics_history.jsonl`: one JSON object per epoch with train/validation
  metrics
- `best_metrics.json`: best validation checkpoint summary
- `train_manifest.jsonl` and `val_manifest.jsonl`: exact split records
- `checkpoints/best.pth`: checkpoint selected by validation score

Use the experiment status report as the day-to-day checklist:

```bash
cd server/technique
python3 -m gt_singer_grader.experiment_status
```

It reports which artifacts exist, whether the GT Singer baseline, full
VocalSet, balanced VocalSet, and app-adapted candidates have complete
train/eval evidence, whether app validation is ready, and the next command to
run.

Before launching a long training job, run the stdlib-only planner:

```bash
python3 -m gt_singer_grader.plan_training \
  --dataset-root ./gt_singer_grader/data/GTSinger \
  --split-group song \
  --output-json ./gt_singer_grader/runs/gtsinger_song_aug_v1/training_plan.json \
  --strict
```

It validates that the planned train and validation splits are non-empty, contain
at least one non-control technique family, and have enough family diversity to
make the run worth starting. It also requires matching trainable technique
families across train and validation so a baseline cannot pass while holding out
families the model never saw during training.

Verify run artifact hashes before evaluating a checkpoint:

```bash
python3 -m gt_singer_grader.verify_run \
  --run-config ./gt_singer_grader/runs/gtsinger_song_aug_v1/run_config.json \
  --strict
```

Every candidate checkpoint should also have an evaluation directory containing:

- `evaluation_config.json`: checkpoint, manifest, threshold settings, and
  environment metadata for this evaluation
- `metrics.json`: top-1, top-2, macro F1, control false-positive rate,
  non-technique false-positive rate, and prediction/status counts
- `predictions.csv`: one row per evaluated clip
- `confusion_matrix.csv`: gold family by predicted family
- `threshold_sweep.json`: confidence/technique threshold comparison for
  `none` and `unclear` behavior
- `operating_point.json`: selected confidence/technique thresholds using the
  configured control and non-technique false-positive-rate gates
- `calibration.json` and `calibration.csv`: confidence-bin accuracy,
  expected calibration error, and maximum calibration error

`evaluate.py` validates non-empty normalized or GT Singer split manifests before
checkpoint loading, so a candidate evaluation should fail fast on malformed
manifest records.
Run `python3 -m gt_singer_grader.verify_evaluation --eval-dir <eval_dir> --strict`
before comparing or packaging to recheck the checkpoint and manifest hashes
stored in `evaluation_config.json`. For evaluations of trained run directories,
pass `--run-config` so the evaluation evidence also rechecks the run-config
artifact hashes. `compare_runs` and `package_candidate` also enforce this
provenance check directly, so stale evaluation directories cannot be promoted
by accident.

For app-recording validation, keep `app_recordings_eval_audit.json` beside the
manifest. It records per-family coverage, negative-example coverage, and
split-group diversity before the model metrics are interpreted.

After app-recording coverage is ready, compare the app-adapted release
candidate in the app domain:

```bash
python3 -m gt_singer_grader.evaluate \
  --checkpoint ./gt_singer_grader/runs/gtsinger_song_aug_v1/checkpoints/best.pth \
  --manifest ./gt_singer_grader/manifests/app_recordings_eval.jsonl \
  --run-config ./gt_singer_grader/runs/gtsinger_song_aug_v1/run_config.json \
  --output-dir ./gt_singer_grader/runs/gtsinger_song_aug_v1/eval_app \
  --max-control-fpr 0.25 \
  --max-non-technique-fpr 0.25

python3 -m gt_singer_grader.compare_runs \
  --baseline ./gt_singer_grader/runs/gtsinger_song_aug_v1/eval_app \
  --candidate ./gt_singer_grader/runs/gtsinger_app_adapted_v1/eval_app \
  --output-json ./gt_singer_grader/runs/run_comparison_app_adapted.json
```

Do not compare app-adapted candidate metrics against GT Singer validation
metrics; baseline and candidate must use the same app evaluation manifest.

Compare evaluated runs with:

```bash
cd server/technique
python3 -m gt_singer_grader.compare_runs \
  --baseline ./gt_singer_grader/runs/gtsinger_song_aug_v1/eval_val \
  --candidate ./gt_singer_grader/runs/gtsinger_vocalset_song_v2/eval_val \
  --candidate ./gt_singer_grader/runs/gtsinger_vocalset_balanced120_song_v2/eval_val \
  --output-json ./gt_singer_grader/runs/run_comparison_public_v2.json
```

The comparison report includes each run's selected operating point when
`operating_point.json` is present, so threshold changes are reviewed alongside
top-2 accuracy, macro F1, false positives, and calibration. The comparison step
requires the full evaluator artifact set and valid evaluation provenance for
baseline and candidate directories, and records evaluation artifact hashes so
packaging can reject stale comparison reports. Candidates must pass both
absolute `gates` and baseline-relative
`regression_gates`. Each candidate also has a `promotion` block; only candidates
with `eligible: true` should be considered for packaging.

Default regression gates require no loss versus the baseline: top-2 accuracy
delta >= `0.0`, macro F1 delta >= `0.0`, control false-positive rate delta <=
`0.0`, non-technique false-positive rate delta <= `0.0`, and expected
calibration error delta <= `0.0`. Override them with `--min-top2-delta`,
`--min-macro-f1-delta`, `--max-control-fpr-delta`,
`--max-non-technique-fpr-delta`, and `--max-ece-delta` only when there is a
documented tradeoff. The absolute non-technique false-positive gate defaults to
`0.25`.

After the app-adapted candidate is eligible, package it with
`python3 -m gt_singer_grader.package_candidate`. The package metadata is part of
the release evidence and should point back to the comparison report and
candidate app-evaluation directory. Include `--app-validation-audit` for any
product-facing checkpoint; normal packaging requires it so the package records
the app-domain coverage gate. Product packaging refuses non-app-adapted
candidates even if public-dataset validation gates pass.
Packaging also checks that the candidate evaluation directory still contains
the required evaluator artifacts listed above, verifies evaluation provenance,
confirms the packaged checkpoint matches the checkpoint recorded by evaluation,
confirms the comparison report was generated from the current evaluation
artifacts, confirms the app-validation audit was generated from the current app
evaluation manifest, and fingerprints the checkpoint and evidence files with
SHA-256 hashes. Use
`python3 -m gt_singer_grader.verify_package --strict` to recheck an existing
package before release or rollback. Verification checks both evidence hashes
and semantic release gates such as promotion eligibility, app-validation
readiness, complete evaluator artifacts, and whether the packaged checkpoint
still matches the checkpoint recorded by the candidate evaluation.

## Baseline Run: `gtsinger_song_aug_v1`

Purpose:

- establish the first trainable GT Singer baseline on the currently available
  local data
- test whether the current conv/GRU architecture learns the target technique
  families before adding supplemental datasets
- measure whether user-recording augmentation hurts or helps validation

Command:

```bash
cd server/technique
python3 -m gt_singer_grader.preflight

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

Decision criteria:

- keep if speaker-held-out top-1/top-2 behavior is useful by family
- keep if frame technique macro F1 is meaningfully above chance
- reject or revise if control/ordinary singing produces frequent forced
  techniques
- treat speaker-held-out as the next generalization check once the local GT
  Singer download has enough family coverage per held-out singer
- do not package as a product checkpoint until app-recording validation exists

Result:

```text
status: complete
checkpoint: gt_singer_grader/runs/gtsinger_song_aug_v1/checkpoints/best.pth
notes: current promotion control; run/evaluation artifacts verify cleanly
```

## Full VocalSet Candidate: `gtsinger_vocalset_song_v2`

Purpose:

- test whether VocalSet improves singer/timbre/vowel coverage after the GT
  Singer baseline is measured

Required work before running:

- build `gt_singer_grader/manifests/vocalset.jsonl` from an extracted VocalSet
  tree
- review VocalSet labels and map only labels that cleanly match the active
  NanoPitch taxonomy
- keep the same GT Singer validation split strategy as the baseline

Command:

```bash
cd server/technique
python3 -m gt_singer_grader.train \
  --dataset-root ./gt_singer_grader/data/GTSinger \
  --output-dir ./gt_singer_grader/runs/gtsinger_vocalset_song_v2 \
  --training-plan ./gt_singer_grader/runs/gtsinger_vocalset_song_v2/training_plan.json \
  --split-group song \
  --user-audio-augmentation \
  --extra-train-manifest ./gt_singer_grader/manifests/vocalset.jsonl \
  --epochs 50 \
  --batch-size 8 \
  --quiet
```

Result:

```text
status: complete_not_promoted
checkpoint: gt_singer_grader/runs/gtsinger_vocalset_song_v2/checkpoints/best.pth
best_epoch: 39
metrics: top1=0.4840, top2=0.6410, clip_macro_f1=0.3384,
  technique_macro_f1_at_0_30=0.2031, control_fpr=0.1864,
  non_technique_fpr=0.1864, ece=0.2790
notes: run/evaluation artifacts verify cleanly. The full VocalSet expansion
  reduces false positives versus the GT Singer baseline, but it overfits and
  regresses top-1, top-2, macro F1, and calibration. Do not promote.
failed_gates: clip_macro_f1, clip_macro_f1_delta,
  expected_calibration_error, expected_calibration_error_delta,
  top2_accuracy_delta
```

## Balanced VocalSet Candidate: `gtsinger_vocalset_balanced120_song_v2`

Purpose:

- check whether capping VocalSet examples per mapped family improves robustness
  without letting the supplemental source dominate the GT Singer baseline
- reduce label/source imbalance before deciding whether more VocalSet work is
  worth pursuing

Result:

```text
status: complete_not_promoted
checkpoint: gt_singer_grader/runs/gtsinger_vocalset_balanced120_song_v2/checkpoints/best.pth
best_epoch: 44
metrics: top1=0.5106, top2=0.6755, clip_macro_f1=0.4317,
  technique_macro_f1_at_0_30=0.2308, control_fpr=0.2768,
  non_technique_fpr=0.2768, ece=0.1835
notes: this is the better public-data candidate and slightly improves
  thresholded technique macro F1 versus the baseline, but it regresses top-1,
  top-2, clip macro F1, and false positives. Do not promote; move effort to
  app-recording collection and app-domain adaptation.
failed_gates: clip_macro_f1_delta, control_false_positive_rate,
  control_false_positive_rate_delta, expected_calibration_error_delta,
  non_technique_false_positive_rate, non_technique_false_positive_rate_delta,
  top2_accuracy_delta
```

## App-Recording Validation

Purpose:

- validate the detector on the target product domain
- tune `none`, `unclear`, and `not_enough_voice` thresholds
- decide whether to fine-tune or only calibrate confidence
- track `non_technique_false_positive_rate` so app clips labeled `none` or
  `unclear` are not treated as successful forced technique detections

Minimum first-pass collection:

- 5-10 second clips
- singer ID or stable anonymized singer grouping
- intended family, if any
- reviewer labels per technique: absent / weak / present / strong
- enough neutral/control takes to measure false positives
- at least 20 held-out clips per target technique before claiming MVP behavior
- a passing `python3 -m gt_singer_grader.audit_app_validation --strict` report

Result:

```text
status: not_collected
checkpoint: n/a
notes: public datasets do not replace this validation set; this is the current
  required next step before an app-adapted product checkpoint can be promoted
```

## App-Adapted Candidate: `gtsinger_app_adapted_v1`

Purpose:

- fine-tune the GT Singer baseline with reviewed app-recording training clips
- evaluate baseline and candidate on the same held-out app evaluation manifest
- promote only if the app-adapted candidate passes absolute gates and does not
  regress against the app-domain baseline

Current status:

```text
status: waiting_for_app_recording_labels
train_manifest: gt_singer_grader/manifests/app_adapted_train.jsonl
eval_manifest: gt_singer_grader/manifests/app_recordings_eval.jsonl
comparison: gt_singer_grader/runs/run_comparison_app_adapted.json
notes: blocked on gt_singer_grader/data/app_recordings/review_labels.csv
```

Command once the app manifests and audit are ready:

```bash
cd server/technique
python3 -m gt_singer_grader.train \
  --train-manifest ./gt_singer_grader/manifests/app_adapted_train.jsonl \
  --val-manifest ./gt_singer_grader/manifests/app_recordings_val.jsonl \
  --output-dir ./gt_singer_grader/runs/gtsinger_app_adapted_v1 \
  --training-plan ./gt_singer_grader/runs/gtsinger_app_adapted_v1/training_plan.json \
  --user-audio-augmentation \
  --epochs 50 \
  --batch-size 8 \
  --quiet
```
