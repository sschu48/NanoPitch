# Technique Model Development Plan

## Goal

Build a model that accepts a user singing recording and detects which vocal
technique or techniques are present. The first production target is detection,
not quality grading.

The model should return:

- likely technique labels with confidence
- `none` / `unclear` when no technique is strong enough
- a coarse timeline for longer recordings
- enough metadata to explain whether the prediction is reliable

## Planned Architecture

Keep the first production candidate simple:

```text
raw WAV
  -> mono 16 kHz
  -> log-mel spectrogram
  -> small causal convolution stack
  -> GRU temporal encoder
  -> voiced-frame-aware pooling
  -> three heads:
       clip family classifier
       frame-level technique classifier
       frame-level VAD classifier
```

The current `TechniqueGraderModel` already follows this shape. Do not replace
it with a large pretrained encoder until the simple model has a clear failure
mode on speaker-held-out GT Singer and app-recording validation.

The deployment contract is detection-first:

- the clip head provides a stable family summary
- the frame technique head provides multi-label scores and timeline windows
- the VAD head keeps silence/noise from driving the technique result
- postprocessing may return `none`, `unclear`, or `not_enough_voice`

## Label Taxonomy

Primary clip families:

- `control`
- `breathy`
- `glissando`
- `mixed_voice`
- `falsetto`
- `pharyngeal`
- `vibrato`
- `none`
- `unclear`

Frame or window-level technique tags:

- `mix`
- `falsetto`
- `breathy`
- `pharyngeal`
- `glissando`
- `vibrato`

The model should be multi-label internally. The UI can still show a dominant
technique, but the model should not be forced to pick exactly one technique for
the whole clip.

## Data Strategy

Use GT Singer as the supervised starting point, but do not treat it as final
product validation. GT Singer teaches the model what clean technique examples
sound like; app-style recordings teach it how real users sound.

Priority order:

1. **GT Singer**: primary labeled technique source.
2. **VocalSet**: next supplemental supervised source after the GT Singer
   baseline, using only labels that map cleanly to the NanoPitch taxonomy.
3. **NanoPitch app recordings**: required target-domain source. Collect short
   consented clips with intended technique and reviewer labels.
4. **DAMP / mobile karaoke datasets**: useful for domain adaptation and
   robustness testing, but not a direct supervised technique source unless we
   label a subset ourselves.
5. **CVT vocal mode dataset / SVQTD**: optional later sources. Use only after
   license and taxonomy review because their labels do not map one-to-one to
   the first NanoPitch technique families.
6. **CCMusic Acapella**: useful later for quality/vocal-skill axes, not the
   core technique detector.

The machine-readable dataset registry lives in `dataset_registry.json`. Audit it
with `python3 -m gt_singer_grader.dataset_strategy --strict` before treating a
new public source as trainable; mobile and quality datasets without technique
labels must stay out of supervised technique training until labels, license, and
taxonomy mapping are reviewed.

## Acquisition Workflow

From `NanoPitch/server/technique`:

```bash
python3 -m gt_singer_grader.download_dataset \
  --output-dir ./gt_singer_grader/data/GTSinger \
  --language English
```

Then build a normalized manifest:

```bash
python3 -m gt_singer_grader.build_manifest gtsinger \
  --root ./gt_singer_grader/data/GTSinger \
  --language English \
  --output ./gt_singer_grader/manifests/gtsinger_english.jsonl
```

For future app recordings, prepare a CSV with:

```text
audio_path,recording_id,singer_id,song_id,families,techniques,split_group,label_source,notes
```

For reviewed app recordings, prefer `app_recordings_review_template.csv` and
the protocol in `APP_RECORDING_LABELING.md`. The reviewer strength columns use
`absent`, `weak`, `present`, or `strong`; `present` and `strong` are converted
to positive technique labels.

If app WAVs are already collected, generate the review CSV starter before
labeling:

```bash
python3 -m gt_singer_grader.prepare_app_recordings \
  --audio-dir ./gt_singer_grader/data/app_recordings/raw \
  --output ./gt_singer_grader/data/app_recordings/review_labels.csv \
  --report-json ./gt_singer_grader/data/app_recordings/prepare_report.json \
  --relative-to . \
  --collection-plan ./gt_singer_grader/data/app_recordings/collection_plan.csv \
  --singer-id-from-parent
```

This command fails when no WAV files are found, so an empty review CSV does not
hide a missing collection step. Use `--allow-empty` only for local plumbing
checks.

Then:

```bash
python3 -m gt_singer_grader.build_manifest app-recordings \
  --csv ./gt_singer_grader/data/app_recordings/review_labels.csv \
  --output ./gt_singer_grader/manifests/app_recordings.jsonl
```

During review, check coverage directly from the CSV:

```bash
python3 -m gt_singer_grader.plan_app_collection \
  --csv ./gt_singer_grader/data/app_recordings/review_labels.csv \
  --output-json ./gt_singer_grader/data/app_recordings/collection_plan.json \
  --output-csv ./gt_singer_grader/data/app_recordings/collection_plan.csv \
  --clips-per-singer 7

python3 -m gt_singer_grader.materialize_app_collection \
  --plan ./gt_singer_grader/data/app_recordings/collection_plan.csv \
  --root ./gt_singer_grader/data/app_recordings \
  --checklist ./gt_singer_grader/data/app_recordings/collection_checklist.csv \
  --report-json ./gt_singer_grader/data/app_recordings/collection_materialize_report.json \
  --strict

python3 -m gt_singer_grader.app_label_coverage \
  --csv ./gt_singer_grader/data/app_recordings/review_labels.csv \
  --audio-root . \
  --require-audio-files
```

Before fine-tuning on app recordings, separate trainable labels from
threshold-analysis labels:

```bash
python3 -m gt_singer_grader.filter_manifest \
  --input ./gt_singer_grader/manifests/app_recordings.jsonl \
  --trainable-output ./gt_singer_grader/manifests/app_recordings_trainable.jsonl \
  --eval-only-output ./gt_singer_grader/manifests/app_recordings_eval_only.jsonl \
  --summary-output ./gt_singer_grader/manifests/app_recordings_filter_summary.json

python3 -m gt_singer_grader.manifest \
  ./gt_singer_grader/manifests/app_recordings_trainable.jsonl \
  --require-trainable
```

Use `app_recordings_trainable.jsonl` for fine-tuning splits. Keep
`app_recordings_eval_only.jsonl` for threshold sweeps and false-positive
analysis. The current trainer accepts only one trainable clip family per row;
multi-family app clips stay eval-only until the clip family head/loss is
multi-label.

For VocalSet, manually download and extract VocalSet 1.2, then build a
conservative manifest from path-derived technique labels:

```bash
python3 -m gt_singer_grader.build_manifest vocalset \
  --root ./gt_singer_grader/data/VocalSet \
  --output ./gt_singer_grader/manifests/vocalset.jsonl
```

Validate any manifest with:

```bash
python3 -m gt_singer_grader.manifest ./gt_singer_grader/manifests/gtsinger_english.jsonl
```

The validator prints dataset, family, and trainability summaries after schema
validation so unusable training rows are visible before `train.py`.

Split normalized manifests with grouped validation:

```bash
python3 -m gt_singer_grader.split_manifest \
  --input ./gt_singer_grader/manifests/app_recordings_trainable.jsonl \
  --train-output ./gt_singer_grader/manifests/app_recordings_train.jsonl \
  --val-output ./gt_singer_grader/manifests/app_recordings_val.jsonl \
  --summary-output ./gt_singer_grader/manifests/app_recordings_split_summary.json \
  --strict-non-empty \
  --strict-family-coverage
```

The strict family-coverage gate fails if train or validation has too few
families, no non-control technique family, or a trainable technique family that
appears on only one side of the split.

Then merge held-out trainable rows with evaluation-only rows for app-domain
validation:

```bash
python3 -m gt_singer_grader.merge_manifest \
  --input ./gt_singer_grader/manifests/app_recordings_val.jsonl \
  --input ./gt_singer_grader/manifests/app_recordings_eval_only.jsonl \
  --output ./gt_singer_grader/manifests/app_recordings_eval.jsonl \
  --summary-output ./gt_singer_grader/manifests/app_recordings_eval_summary.json
```

Build the release candidate as an app-adapted training manifest by combining
the GT Singer baseline train split with the app train split:

```bash
python3 -m gt_singer_grader.merge_manifest \
  --input ./gt_singer_grader/runs/gtsinger_song_aug_v1/train_manifest.jsonl \
  --input ./gt_singer_grader/manifests/app_recordings_train.jsonl \
  --output ./gt_singer_grader/manifests/app_adapted_train.jsonl \
  --summary-output ./gt_singer_grader/manifests/app_adapted_train_summary.json
```

Evaluate the GT Singer baseline on the same app evaluation manifest before
judging the app-adapted candidate. That keeps the comparison in the target
recording domain instead of comparing app metrics against GT Singer validation
metrics.

## Training Plan

Stage 1: make the existing model honest.

- Train the first local GT Singer baseline with `--split-group song` so every
  trainable technique family is represented in both train and validation. Move
  to `--split-group speaker` once the local GT Singer coverage supports a
  speaker-held-out split with matching family coverage.
- Require non-empty train and validation splits; an empty grouped validation
  split is a setup problem, not a usable experiment.
- Require primary train and validation splits to contain at least two clip
  families and at least one non-control technique family. A single-class split
  is not useful evidence for technique detection.
- Keep clip-family prediction, VAD, and frame technique heads.
- Use realistic augmentation before any production claim. The current
  `--user-audio-augmentation` switch covers:
  - room noise
  - room reverb
  - phone/laptop microphone EQ
  - clipping / over-hot input
  - gain variation
- Record all runs in `EXPERIMENTS.md` and keep the generated run artifacts:
  `run_config.json`, `metrics_history.jsonl`, `best_metrics.json`,
  `train_manifest.jsonl`, and `val_manifest.jsonl`.
- Keep the `run_config.json` environment block with Git revision, dirty flag,
  Python version, and platform metadata for every serious run.
- Keep the `run_config.json` artifact block with SHA-256 hashes for the copied
  train, validation, source, and supplemental manifests.
- Use `python3 -m gt_singer_grader.verify_run --strict` before evaluation to
  confirm those run artifacts still match `run_config.json`.

Stage 2: make the output match the product.

- Replace forced softmax-only behavior with multi-label technique reporting.
- Add thresholds for `none` and `unclear`.
- Evaluate top-1, top-2, macro F1, false-positive rate, and calibration.
- For app-domain evaluation, track `non_technique_false_positive_rate` across
  gold `control`, `none`, and `unclear` rows in addition to the stricter
  `control_false_positive_rate`.
- Use `python3 -m gt_singer_grader.evaluate` on every candidate checkpoint and
  keep `metrics.json`, `predictions.csv`, `confusion_matrix.csv`, and
  `threshold_sweep.json` beside the run artifacts.
- Keep `operating_point.json` from the evaluator as the first-pass threshold
  recommendation. It should satisfy both control and non-technique
  false-positive gates, then the full sweep should still be inspected before
  packaging a checkpoint.
- Use `calibration.json` and `calibration.csv` to compare confidence
  reliability across candidate checkpoints.
- Use `python3 -m gt_singer_grader.compare_runs` to compare baseline and
  candidate evaluation directories against the same gates. Treat
  `promotion.eligible` as the first pass/fail signal, then inspect absolute
  `gates`, baseline-relative `regression_gates`, and predictions before
  packaging any checkpoint.
- Require comparison gates for both `control_false_positive_rate` and
  `non_technique_false_positive_rate` so app-domain `none` and `unclear`
  regressions are visible before packaging.

Stage 3: add the first supplemental supervised dataset.

- Use the VocalSet manifest builder only after the GT Singer baseline is
  measured.
- Map only trusted VocalSet labels to the active NanoPitch technique taxonomy.
- Train with `--extra-train-manifest ./gt_singer_grader/manifests/vocalset.jsonl`
  while keeping the validation set GT Singer speaker-held-out or app-recording
  held-out. VocalSet records without alignment JSON use weak full-clip
  technique labels.
- If full-strength VocalSet mixing regresses GT Singer validation, build a
  capped/balanced supplemental manifest before retraining:

```bash
python3 -m gt_singer_grader.sample_manifest \
  --input ./gt_singer_grader/manifests/vocalset.jsonl \
  --output ./gt_singer_grader/manifests/vocalset_balanced_120.jsonl \
  --summary-output ./gt_singer_grader/manifests/vocalset_balanced_120_summary.json \
  --max-per-family 120
```

- Compare GT Singer-only against GT Singer + VocalSet using the same validation
  gates. Do not accept the larger dataset if app-style false positives get
  worse.

Stage 4: adapt to real users.

- Collect app recordings in 5-10 second clips.
- Label each as absent / weak / present / strong per technique.
- Use `present` / `strong` rows for supervised fine-tuning. Keep weak-only
  `unclear` rows and intended `none` rows for evaluation and threshold tuning
  until the model has explicit training targets for those states. Keep clips
  with multiple positive technique families in evaluation until the clip-family
  training objective supports multi-label targets. Evaluate those rows as gold
  `multiple` so they remain useful for inspection without inflating a
  single-family metric.
- Keep a singer-held-out test set that is never used for tuning.
- Fine-tune or calibrate on app recordings.
- Train an app-adapted candidate from `app_adapted_train.jsonl`, evaluate both
  the baseline and app-adapted candidate on `app_recordings_eval.jsonl`, then
  compare those app-domain evaluations before packaging. Public-dataset
  validation can justify a recipe, but app-domain comparison is the release
  gate.

Stage 5: consider a larger encoder.

If the current conv/GRU model plateaus, train a second prototype with a
pretrained audio/music encoder such as MERT, BEATs, or HTS-AT, then distill it
back to a smaller model if local/browser deployment matters.

## Evaluation Gates

Before replacing the packaged demo checkpoint, require:

- GT Singer or public-dataset validation metrics for the training recipe
- app-recording validation metrics for the app-adapted release candidate
- confusion matrix by technique
- false positive rate on `none` / ordinary singing clips
- confidence calibration check
- at least 20 labeled held-out clips per target technique, preferably more

Initial acceptance bar for a usable MVP:

- no forced technique when confidence is weak
- top-2 technique accuracy is useful on held-out singers
- clear failure state for short/quiet recordings
- app-recording validation does not collapse relative to GT Singer validation

## Current Gaps

- GT Singer raw data and Python training dependencies are local development
  prerequisites. They are not committed to the branch.
- We do not yet have labeled app-user recordings.
- VocalSet raw data is a local development prerequisite. The branch can build a
  VocalSet manifest and train it as weak supplemental data after the GT Singer
  baseline has metrics.
- The evaluation script writes binned calibration summaries, but it does not yet
  render chart images.
- The current packaged checkpoint is demo-grade and trained on clean GT Singer
  English recordings only.
