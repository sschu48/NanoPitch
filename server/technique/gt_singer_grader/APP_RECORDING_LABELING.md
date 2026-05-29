# App Recording Labeling Protocol

The public datasets are not enough to prove product behavior. NanoPitch needs a
small target-domain validation set made from app-style recordings.

## Clip Requirements

- 5-10 seconds of singing
- stable anonymized `singer_id`
- one recording per row
- consented storage and review
- keep train/validation/test splits singer-held-out

## Reviewer Labels

Use `app_recordings_review_template.csv`.

If recordings have already been collected into a folder, generate a starter
review CSV first:

```bash
cd server/technique
python3 -m gt_singer_grader.prepare_app_recordings \
  --audio-dir ./gt_singer_grader/data/app_recordings/raw \
  --output ./gt_singer_grader/data/app_recordings/review_labels.csv \
  --report-json ./gt_singer_grader/data/app_recordings/prepare_report.json \
  --relative-to . \
  --collection-plan ./gt_singer_grader/data/app_recordings/collection_plan.csv \
  --singer-id-from-parent
```

This command does not create truth labels. It writes one row per WAV and leaves
the technique strength columns blank for reviewer markup. If recordings are
organized as `raw/<singer_id>/<take>.wav`, `--singer-id-from-parent` also fills
`singer_id` and `split_group` so later train/validation splits stay
singer-held-out.
If a collection plan is provided, rows whose `audio_path` matches a planned
`suggested_filename` are prefilled with the planned `singer_id`,
`intended_family`, and collection notes. Reviewer strength columns still remain
blank until a human reviewer marks them.
The optional prepare report lists matched planned WAVs, unplanned WAVs, and
planned filenames that have not been collected yet.

Each technique column accepts:

- `absent`: technique is not present
- `weak`: possible trace, not strong enough for a positive label
- `present`: clear enough to count as present
- `strong`: obvious target technique

Technique columns:

- `mix`
- `falsetto`
- `breathy`
- `pharyngeal`
- `glissando`
- `vibrato`

`present` and `strong` become training-positive technique labels. `weak` and
`absent` are preserved in metadata for threshold analysis but do not become
positive training labels.
Every labeled row should include `reviewer_id`; the coverage check treats
missing reviewer provenance as a blocking warning.
For the current single-family clip trainer, clips with more than one positive
family are kept for evaluation/threshold analysis rather than fine-tuning.

If you fill the optional `families` or `techniques` override columns, they must
agree with the per-technique review columns. For example, a row with
`breathy=present` cannot list `techniques=vibrato`, and a `families=none` row
cannot also have any `present` or `strong` technique. The manifest builder
rejects contradictory rows.

If every technique is marked `absent`, the manifest builder labels the clip as
`control`. If only `weak` techniques are present, the builder labels the clip as
`unclear` for validation/threshold analysis.

Use `intended_family` when the singer was prompted to attempt a technique. This
is not treated as truth by itself; reviewer columns are the truth labels.

## Building The Manifest

From the repository root, the repeatable branch path is:

```bash
make technique-app-collection-plan
make technique-app-materialize-collection
make technique-app-export-collection-packet
make technique-app-check-collection
make technique-app-prepare-review
# Human reviewers fill server/technique/gt_singer_grader/data/app_recordings/review_labels.csv.
make technique-app-review-progress
make technique-app-manifests
make technique-app-adapted-plan
make technique-app-adapted-train
make technique-app-baseline-eval
make technique-app-adapted-eval
make technique-app-adapted-compare
make technique-app-adapted-package
```

Run `make technique-app-collection-plan` before collecting new clips. After the
plan exists, run `make technique-app-materialize-collection` to create the
recording folders, `collection_checklist.csv`, and the smaller
`collection_missing.csv` work queue. Then run
`make technique-app-export-collection-packet` to create one Markdown recording
sheet per planned singer group under `data/app_recordings/collection_packet`.
After the WAVs exist under `gt_singer_grader/data/app_recordings/raw`, run
`make technique-app-check-collection` to refresh the checklist and fail if any
planned WAV is still missing, unreadable as WAV audio, or outside the 5-10
second collection window. Then run `make technique-app-prepare-review` to
generate the review CSV skeleton. While reviewers are filling labels, run
`make technique-app-review-progress` to write
`data/app_recordings/label_coverage_report.json` without requiring release
coverage yet. The manifest target intentionally fails until the reviewed CSV has
enough labeled coverage, every referenced WAV exists, duplicates are removed,
and the held-out app validation audit passes. The package target intentionally
fails unless the app-adapted candidate passes the app-domain comparison gates
against the baseline evaluated on the same audited app validation manifest.

The commands below are the equivalent manual steps.

```bash
cd server/technique
python3 -m gt_singer_grader.build_manifest app-recordings \
  --csv ./gt_singer_grader/data/app_recordings/review_labels.csv \
  --output ./gt_singer_grader/manifests/app_recordings.jsonl

python3 -m gt_singer_grader.manifest ./gt_singer_grader/manifests/app_recordings.jsonl
```

Then separate rows that can be used for supervised fine-tuning from rows that
must stay evaluation-only:

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

Create singer-held-out train/validation manifests from the trainable subset:

```bash
python3 -m gt_singer_grader.split_manifest \
  --input ./gt_singer_grader/manifests/app_recordings_trainable.jsonl \
  --train-output ./gt_singer_grader/manifests/app_recordings_train.jsonl \
  --val-output ./gt_singer_grader/manifests/app_recordings_val.jsonl \
  --summary-output ./gt_singer_grader/manifests/app_recordings_split_summary.json \
  --val-ratio 0.2 \
  --strict-non-empty \
  --strict-family-coverage
```

The splitter keeps every `split_group` on one side of the split. For app
recordings, `split_group` should be the anonymized singer ID unless there is a
stronger grouping requirement.
Before fine-tuning, both train and validation splits need at least two clip
families and at least one non-control technique family. Every trainable
technique family also needs examples on both sides. If the strict split check
fails, collect more singers or adjust the split before training.

Merge the held-out trainable validation rows with evaluation-only rows before
running app-domain evaluation and coverage audit:

```bash
python3 -m gt_singer_grader.merge_manifest \
  --input ./gt_singer_grader/manifests/app_recordings_val.jsonl \
  --input ./gt_singer_grader/manifests/app_recordings_eval_only.jsonl \
  --output ./gt_singer_grader/manifests/app_recordings_eval.jsonl \
  --summary-output ./gt_singer_grader/manifests/app_recordings_eval_summary.json
```

Audit the held-out app validation manifest before using it for model promotion:

```bash
python3 -m gt_singer_grader.audit_app_validation \
  --manifest ./gt_singer_grader/manifests/app_recordings_eval.jsonl \
  --output-json ./gt_singer_grader/manifests/app_recordings_eval_audit.json \
  --strict
```

The default audit requires at least 20 held-out clips per non-control technique
family, at least 20 negative `control` / `none` / `unclear` clips, and at least
3 singer groups. The audit output includes the SHA-256 hash of the app
evaluation manifest; rerun the audit after any manifest change before packaging
or promotion review.

To inspect coverage directly from a partially reviewed CSV before building the
full manifest pipeline:

```bash
python3 -m gt_singer_grader.app_label_coverage \
  --csv ./gt_singer_grader/data/app_recordings/review_labels.csv \
  --audio-root . \
  --require-audio-files
```

This reports missing target-family counts, negative-example shortfall,
split-group shortfall, missing audio files, reviewer provenance gaps, and
unreviewed/unlabeled rows. The `review_progress` section also breaks review
completion down by intended family and split group, which is useful while the
140-clip collection packet is being reviewed in batches.

To turn those shortfalls into a concrete next collection list:

```bash
python3 -m gt_singer_grader.plan_app_collection \
  --csv ./gt_singer_grader/data/app_recordings/review_labels.csv \
  --output-json ./gt_singer_grader/data/app_recordings/collection_plan.json \
  --output-csv ./gt_singer_grader/data/app_recordings/collection_plan.csv \
  --clips-per-singer 7
```

The planner uses the same default coverage targets as app validation: 20
held-out clips per non-control technique family, 20 negative `control` /
`none` / `unclear` clips, and at least 3 singer groups. If the review CSV does
not exist yet, it plans from zero coverage. By default, it groups up to 7
planned takes under the same synthetic `singer_id`, so the first full collection
target is about 20 singer groups with one pass through the six technique
families plus one control take per group.

Materialize the plan into the expected `raw/<singer_id>/` folders and a
recording checklist:

```bash
python3 -m gt_singer_grader.materialize_app_collection \
  --plan ./gt_singer_grader/data/app_recordings/collection_plan.csv \
  --root ./gt_singer_grader/data/app_recordings \
  --checklist ./gt_singer_grader/data/app_recordings/collection_checklist.csv \
  --missing-csv ./gt_singer_grader/data/app_recordings/collection_missing.csv \
  --report-json ./gt_singer_grader/data/app_recordings/collection_materialize_report.json \
  --strict
```

This creates directories, full checklist rows, and a missing-recordings CSV. It
does not create WAV files or labels; collected recordings should be saved at the
checklist `expected_audio_path` values before running `prepare_app_recordings`.

Export the checklist into per-singer recording sheets for collection handoff:

```bash
python3 -m gt_singer_grader.export_app_collection_packet \
  --checklist ./gt_singer_grader/data/app_recordings/collection_checklist.csv \
  --output-dir ./gt_singer_grader/data/app_recordings/collection_packet \
  --summary-json ./gt_singer_grader/data/app_recordings/collection_packet_summary.json \
  --strict
```

## First Validation Target

Before claiming an MVP detector:

- at least 20 held-out clips per target technique
- neutral/control clips from multiple singers
- short/quiet failure cases
- singer-held-out split
- evaluator run with `metrics.json`, `confusion_matrix.csv`,
  `threshold_sweep.json`, and `calibration.json`
