# GT Singer Grader

This module is a separate singing-grader pipeline that sits beside NanoPitch. It does not modify NanoPitch. Instead, it borrows a few ideas that already work well there:

- the same 10 ms frame cadence for audio features
- a dedicated VAD head so we grade voiced singing instead of silence
- the same onset penalty (`0.75`) for smoothing voiced/unvoiced decisions during feedback

For the model-building roadmap, dataset choices, manifest schema, and evaluation
gates, see [`MODEL_DEVELOPMENT.md`](MODEL_DEVELOPMENT.md).

For no-data/no-model onboarding and light checks, see
[`DEVELOPER_SETUP.md`](DEVELOPER_SETUP.md).

For run tracking and experiment decisions, see [`EXPERIMENTS.md`](EXPERIMENTS.md).

For target-domain app recording review labels, see
[`APP_RECORDING_LABELING.md`](APP_RECORDING_LABELING.md).

To audit the dataset expansion plan from the machine-readable registry:

```bash
cd NanoPitch/server/technique
python3 -m gt_singer_grader.dataset_strategy --strict
```

## What it trains

The first version is a technique detector, not a generic "good singer / bad
singer" model. It learns from GT Singer control-vs-emphasis pairs and predicts:

- clip-level family: `control`, `breathy`, `glissando`, `mixed_voice`, `falsetto`, `pharyngeal`, `vibrato`
- frame-level VAD
- frame-level technique activity for `mix`, `falsetto`, `breathy`, `pharyngeal`, `glissando`, `vibrato`

The current architecture is intentionally small: log-mel features, causal
convolutions, a GRU temporal encoder, voiced-frame-aware pooling, and separate
clip, frame-technique, and VAD heads. Keep this baseline until its failure mode
is clear on speaker-held-out GT Singer and app-recording validation.

That gives us a solid base for later quality grading. Once you add your own
good/bad labels, we can hang a separate quality head off the same clip embedding
without rewriting the whole pipeline.

## Expected dataset layout

The scanner matches the English tree shown on Hugging Face:

```text
English/
  EN-Alto-1/
    Breathy/
      all is found/
        Breathy_Group/
        Control_Group/
        Paired_Speech_Group/
    Mixed_Voice_and_Falsetto/
      all is found/
        Mixed_Voice_Group/
        Falsetto_Group/
        Control_Group/
```

Each `.wav` is paired with the GT Singer `.json` alignment file in the same folder. The training code uses those JSON technique flags to build frame labels.

## Download

```bash
cd NanoPitch/server/technique
python3 -m pip install -r gt_singer_grader/requirements-training.txt
python3 -m gt_singer_grader.download_dataset --output-dir ./gt_singer_grader/data/GTSinger
```

## Train

Before a real training run, check local dependencies and dataset paths:

```bash
cd NanoPitch/server/technique
python3 -m gt_singer_grader.preflight
```

If anything required is missing, the JSON report includes `next_steps` with the
exact install or download command to run. The same report also includes optional
follow-up work under `optional_next_steps`, including app-recording labels and
whether packaged checkpoint metadata is verifiable after promotion.

To see the current experiment stage and the next concrete command from the run
artifacts already on disk:

```bash
python3 -m gt_singer_grader.experiment_status
```

This is the lightweight control loop for the branch: it checks preflight,
baseline plan/run artifacts, evaluation artifacts, VocalSet candidate
plan/artifacts, the comparison report, and app-validation audit readiness. The
`current_stage` field gives the active stage name and blocking detail for CI or
handoff notes.

After dependencies and GT Singer are present, but before launching a long run,
plan the split without importing PyTorch:

```bash
python3 -m gt_singer_grader.plan_training \
  --dataset-root ./gt_singer_grader/data/GTSinger \
  --split-group song \
  --output-json ./gt_singer_grader/runs/gtsinger_song_aug_v1/training_plan.json \
  --strict
```

This reports train/validation family counts, speaker counts, split-group counts,
and any trainability errors. The planner also rejects mismatched family coverage:
every trainable technique family in validation must have training examples, and
every trainable technique family in training must have validation examples.

```bash
cd NanoPitch/server/technique
python3 -m gt_singer_grader.train \
  --dataset-root ./gt_singer_grader/data/GTSinger \
  --output-dir ./gt_singer_grader/runs/gtsinger_song_aug_v1 \
  --training-plan ./gt_singer_grader/runs/gtsinger_song_aug_v1/training_plan.json \
  --split-group song \
  --user-audio-augmentation \
  --epochs 20 \
  --batch-size 8 \
  --quiet
```

Useful outputs:

- `train_manifest.jsonl`
- `val_manifest.jsonl`
- `run_config.json` with command, split, Git, Python, platform, training-plan
  hash, and manifest artifact metadata
- `metrics_history.jsonl`
- `best_metrics.json`
- `checkpoints/best.pth`
- TensorBoard logs in `tb/`

Before evaluating or comparing a run, verify the manifest artifacts captured in
`run_config.json`:

```bash
python3 -m gt_singer_grader.verify_run \
  --run-config ./gt_singer_grader/runs/gtsinger_song_aug_v1/run_config.json \
  --strict
```

You can also train from prebuilt normalized manifests:

```bash
python3 -m gt_singer_grader.train \
  --train-manifest ./gt_singer_grader/manifests/gtsinger_train.jsonl \
  --val-manifest ./gt_singer_grader/manifests/gtsinger_val.jsonl \
  --output-dir ./gt_singer_grader/runs/manifest_baseline \
  --training-plan ./gt_singer_grader/runs/manifest_baseline/training_plan.json \
  --user-audio-augmentation
```

Both training and validation manifests must be non-empty. If grouped splitting
produces an empty validation set, collect more labeled groups or lower the
grouping granularity before training.
The primary train and validation splits must also contain at least two labeled
clip families, including at least one non-control technique family, so a run
cannot start from a degenerate single-class split.
The trainable technique-family sets must match across train and validation;
otherwise the model would either be evaluated on a family it never learned or
learn a family with no held-out measurement.

After the GT Singer baseline is measured, add VocalSet as weak supplemental
training data without changing the validation set:

```bash
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

## Evaluate

After training, run the evaluator against the saved validation manifest:

```bash
cd NanoPitch/server/technique
python3 -m gt_singer_grader.evaluate \
  --checkpoint ./gt_singer_grader/runs/gtsinger_song_aug_v1/checkpoints/best.pth \
  --manifest ./gt_singer_grader/runs/gtsinger_song_aug_v1/val_manifest.jsonl \
  --run-config ./gt_singer_grader/runs/gtsinger_song_aug_v1/run_config.json \
  --output-dir ./gt_singer_grader/runs/gtsinger_song_aug_v1/eval_val \
  --max-control-fpr 0.25 \
  --max-non-technique-fpr 0.25
```

The evaluator validates the manifest and rejects empty evaluation sets before
loading the checkpoint. It supports both normalized JSONL manifests and legacy
GT Singer split manifests written by `train.py`. Pass `--run-config` for
training-run evaluations so the evaluation evidence also fingerprints and
verifies the training manifest artifacts recorded by `run_config.json`.

Useful outputs:

- `evaluation_config.json`
- `metrics.json`
- `predictions.csv`
- `confusion_matrix.csv`
- `threshold_sweep.json`
- `operating_point.json`
- `calibration.json`
- `calibration.csv`

Verify the evaluation directory before comparing or packaging:

```bash
python3 -m gt_singer_grader.verify_evaluation \
  --eval-dir ./gt_singer_grader/runs/gtsinger_song_aug_v1/eval_val \
  --strict
```

For app-recording validation, inspect both `control_false_positive_rate` and
`non_technique_false_positive_rate`. The second metric treats gold `control`,
`none`, and `unclear` rows as negative examples, so it catches forced technique
detections on ordinary or ambiguous app clips.
The selected `operating_point.json` must satisfy both false-positive gates when
those negative rows are present.

Compare two evaluated runs with:

```bash
python3 -m gt_singer_grader.compare_runs \
  --baseline ./gt_singer_grader/runs/gtsinger_song_aug_v1/eval_val \
  --candidate ./gt_singer_grader/runs/gtsinger_vocalset_song_v2/eval_val \
  --candidate ./gt_singer_grader/runs/gtsinger_vocalset_balanced120_song_v2/eval_val \
  --output-json ./gt_singer_grader/runs/run_comparison_public_v2.json
```

Comparison requires the complete evaluator artifact set in both baseline and
candidate directories, verifies `evaluation_config.json` checkpoint/manifest
provenance before it computes promotion gates, and records SHA-256 hashes for
the evaluation artifacts it compared.

If evaluation wrote `operating_point.json`, the comparison report includes the
selected confidence and technique thresholds for each run.

By default, candidates must pass absolute gates and avoid baseline regressions:
top-2 accuracy delta >= `0.0`, macro F1 delta >= `0.0`, control false-positive
rate delta <= `0.0`, non-technique false-positive rate delta <= `0.0`, and
expected calibration error delta <= `0.0`. Relax or tighten those with
`--min-top2-delta`, `--min-macro-f1-delta`, `--max-control-fpr-delta`,
`--max-non-technique-fpr-delta`, and `--max-ece-delta`. The absolute
non-technique false-positive gate defaults to `0.25` and can be changed with
`--max-non-technique-fpr`.

Package only the promoted app-adapted candidate:

```bash
python3 -m gt_singer_grader.package_candidate \
  --checkpoint ./gt_singer_grader/runs/gtsinger_app_adapted_v1/checkpoints/best.pth \
  --comparison ./gt_singer_grader/runs/run_comparison_app_adapted.json \
  --candidate-eval-dir ./gt_singer_grader/runs/gtsinger_app_adapted_v1/eval_app \
  --app-validation-audit ./gt_singer_grader/manifests/app_recordings_eval_audit.json \
  --output-checkpoint ./gt_singer_grader/models/technique_demo_best.pth \
  --metadata ./gt_singer_grader/models/technique_demo_metadata.json
```

The package step refuses candidates whose comparison entry does not have
`promotion.eligible: true`, candidates that are not app-adapted app-domain
evaluations, whose app-validation audit is missing or not ready, whose
evaluation directory is missing required evaluator artifacts, or whose
evaluation provenance no longer matches the checkpoint/manifest used by
`evaluate.py`. It also verifies that `--checkpoint` has the same byte count and
SHA-256 hash as the checkpoint recorded in the candidate evaluation config, and
that the comparison report was generated from the current candidate evaluation
artifacts. The app-validation audit also records the app evaluation manifest
hash, so rerun `audit_app_validation` after changing
`app_recordings_eval.jsonl`. It then writes metadata with metrics, operating point, gate
thresholds, baseline deltas, app-validation coverage, the evaluation artifact
checklist, and SHA-256 hashes for the checkpoint and evidence files.

Verify an existing packaged checkpoint against its metadata with:

```bash
python3 -m gt_singer_grader.verify_package \
  --metadata ./gt_singer_grader/models/technique_demo_metadata.json \
  --strict
```

Verification checks both file hashes and package semantics: promotion eligibility,
app-validation readiness, complete evaluator artifact evidence, and whether the
current packaged checkpoint still matches the checkpoint recorded by evaluation.

## VocalSet Manifest

After the GT Singer baseline has metrics, VocalSet can be added as the first
supplemental supervised source. Download and extract VocalSet 1.2 manually, then
build a conservative path-label manifest:

```bash
cd NanoPitch/server/technique
python3 -m gt_singer_grader.build_manifest vocalset \
  --root ./gt_singer_grader/data/VocalSet \
  --output ./gt_singer_grader/manifests/vocalset.jsonl
```

The builder maps only known VocalSet technique tokens into the current
NanoPitch taxonomy and skips unrecognized styles.

Use the manifest validator as a quick audit before training or evaluation:

```bash
python3 -m gt_singer_grader.manifest ./gt_singer_grader/manifests/app_recordings.jsonl
```

On success it prints dataset counts, family counts, and trainability reason
counts.

## App Recording Labels

Use `app_recordings_review_template.csv` when collecting app-style clips. The
reviewer columns accept `absent`, `weak`, `present`, or `strong`; `present` and
`strong` become training-positive technique labels.
Fill `reviewer_id` for every labeled row so provenance survives later manifest
audits.
Rows with only `weak` labels become `unclear`, and fully absent rows become
`control`. Keep `unclear` / `none` rows in evaluation manifests for threshold
analysis; `train.py` rejects them as supervised training classes for now. Rows
with multiple positive families are also held out of fine-tuning until the clip
family loss is changed from single-class to multi-label. During evaluation,
those rows are counted as gold `multiple` rather than credited to the first
listed family.
If the optional `families` or `techniques` override columns are used, they must
match the per-technique absent/weak/present/strong columns; contradictory rows
are rejected during manifest build.

If WAVs have already been collected, generate the starter review CSV before
labeling. The command fails when no WAVs are found unless `--allow-empty` is
passed explicitly.

```bash
cd NanoPitch/server/technique
python3 -m gt_singer_grader.prepare_app_recordings \
  --audio-dir ./gt_singer_grader/data/app_recordings/raw \
  --output ./gt_singer_grader/data/app_recordings/review_labels.csv \
  --report-json ./gt_singer_grader/data/app_recordings/prepare_report.json \
  --relative-to . \
  --collection-plan ./gt_singer_grader/data/app_recordings/collection_plan.csv \
  --singer-id-from-parent
```

Build and split a reviewed app-recording manifest with singer-held-out groups:

```bash
cd NanoPitch/server/technique
python3 -m gt_singer_grader.plan_app_collection \
  --csv ./gt_singer_grader/data/app_recordings/review_labels.csv \
  --output-json ./gt_singer_grader/data/app_recordings/collection_plan.json \
  --output-csv ./gt_singer_grader/data/app_recordings/collection_plan.csv \
  --clips-per-singer 7

python3 -m gt_singer_grader.materialize_app_collection \
  --plan ./gt_singer_grader/data/app_recordings/collection_plan.csv \
  --root ./gt_singer_grader/data/app_recordings \
  --checklist ./gt_singer_grader/data/app_recordings/collection_checklist.csv \
  --missing-csv ./gt_singer_grader/data/app_recordings/collection_missing.csv \
  --report-json ./gt_singer_grader/data/app_recordings/collection_materialize_report.json \
  --strict

python3 -m gt_singer_grader.export_app_collection_packet \
  --checklist ./gt_singer_grader/data/app_recordings/collection_checklist.csv \
  --output-dir ./gt_singer_grader/data/app_recordings/collection_packet \
  --summary-json ./gt_singer_grader/data/app_recordings/collection_packet_summary.json \
  --strict

python3 -m gt_singer_grader.app_label_coverage \
  --csv ./gt_singer_grader/data/app_recordings/review_labels.csv \
  --audio-root . \
  --require-audio-files \
  --output-json ./gt_singer_grader/data/app_recordings/label_coverage_report.json \
  --strict

python3 -m gt_singer_grader.build_manifest app-recordings \
  --csv ./gt_singer_grader/data/app_recordings/review_labels.csv \
  --output ./gt_singer_grader/manifests/app_recordings.jsonl

python3 -m gt_singer_grader.filter_manifest \
  --input ./gt_singer_grader/manifests/app_recordings.jsonl \
  --trainable-output ./gt_singer_grader/manifests/app_recordings_trainable.jsonl \
  --eval-only-output ./gt_singer_grader/manifests/app_recordings_eval_only.jsonl \
  --summary-output ./gt_singer_grader/manifests/app_recordings_filter_summary.json

python3 -m gt_singer_grader.manifest \
  ./gt_singer_grader/manifests/app_recordings_trainable.jsonl \
  --require-trainable

python3 -m gt_singer_grader.split_manifest \
  --input ./gt_singer_grader/manifests/app_recordings_trainable.jsonl \
  --train-output ./gt_singer_grader/manifests/app_recordings_train.jsonl \
  --val-output ./gt_singer_grader/manifests/app_recordings_val.jsonl \
  --summary-output ./gt_singer_grader/manifests/app_recordings_split_summary.json \
  --val-ratio 0.2 \
  --strict-non-empty \
  --strict-family-coverage

python3 -m gt_singer_grader.merge_manifest \
  --input ./gt_singer_grader/manifests/app_recordings_val.jsonl \
  --input ./gt_singer_grader/manifests/app_recordings_eval_only.jsonl \
  --output ./gt_singer_grader/manifests/app_recordings_eval.jsonl \
  --summary-output ./gt_singer_grader/manifests/app_recordings_eval_summary.json

python3 -m gt_singer_grader.audit_app_validation \
  --manifest ./gt_singer_grader/manifests/app_recordings_eval.jsonl \
  --output-json ./gt_singer_grader/manifests/app_recordings_eval_audit.json
```

Use `app_recordings_train.jsonl` only for fine-tuning. Use
`app_recordings_eval.jsonl` for app-domain validation and threshold sweeps.

## Inference

```bash
cd NanoPitch/server/technique
python3 -m gt_singer_grader.infer \
  --checkpoint ./gt_singer_grader/runs/gtsinger_song_aug_v1/checkpoints/best.pth \
  --audio path/to/sample.wav \
  --target-family vibrato
```

If `--target-family` is set, the script also emits:

- a `grade` from 0-100
- `target_strength`
- `off_target_strength`
- a short feedback sentence

## Browser Demo

The packaged browser demo does not require the GT Singer dataset. It only needs:

- the `gt_singer_grader` code
- the packaged checkpoint at `gt_singer_grader/models/technique_demo_best.pth`
- Python with `numpy` and `torch`

For a fresh environment:

```bash
cd NanoPitch/server/technique
python3 -m pip install -r gt_singer_grader/requirements-demo.txt
```

```bash
python3 -m gt_singer_grader.demo \
  --checkpoint ./gt_singer_grader/models/technique_demo_best.pth \
  --port 8765 \
  --open-browser
```

Then open `http://127.0.0.1:8765`.

On Windows, the easier launcher is:

```powershell
.\gt_singer_grader\launch_demo.ps1
```

or double-click:

```text
gt_singer_grader\launch_demo.bat
```

The PowerShell launcher now tries, in order:

- `NanoPitch/.venv/Scripts/python.exe`
- `../.venvs/nanopitch/Scripts/python.exe`
- `../.venv/Scripts/python.exe`
- `python`
- `py -3`

The demo accepts a `.wav` upload, optionally lets you choose the intended
technique, and returns:

- detected technique
- confidence
- a `well done / developing / needs work / uncertain` verdict
- short feedback text
- clip-level and frame-level score breakdowns

## Notes

- `Paired_Speech_Group` is skipped by default for now.
- Validation can be grouped with `--split-group song` or `--split-group speaker`. Use `speaker` for a more realistic generalization check; it keeps entire GT Singer performers out of training.
- Public supplemental datasets should be added in order: GT Singer baseline
  first, VocalSet second, app-recording validation as soon as labeled clips
  exist. DAMP-style mobile datasets are useful for robustness, but they are not
  direct technique-label sources unless we label a subset.
- Audio loading uses only Python stdlib + PyTorch, so there are no new heavy dependencies beyond the repo's existing stack.
- The packaged demo checkpoint lives at `gt_singer_grader/models/technique_demo_best.pth`.
