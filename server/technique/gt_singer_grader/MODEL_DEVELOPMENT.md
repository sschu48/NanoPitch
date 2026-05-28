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
2. **NanoPitch app recordings**: required target-domain source. Collect short
   consented clips with intended technique and reviewer labels.
3. **VocalSet / CVT vocal mode dataset / SVQTD**: optional supplemental sources
   if license and taxonomy mapping are acceptable.
4. **DAMP / mobile karaoke datasets**: useful for domain adaptation and
   robustness testing, but not a direct supervised technique source unless we
   label a subset ourselves.
5. **CCMusic Acapella**: useful later for quality/vocal-skill axes, not the
   core technique detector.

The machine-readable dataset registry lives in `dataset_registry.json`.

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

Then:

```bash
python3 -m gt_singer_grader.build_manifest app-recordings \
  --csv ./gt_singer_grader/data/app_recordings/labels.csv \
  --output ./gt_singer_grader/manifests/app_recordings.jsonl
```

Validate any manifest with:

```bash
python3 -m gt_singer_grader.manifest ./gt_singer_grader/manifests/gtsinger_english.jsonl
```

## Training Plan

Stage 1: make the existing model honest.

- Train on GT Singer with `--split-group speaker`.
- Keep clip-family prediction, VAD, and frame technique heads.
- Add realistic augmentation before any production claim:
  - room noise
  - room reverb
  - phone/laptop microphone EQ
  - compression and clipping
  - gain variation
  - optional backing-track leakage

Stage 2: make the output match the product.

- Replace forced softmax-only behavior with multi-label technique reporting.
- Add thresholds for `none` and `unclear`.
- Evaluate top-1, top-2, macro F1, false-positive rate, and calibration.

Stage 3: adapt to real users.

- Collect app recordings in 5-10 second clips.
- Label each as absent / weak / present / strong per technique.
- Keep a singer-held-out test set that is never used for tuning.
- Fine-tune or calibrate on app recordings.

Stage 4: consider a larger encoder.

If the current conv/GRU model plateaus, train a second prototype with a
pretrained audio/music encoder such as MERT, BEATs, or HTS-AT, then distill it
back to a smaller model if local/browser deployment matters.

## Evaluation Gates

Before replacing the packaged demo checkpoint, require:

- speaker-held-out GT Singer metrics
- app-recording validation metrics
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

- GT Singer raw data is not present locally.
- Python `torch` is not installed in the current environment.
- We do not yet have labeled app-user recordings.
- We have not implemented the augmentation layer in the technique trainer.
- The current packaged checkpoint is demo-grade and trained on clean GT Singer
  English recordings only.
