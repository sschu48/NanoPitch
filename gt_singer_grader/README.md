# GT Singer Grader

This module is a separate singing-grader pipeline that sits beside NanoPitch. It does not modify NanoPitch. Instead, it borrows a few ideas that already work well there:

- the same 10 ms frame cadence for audio features
- a dedicated VAD head so we grade voiced singing instead of silence
- the same onset penalty (`0.75`) for smoothing voiced/unvoiced decisions during feedback

## What it trains

The first version is a technique grader, not a generic "good singer / bad singer" model. It learns from GT Singer control-vs-emphasis pairs and predicts:

- clip-level family: `control`, `breathy`, `glissando`, `mixed_voice`, `falsetto`, `pharyngeal`, `vibrato`
- frame-level VAD
- frame-level technique activity for `mix`, `falsetto`, `breathy`, `pharyngeal`, `glissando`, `vibrato`

The packaged demo now has two separate model profiles:

- `gt_singer_only`: GT Singer technique recognition and segment feedback.
- `gt_singer_vocalset`: the same GT Singer technique recognizer plus a VocalSet weakly supervised execution-quality calibrator.

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
cd NanoPitch
python -m gt_singer_grader.download_dataset --output-dir ./gt_singer_grader/data/GTSinger
```

## Train

```bash
cd NanoPitch
python -m gt_singer_grader.train \
  --dataset-root ./gt_singer_grader/data/GTSinger \
  --output-dir ./gt_singer_grader/runs/exp1 \
  --epochs 20 \
  --batch-size 8
```

Useful outputs:

- `train_manifest.jsonl`
- `val_manifest.jsonl`
- `checkpoints/best.pth`
- TensorBoard logs in `tb/`

## VocalSet Quality Training

VocalSet adds professional singer examples across many techniques. The current
quality trainer keeps GT Singer as the technique recognizer, then trains a
separate execution-quality calibrator from VocalSet audio-derived features.
Because VocalSet does not include explicit good/bad execution ratings, this is
weak supervision: matching professional technique examples are treated as good,
normal/straight/control-like clips are treated as average, and mismatched target
techniques provide low-quality contrast.

Download and extract VocalSet 1.2, then train:

```powershell
.\gt_singer_grader\start_vocalset_quality_training.ps1 -Download -Extract -Epochs 10
```

If VocalSet is already present:

```bash
python -m gt_singer_grader.vocalset_quality \
  --vocalset-root ./gt_singer_grader/data/VocalSet \
  --output-dir ./gt_singer_grader/runs/vocalset_quality \
  --epochs 10 \
  --batch-size 64
```

Outputs:

- `vocalset_manifest.json`
- `features.csv`
- `epoch_*.pth`
- `best.pth`

## Inference

```bash
cd NanoPitch
python -m gt_singer_grader.infer \
  --checkpoint ./gt_singer_grader/runs/exp1/checkpoints/best.pth \
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
- one of the packaged model folders under `gt_singer_grader/models/`
- Python with `numpy` and `torch`

For a fresh environment:

```bash
pip install -r gt_singer_grader/requirements-demo.txt
```

```bash
cd NanoPitch
python -m gt_singer_grader.demo \
  --model-profile gt_singer_only \
  --port 8765 \
  --open-browser
```

To demo the VocalSet-added quality model:

```bash
python -m gt_singer_grader.demo \
  --model-profile gt_singer_vocalset \
  --port 8765 \
  --open-browser
```

Then open `http://127.0.0.1:8765`.

On Windows, the easier launcher is:

```powershell
.\gt_singer_grader\launch_demo.ps1
```

or with the VocalSet quality calibrator:

```powershell
.\gt_singer_grader\launch_demo.ps1 -ModelProfile gt_singer_vocalset
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
- section-by-section timeline feedback
- VocalSet quality score when `gt_singer_vocalset` is selected and an intended technique is chosen

## Notes

- `Paired_Speech_Group` is skipped by default for now.
- Validation splits are grouped by `speaker + parent technique folder + song` so control, mixed-voice, and falsetto takes from the same song stay together and do not leak across train/val.
- Audio loading uses only Python stdlib + PyTorch, so there are no new heavy dependencies beyond the repo's existing stack.
- The GT Singer-only package lives at `gt_singer_grader/models/gt_singer_only/`.
- The GT Singer + VocalSet package lives at `gt_singer_grader/models/gt_singer_vocalset/`.
