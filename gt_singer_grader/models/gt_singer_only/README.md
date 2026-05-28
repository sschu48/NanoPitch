# GT Singer Only Technique Model

This package is the baseline technique coach model trained on GT Singer data.

Files:

- `technique_demo_best.pth` - GT Singer technique/VAD checkpoint.
- `metadata.json` - training source and validation metrics for the packaged checkpoint.

Demo launch:

```powershell
python -m gt_singer_grader.demo --model-profile gt_singer_only --host 127.0.0.1 --port 8765
```

This profile predicts technique and section-level feedback from the GT Singer model only.
