# GT Singer + VocalSet Quality Model

This package keeps GT Singer as the technique detector and adds the VocalSet weakly supervised quality calibrator.

Files:

- `technique_demo_best.pth` - GT Singer technique/VAD checkpoint.
- `vocalset_quality_best.pth` - VocalSet execution-quality calibrator.
- `metadata.json` - training source and validation metrics for the GT Singer checkpoint.

Demo launch:

```powershell
python -m gt_singer_grader.demo --model-profile gt_singer_vocalset --host 127.0.0.1 --port 8765
```

This profile predicts technique, section-level feedback, and a VocalSet quality score when an intended technique is selected.
