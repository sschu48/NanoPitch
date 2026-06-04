# Technique Training Snapshot

This snapshot records the model-training approaches we tried for the technique
axis. It is documentation only: experimental checkpoints, raw datasets,
TensorBoard logs, generated manifests, and run directories are intentionally not
committed to `main`.

The app continues to use the submitted packaged checkpoint:

```text
server/technique/gt_singer_grader/models/technique_demo_best.pth
```

## Goal

The technique axis was meant to classify broad singing-technique families from
the same WAV that drives the pitch, tempo, and dynamics report. The practical
goal for Project 2 was an end-to-end local API and a demo-grade checkpoint, not
a production-ready vocal-technique judge.

## Approaches Tried

| Approach | Architecture / data | What we learned | Decision |
|---|---|---|---|
| Packaged demo checkpoint | Compact Conv1d + GRU classifier trained on GT Singer English technique/control data | Works end-to-end through the local technique API and gives the app a fourth feedback axis. The dataset is not close enough to app recordings to call it production quality. | Submitted model |
| Locked GT Singer comparison baseline | Conv-GRU trained/evaluated on the same GT Singer song split used for RoFormer comparison | Best overall balance in the comparison set: stronger top-1/top-2 accuracy, macro F1, false-positive rate, and calibration than RoFormer candidates. | Keep as baseline |
| Band RoFormer v1 | Non-causal RoFormer-style encoder over log-mel frames, same output heads as Conv-GRU | Technique detection improved, but the model overfired and calibration was much worse. | Rejected, do not package |
| Band RoFormer v2 | Smaller RoFormer with stronger regularization and lower learning rate | Calibration improved versus v1, but top-1, top-2, clip macro F1, and raw false positives regressed. | Rejected, do not package |
| Supplemental/public-data plan | VocalSet and other singing datasets | Useful future training sources, but taxonomy, license, and app-domain mismatch need review before promoting a checkpoint. | Documented future work |
| App-recording adaptation plan | Labeled NanoPitch app recordings | This is the right validation/training direction, but we did not have enough reviewed app-style recordings by submission time. | Required next step |

## Comparable Validation Metrics

These comparison runs used the same locked GT Singer validation split. They are
included to show why the RoFormer experiments were not promoted.

| Run | Architecture | Top-1 | Top-2 | Clip macro F1 | Technique macro F1 | Control FPR | ECE | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `gtsinger_song_aug_v1` | Conv-GRU baseline | 0.5559 | 0.7207 | 0.4714 | 0.2183 | 0.2090 | 0.1697 | Baseline |
| `gtsinger_band_roformer_v1` | Band RoFormer | 0.5239 | 0.7021 | 0.4613 | 0.4225 | 0.3220 | 0.4159 | Rejected |
| `gtsinger_band_roformer_v2` | Band RoFormer | 0.5000 | 0.6649 | 0.4306 | 0.2230 | 0.3559 | 0.1318 | Rejected |

Promotion required more than one better number. A candidate needed to improve
macro F1 or top-2 accuracy without increasing control false positives,
non-technique false positives, or calibration error. RoFormer v1 was interesting
because technique macro F1 increased, but it failed the reliability gates. V2
was better calibrated, but it lost too much accuracy.

## What Is On `main`

`main` keeps the submitted app behavior stable:

- the packaged technique checkpoint remains unchanged
- no experimental `.pth` files are committed
- no generated `runs/`, `data/`, or local manifest directories are committed
- the browser app and local technique API continue to load the current submitted
  model
- technique inference now also runs the same packaged model over overlapping
  5-second sections with a 2.5-second stride, then aggregates repeated
  section-level evidence

The deeper RoFormer experiment code and run notes live on the
`technique-roformer-gameplan` branch. That branch is useful for comparing model
families and reproducing the rejected experiments, but it is not required for
the submitted app to run.

## Section-Level Detection Layer

The submitted checkpoint was trained with a 10-second model context. Instead of
changing that checkpoint or increasing the runtime context out of distribution,
the API keeps the same model and adds a detection layer around it:

```text
recording WAV
  -> 5s overlapping sections, stride 2.5s
  -> current packaged technique model
  -> per-section technique evidence
  -> aggregate techniques that repeat in adjacent sections
```

This better matches the product task. Singing technique is often local:
vibrato may appear near note endings, glissando may appear during one
transition, and breathiness may be phrase-specific. A single whole-recording
label can hide those events, especially when the singer changes technique during
the take.

The section layer is deliberately conservative:

- it does not change or retrain `technique_demo_best.pth`
- it preserves the existing whole-take summary fields
- it reports section-level evidence separately in the API response
- it requires repeated adjacent-section evidence for multi-section recordings
  before listing a technique as detected across the take

## Takeaway

The strongest evidence from these experiments is not that the technique model is
finished. It is that we tested a stronger architecture, measured the tradeoffs,
and chose not to ship it because it made the user-facing behavior less reliable.
For the submitted model, section-level detection is the safer improvement path:
it improves how we interpret the current checkpoint without changing which model
the app runs. The best next step is to collect and review app-style recordings,
then validate any future checkpoint against that target-domain set before
packaging it.
