# Technique API

Local JSON wrapper around Brady's GT Singer technique model.

```bash
make technique-api
```

Run this from the repo root while serving `coach/web/`. The browser records a
single take, builds the pitch/tempo/dynamics axes locally, then sends the same
`take.wav` to this API for the fourth project axis.

The browser app sends the recorded `take.wav` to:

```text
POST http://127.0.0.1:8765/analyze
```

The response includes `axis_result`, which matches the Project 2 report shape
used by the browser for pitch, tempo, dynamics, and technique.

The packaged checkpoint is a GT Singer English demo model. It is included so
the submitted project can run end-to-end with four axes, while the training and
validation workflow under `gt_singer_grader/` documents how to improve it with
additional datasets and app-recording validation.
