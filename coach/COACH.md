# NanoPitch Coach (Project 2)

A browser-based singing coach. Pick a pre-loaded song, sing along to a
metronome, and get a per-axis report after the take.

Pitch detection comes from the NanoPitch model trained in project 1
(`training/runs/v1_aug/`). Future axes (tempo, technique, dynamics)
add their own detectors alongside it.

## The four axes

A singing coach grades more than pitch. We treat singing as four
independent measurable axes — each with its own data source and
evaluation strategy. They are deliberately *not* symmetric: only one of
them is a clean NanoPitch-style supervised learning problem, two are
DSP, and one is "already done" (pitch).

| Axis | Reference | Method | Output |
|---|---|---|---|
| **Pitch** | MIDI note → target Hz | NanoPitch f0 vs. target, in cents | **Graded** |
| **Tempo** | Score onset time = `start_beat × 60/bpm` | Onset detection vs. target onsets | **Graded** |
| **Technique** | (none in v1–v4) | Classifier trained on GTSinger labels | **Detected** |
| **Dynamics** | (none in v1–v4) | A-weighted RMS curve | **Detected** |

- **Graded** — the score declares the right answer. We score the user against it.
- **Detected** — we report what happened. We do not claim it should or shouldn't have happened.

This split is the most important design choice in the project. We
deliberately avoid grading subjective qualities (was the vibrato good?
was the dynamic appropriate?) because we lack both labeled training
data and the musical expertise to author per-song targets by hand.
Reporting without judgment still gives users actionable feedback —
they decide whether each detected event matched their intent.

## Detector-first, coach later

A pure detector becomes a coach once each song has *per-note targets*
("this note should have vibrato", "this phrase should be forte"). We
don't author those targets by hand. Instead:

1. Build the four detectors (v1 → v4).
2. Record a reference take of each preset song (us, a guest singer, or
   borrowed from GTSinger).
3. Run the detectors on the reference take. The detector's output *is*
   the per-note target.
4. Future user takes are scored against that auto-generated target,
   promoting reports into coaching messages ("target vibrato was 6Hz,
   yours was 4Hz — a little faster").

This sidesteps the annotation problem entirely — the musical expertise
lives in the reference singer, not in us or an LLM.

## Phased roadmap

| Phase | Scope | Status |
|---|---|---|
| **v1** | Pitch — graded against score. Metronome, piano-roll, post-record report. | In progress |
| v2 | Tempo — onset detection (DSP first; small model only if needed). Graded against score. | Planned |
| v3 | Dynamics — A-weighted RMS curve. Reported, not graded. | Planned |
| v4 | Technique — classifier trained on GTSinger (vibrato, falsetto, mixed voice, breathy, pharyngeal, glissando). Reported, not graded. | Planned |
| v5 | Reference-derived targets — record a pro take per song; run detectors; promote reports to coaching messages. | Future |
| v6 | Upload your own MIDI; key transposition; free-pace recording with DTW. | Future |

## v1 scope (this branch)

- 2–3 pre-loaded songs as JSON (`web/songs/*.song.json`).
- Pick song → countdown → record while metronome ticks (assumes user
  stays on tempo; no time alignment / DTW).
- During recording: piano-roll showing the next notes scrolling toward
  a "now" line, with a live dot for the user's current detected pitch.
- After recording: per-note table (MIDI note, expected vs. mean
  detected cents, in-tune %), plus an aggregate score.
- All in the browser. No backend.

Out of scope for v1 (deferred to later phases per the roadmap above):
- Tempo / onset grading
- Vibrato, falsetto, technique detection
- Dynamics
- User-uploaded MIDI, free-tempo / DTW alignment

## Tech stack

- Pitch detection: existing `deployment/web/nanopitch.{js,wasm}` + the
  v1_aug model from `training/runs/v1_aug/model.json` (regenerate with
  `python deployment/export_weights.py training/runs/v1_aug/checkpoints/best.pth -o training/runs/v1_aug/model.json`).
- Audio: `getUserMedia` + `AudioWorkletNode` (same approach as the
  existing live-pitch page).
- Metronome: `Web Audio API` (`OscillatorNode` clicks scheduled
  against `AudioContext.currentTime`).
- Display: vanilla canvas for the piano-roll. No framework, no
  external CSS lib.
- Songs: hand-authored JSON. See `coach/web/songs/scale.song.json`
  for the schema.

## Song JSON schema (v1)

```json
{
  "title": "C Major Scale",
  "bpm": 80,
  "time_signature": [4, 4],
  "key": "C major",
  "notes": [
    { "midi": 60, "start_beat": 0, "duration_beats": 1 },
    ...
  ]
}
```

- `midi`: MIDI note number (60 = middle C, 69 = A4 = 440 Hz).
- `start_beat`: when the note starts, in beats from song start.
- `duration_beats`: note length in beats.
- Quarter note = 1 beat; tempo determined by `bpm`.

Later phases extend per-note metadata with optional fields populated
either by hand or by running detectors on a reference take (v5). None
are required in v1.

## v1 scoring math (pitch only)

Convert MIDI → Hz: `f_ref = 440 * 2 ** ((midi - 69) / 12)`.

For each reference note over its time window `[t_start, t_end]`:
- Pull NanoPitch's f0 estimates `f0(t)` over that window (frames where
  the model said voiced).
- `cents_off(t) = 1200 * log2(f0(t) / f_ref)`
- **Note in-tune % = fraction of voiced frames with |cents_off| < 50**.
- **Note mean cents off = mean of |cents_off| across voiced frames**.
- If the user produced no voiced frames in the window: note is "missed"
  and contributes 0% in-tune.

Aggregate (across all notes in the song):
- **Overall in-tune % = duration-weighted mean of per-note in-tune %**.
- **Overall mean cents off = duration-weighted mean of |cents_off|
  across all voiced frames within reference note windows**.

Frames *outside* any reference note window (i.e. user singing during a
rest) are reported separately as "extra voicing" but don't affect the
in-tune score.

## Future axes — implementation notes

These are sketches to keep v1 honest about where the project is
heading. None are implemented yet.

### Tempo (v2)
- **Reference** per note: `t_ref = start_beat × 60 / bpm` (seconds
  from take start).
- **Detection**: start with spectral-flux onset detection
  (`librosa.onset`-style). Only train a model on GTSinger's
  phoneme-to-audio alignments if DSP proves too noisy on sung vocals.
- **Per-note metric**: signed onset error in ms.
- **Aggregate**: % of onsets within ±50ms; mean absolute error.

### Dynamics (v3)
- No model. A-weighted RMS over a short window (~25 ms), smoothed.
- **Output**: loudness curve overlaid on the piano-roll, plus per-note
  mean loudness relative to the take's overall mean (so the curve is
  expressive shape, not absolute SPL — mic gain washes out absolute).
- **Reported only.** Coaching ("more here, less here") waits for v5,
  when a reference recording supplies a target loudness curve.

### Technique (v4)
- **Training data**: [GTSinger](https://github.com/AaronZ345/GTSinger) —
  phoneme-level annotations for six techniques (mixed voice, falsetto,
  breathy, pharyngeal, vibrato, glissando) with controlled comparison
  groups (natural vs. technique-densely-employed). 80h, 20 singers,
  9 languages.
- **Model shape**: NanoPitch-style frame-level encoder + multi-label
  classification head. Train, export to ONNX, load alongside
  `nanopitch.wasm` in the browser.
- **Output**: per-note technique labels with confidence.
- **Reported only.** GTSinger has no "good/bad" labels — every singer
  is a pro. We can detect *what* technique is happening; we cannot
  judge quality from this data alone.

### Aggregation and weighting

Once multiple axes exist, the report needs an overall score. Two
constraints before averaging anything:

1. **Normalize each axis to a 0–100 scale** through a per-axis curve
   (e.g. 0¢ → 100, 50¢ → 50, 200¢ → 0). Cents-off, ms-off, and
   technique-correctness are not directly comparable.
2. **Weights are song-dependent** — put them in the song JSON, not in
   global code. A hymn weights pitch + timing heavily; an opera aria
   weights technique. Default to equal weights; tune per song after
   testing.

**Detected** axes (technique, dynamics) do not enter the aggregate
score in v1–v4. Only **graded** axes count toward the overall number.
This keeps the headline score meaningful and avoids penalizing users
for stylistic choices the score did not specify.

## Repo layout

```
coach/
├── COACH.md              this file
├── README.md             quick-start (how to run locally)
└── web/
    ├── index.html        UI shell (song picker, record button, canvas, report)
    ├── coach.js          main app logic — recording lifecycle, frame loop
    ├── analyzer.js       pure scoring functions (testable in isolation)
    ├── songs/
    │   ├── scale.song.json
    │   └── twinkle.song.json
    └── lib/              third-party JS goes here (empty for v1)
```

The coach loads `nanopitch.{js,wasm}` and `model.json` from the
`deployment/web/` and `training/runs/v1_aug/` directories via relative
paths — no copy/duplicate.

Future phases add sibling models (`tempo.{js,wasm}`, `technique.onnx`,
etc.) following the same pattern.

## Results

To be populated after v1 is implemented and tested against a few sung
takes. Will include:
- Sample takes with their generated reports
- Subjective notes on whether the score matches what a human would say
- Limitations observed during testing
