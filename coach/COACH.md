# NanoPitch Coach (Project 2)

A browser-based singing coach: pick a pre-loaded song, sing along to a
metronome, and get a per-note quality report after the take. Pitch
detection comes from the NanoPitch model trained in project 1
(`training/runs/v1_aug/`).

## v1 scope (what we're shipping first)

- 2-3 pre-loaded songs as JSON (`web/songs/*.song.json`).
- Pick song → countdown → record while metronome ticks (assumes user
  stays on tempo; no time alignment / DTW).
- During recording: piano-roll showing the next notes scrolling toward
  a "now" line, with a live dot for the user's current detected pitch.
- After recording: per-note table (MIDI note, expected vs. mean detected
  cents, in-tune %), plus an aggregate score.
- All in the browser. No backend.

Explicitly out of scope for v1:
- Upload your own MIDI / sheet music (v2)
- Free-tempo / DTW alignment (v2)
- Onset detection (v3)
- Vibrato / dynamics scoring (v3)
- Staff notation rendering (v3 — piano-roll is enough for pitch comparison)

## Tech stack

- Pitch detection: existing `deployment/web/nanopitch.{js,wasm}` + the
  v1_aug model from `training/runs/v1_aug/model.json` (regenerate with
  `python deployment/export_weights.py training/runs/v1_aug/checkpoints/best.pth -o training/runs/v1_aug/model.json`).
- Audio: `getUserMedia` + `AudioWorkletNode` (same approach as the
  existing live-pitch page).
- Metronome: `Web Audio API` (`OscillatorNode` clicks scheduled against
  `AudioContext.currentTime`).
- Display: vanilla canvas for the piano-roll. No framework, no
  external CSS lib.
- Songs: hand-authored JSON. See `coach/web/songs/scale.song.json` for
  the schema.

## Song JSON schema

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

## Quality scoring math

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

## Roadmap

| Phase | Scope |
|-------|-------|
| **v1 (this branch)** | Pre-loaded songs, metronome, piano-roll, post-record report |
| v2 | Upload your own MIDI; key transposition; free-pace recording with DTW |
| v3 | Onset detection (you don't have to start on the click); vibrato / dynamics scoring; staff notation |

## Results

To be populated after v1 is implemented and tested against a few sung
takes. Will include:
- Sample takes with their generated reports
- Subjective notes on whether the score matches what a human would say
- Limitations observed during testing
