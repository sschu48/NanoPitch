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

| Axis | Reference | Method | Grade tier (v1–v4) |
|---|---|---|---|
| **Pitch** | Score (MIDI → target Hz) | NanoPitch f0 vs. target, in cents | Score-match |
| **Tempo** | Score (`start_beat × 60/bpm`) | Onset detection vs. target onsets | Score-match |
| **Dynamics** | Physiology (used any range?) | A-weighted RMS curve | Range-used |
| **Technique** | Physiology (healthy vibrato?) | Classifier trained on GTSinger | Healthiness |

Pitch and tempo are score-matched because their references are
derivable from any MIDI. Dynamics and technique fall back to
physiological rules (healthy vibrato rate, dynamic range used) until a
reference recording exists (v5) or annotations are authored.

We deliberately avoid grading subjective qualities (was the expression
appropriate, was the phrasing musical) because we lack both labeled
data and the musical expertise to author per-song targets by hand.

## Detector-first, coach later

Every axis has a *physiology* or *score-derived* grade available
without authoring any per-song annotations (see "Grading per axis"
below). That gets us a working coach as soon as the detectors exist.

To upgrade from "graded by physiology" to "graded against this specific
song's musical intent," each song needs per-note targets. We don't
author those targets by hand. Instead:

1. Build the four detectors (v1 → v4).
2. Record a reference take of each preset song (us, a guest singer, or
   borrowed from GTSinger).
3. Run the detectors on the reference take. The detector's output *is*
   the per-note target.
4. Future user takes are scored against that auto-generated target,
   promoting Grade A into Grade C — concrete coaching messages
   ("target vibrato was 6Hz, yours was 4Hz — a little faster").

This sidesteps the annotation problem entirely — the musical expertise
lives in the reference singer, not in us or an LLM.

## Grading per axis

Each axis has up to three grade tiers. The system uses the strongest
tier available for the song being sung, and every axis always has at
least one available — no axis is left ungraded.

| Tier | Baseline | Needs |
|---|---|---|
| **A** | Physiology / take-internal | Nothing (always available) |
| **B** | Score-declared expectation | Hand-annotated targets in song JSON |
| **C** | Reference recording | Pro take of the same song, run through detectors |

Concrete measurement per axis:

| Axis | Grade A (physiology / always) | Grade B (score-annotated) | Grade C (reference take) |
|---|---|---|---|
| Pitch | — | cents off vs. MIDI target *(always available — MIDI is the score)* | — |
| Tempo | — | onset error vs. `start_beat × 60/bpm` *(always available)* | — |
| Dynamics | RMS range used (0dB → 0, 20dB → 100) | per-note loudness matches marked `f`/`p`/etc. | Pearson r of user vs. reference RMS curve |
| Technique | vibrato rate/depth/regularity in healthy range; falsetto pitch stability | detected technique matches `expect.technique` per note | detected technique matches reference's detected technique per note |

Pitch and tempo collapse to a single grade because MIDI inherently
provides their reference. Dynamics and technique have a useful Grade A
that works for any song without extra investment, and earn richer
grades when a song is annotated (B) or has a reference recording (C).

Each grade is a 0–100 score from a per-axis scaling curve (linear with
a tolerance knee — e.g. 0¢ → 100, 50¢ → 50, 200¢ → 0). The overall
score is a weighted mean across axes, with weights stored in the song
JSON so they can vary per song.

## Phased roadmap

| Phase | Scope | Estimate | Status |
|---|---|---|---|
| **v1** | Pitch — score-match grading. Metronome, piano-roll, post-record report. | ~1 week | In progress |
| v2 | Tempo — DSP onset detection vs. score onsets. | ~few days | Planned |
| v3 | Dynamics — RMS curve + range-used grade. | ~1–2 days | Planned |
| v4 | Technique — classifier trained on GTSinger; healthiness grade from physiology. | **2–4 weeks** (ML risk) | Planned |
| v5 | Reference-derived targets — record a pro take per song; run detectors; promote to reference-match grading. | ~few days once detectors work | Future |
| v6 | Upload your own MIDI; key transposition; free-pace recording with DTW. | Future | Future |

**Realistic scope for ~3 weeks: v1 + v2 + v3.** That delivers a coach
that grades pitch and tempo and shows dynamics — a complete-feeling
report without the ML training risk. v4 is the only true model-training
milestone and can easily eat 2–4 weeks on its own; treat it as a
separate sprint rather than part of the initial push.

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
- **Grade A (v3 default)**: range-used score. `range_db = p95(rms) - p5(rms)`;
  map 0 dB → 0, 20 dB → 100. Catches the common amateur problem of
  singing everything at one volume without needing annotations.
- **Display**: loudness curve overlaid on the piano-roll, normalized to
  the take's mean (so mic gain washes out and only expressive shape
  shows).
- **Grade B / C** (later): per-note loudness vs. marked dynamic, or
  curve correlation against a reference take.

### Technique (v4)
- **Training data**: [GTSinger](https://github.com/AaronZ345/GTSinger) —
  phoneme-level annotations for six techniques (mixed voice, falsetto,
  breathy, pharyngeal, vibrato, glissando) with controlled comparison
  groups. 80h, 20 singers, 9 languages.
- **Model shape**: NanoPitch-style frame-level encoder + multi-label
  classification head. Train, export to ONNX, load alongside
  `nanopitch.wasm` in the browser.
- **Grade A (v4 default)**: healthiness from physiology. For each
  detected vibrato event: rate (FFT of f0) scored on a bell curve
  centered at 6 Hz; depth scored on a bell curve centered at 60¢;
  regularity from cycle-period variance. For each falsetto event:
  pitch stability from f0 stddev. Aggregate = mean across detected
  events. If no techniques detected, no grade (don't penalize users
  for not attempting techniques).
- **Grade B / C** (later): detected technique vs. annotated expectation
  or vs. reference take's detected techniques per note.

### Overall score

Each axis is normalized to 0–100 by its scaling curve. The overall
score is a weighted mean across the axes graded for the current phase,
with weights stored in the song JSON (`grading.weights`). Default to
equal weights; tune per song from testing.

A hymn might weight pitch + tempo heavily; an aria might weight
technique. Per-song weights let the same engine grade different repertoire fairly.

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
