// analyzer.js — pure scoring functions, no DOM, no audio.
// Kept separate from coach.js so we can unit-test these in isolation.
//
// All functions here operate on plain data:
//   - notes:    array of { midi, start_beat, duration_beats }
//   - bpm:      number
//   - frames:   array of { time_s, f0_hz, voiced }  (output of NanoPitch)
//
// Time conventions:
//   - All times are in seconds, measured from the first beat of the song
//     (i.e. after the metronome countdown, not from recording start).

const HOP_S = 0.01; // NanoPitch frame hop (10 ms) — keep in sync with model

// ── Music-theory helpers ─────────────────────────────────────────────

function midiToHz(midi) {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

function centsOff(f0Hz, midi) {
  if (!(f0Hz > 0)) return null;
  return 1200 * Math.log2(f0Hz / midiToHz(midi));
}

function beatsToSeconds(beats, bpm) {
  return beats * 60 / bpm;
}

// ── Per-note scoring ─────────────────────────────────────────────────

// Score a single reference note against the recorded frame stream.
// Returns:
//   {
//     midi, start_s, end_s,
//     n_voiced_frames, n_in_tune_frames,
//     in_tune_pct,           // null if no voiced frames in window
//     mean_abs_cents,        // null if no voiced frames in window
//     status: 'hit' | 'flat' | 'sharp' | 'missed'
//   }
function scoreNote(note, bpm, frames) {
  // TODO(impl):
  //   1. Convert note window to seconds (start_s, end_s).
  //   2. Filter frames to that window.
  //   3. Of voiced frames, compute centsOff for each.
  //   4. in_tune_pct = fraction with |cents| < 50.
  //   5. mean_abs_cents = mean(|cents|).
  //   6. status: 'missed' if no voiced; else 'hit' if in_tune_pct > 0.6;
  //              else 'flat' if mean cents < 0; else 'sharp'.
  throw new Error('scoreNote not implemented');
}

// Aggregate a set of per-note results into a song-level summary.
// Returns:
//   {
//     overall_in_tune_pct,    // duration-weighted mean of per-note in-tune %
//     overall_mean_abs_cents, // duration-weighted mean of |cents|
//     n_notes, n_hit, n_missed, n_flat, n_sharp,
//   }
function aggregateScore(noteResults) {
  // TODO(impl): duration-weighted aggregation; ignore null in-tune-pct
  // entries (missed notes count toward n_missed but contribute 0% in-tune).
  throw new Error('aggregateScore not implemented');
}

// ── Public surface ───────────────────────────────────────────────────

window.NanoPitchAnalyzer = {
  midiToHz, centsOff, beatsToSeconds,
  scoreNote, aggregateScore,
  HOP_S,
};
