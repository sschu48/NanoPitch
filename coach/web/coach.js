// coach.js — main app: load model, fetch song, run recording lifecycle,
// drive the piano-roll, hand off frames to analyzer.js for scoring.
//
// Lifecycle:
//   1. Page load        → load NanoPitch wasm + v1_aug model.json
//   2. Song selected    → fetch song JSON, render expected notes on the roll
//   3. Record clicked   → request mic, schedule metronome ticks, start countdown
//   4. Countdown ends   → start the audio worklet feeding frames into NanoPitch
//   5. Each frame       → push (time_s, f0_hz, voiced) to a buffer; redraw roll
//   6. Last note ends   → stop, score, render the report

const MODEL_URL = '../../training/runs/v1_aug/model.json';
const COUNTDOWN_BEATS = 4; // one full bar of metronome before recording starts

// ── Module state ─────────────────────────────────────────────────────

const state = {
  pitchEngine: null,    // WASM module instance (from nanopitch.js)
  modelLoaded: false,
  song: null,           // currently selected song (parsed JSON)
  audioCtx: null,
  micStream: null,
  workletNode: null,
  recording: false,
  frames: [],           // [{ time_s, f0_hz, voiced }, ...]
  recordStart_s: 0,     // AudioContext time when beat 0 begins
};

// ── Init: load WASM + model ──────────────────────────────────────────

async function loadModel() {
  // TODO(impl):
  //   - call window.NanoPitch() (or whatever the wasm glue exposes) to
  //     instantiate the engine
  //   - fetch MODEL_URL, parse JSON, hand weights to engine
  //   - on success: state.modelLoaded = true, enable record button
  //   - on failure: show a clear error in #status and leave the button disabled
  setStatus('loading model… (not yet implemented)');
}

// ── Song handling ────────────────────────────────────────────────────

async function loadSong(url) {
  // TODO(impl):
  //   - fetch + parse JSON
  //   - validate (notes sorted, no overlaps, bpm > 0)
  //   - state.song = parsed; redraw the empty piano-roll with reference notes
  //   - update #song-info with title + duration
}

// ── Recording lifecycle ──────────────────────────────────────────────

async function startTake() {
  // TODO(impl):
  //   - getUserMedia({ audio: ... })
  //   - new AudioContext, attach mic to AudioWorkletNode
  //   - schedule metronome ticks (4 countdown + N song beats)
  //   - mark state.recordStart_s = audioCtx.currentTime + countdown_seconds
  //   - state.frames = []; state.recording = true
}

function stopTake() {
  // TODO(impl):
  //   - stop worklet + tracks
  //   - state.recording = false
  //   - call scoreTake() and renderReport(result)
}

// AudioWorklet message handler — runs whenever NanoPitch emits a frame
function onFrame(msg) {
  // TODO(impl):
  //   - msg = { f0_hz, voiced } from the worklet
  //   - compute time_s relative to state.recordStart_s
  //   - state.frames.push({ time_s, f0_hz: msg.f0_hz, voiced: msg.voiced })
  //   - if past last note end → stopTake()
}

// ── Scoring ──────────────────────────────────────────────────────────

function scoreTake() {
  const { song, frames } = state;
  const noteResults = song.notes.map(n =>
    NanoPitchAnalyzer.scoreNote(n, song.bpm, frames));
  const summary = NanoPitchAnalyzer.aggregateScore(noteResults);
  return { noteResults, summary };
}

function renderReport({ noteResults, summary }) {
  // TODO(impl): pretty-print into #report (per-note rows + overall summary)
}

// ── Piano-roll drawing ───────────────────────────────────────────────

function drawRoll() {
  // TODO(impl):
  //   - clear canvas
  //   - draw semitone grid for the pitch range covered by the song
  //   - draw reference notes as horizontal bars
  //   - draw recorded f0 trace as a polyline (only voiced frames)
  //   - if recording, draw the "now" line + scroll
}

// ── Wiring ───────────────────────────────────────────────────────────

function setStatus(s) { document.getElementById('status').textContent = s; }

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('song-select').addEventListener('change', e => {
    loadSong(e.target.value);
  });
  document.getElementById('record-btn').addEventListener('click', () => {
    if (state.recording) { stopTake(); } else { startTake(); }
  });
  loadModel();
  loadSong(document.getElementById('song-select').value);
});
