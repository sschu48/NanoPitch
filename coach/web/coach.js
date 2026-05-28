// coach.js - Project 2 final MVP browser app.
//
// Flow:
//   1. Load NanoPitch WASM + model weights.
//   2. Record one free take while showing live preview meters.
//   3. Convert the recorded 16 kHz PCM into a WAV artifact.
//   4. Re-analyze that same take for pitch, tempo, and dynamics.
//   5. Send the same WAV to the optional local technique API.

const MODEL_URL = '../../deployment/web/model.json';
const TECHNIQUE_API_URL = 'http://127.0.0.1:8765/analyze';
const TECHNIQUE_HEALTH_URL = 'http://127.0.0.1:8765/health';
const SCRIPT_BUFFER_SIZE = 512;

const state = {
  wasmModule: null,
  weightsPtr: null,
  statePtr: null,
  modelDataPtr: null,
  loadedGruSize: 96,
  modelLoaded: false,

  audioCtx: null,
  micStream: null,
  processorNode: null,
  recording: false,
  resampleBuf: new Float32Array(0),
  recordedChunks: [],
  liveFrames: [],
  timelineFrames: [],
  timelineOnsets: [],

  takeUrl: null,
  takeBlob: null,
  report: null,
};

function $(id) {
  return document.getElementById(id);
}

function setStatus(text, tone = 'neutral') {
  const el = $('status');
  el.textContent = text;
  el.dataset.tone = tone;
}

function setBadge(id, text, tone = 'neutral') {
  const el = $(id);
  el.textContent = text;
  el.dataset.tone = tone;
}

function formatMetricLabel(key) {
  return key.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatMetricValue(value) {
  if (value == null) return '-';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(1);
  if (typeof value === 'string') return value;
  return '';
}

async function loadModel() {
  if (typeof NanoPitchModule === 'undefined') {
    setStatus('NanoPitch WASM glue not found.', 'bad');
    setBadge('model-badge', 'Model unavailable', 'bad');
    return;
  }

  try {
    setStatus('Loading NanoPitch model...', 'neutral');
    state.wasmModule = await NanoPitchModule();
    const response = await fetch(MODEL_URL);
    if (!response.ok) throw new Error(`model fetch failed: ${response.status}`);
    const data = await response.json();
    loadModelObject(data);
    setStatus('Ready to record a take.', 'good');
    setBadge('model-badge', 'Model ready', 'good');
    $('record-btn').disabled = false;
  } catch (err) {
    setStatus(`Model load failed: ${err.message}`, 'bad');
    setBadge('model-badge', 'Model unavailable', 'bad');
  }
}

function loadModelObject(data) {
  const module = state.wasmModule;
  if (!module) throw new Error('WASM runtime is not loaded');
  if (!Array.isArray(data.weights) || data.weights.length === 0) {
    throw new Error('model JSON is missing weights');
  }

  const n = data.n_weights || data.weights.length;
  const condSize = data.cond_size || 64;
  const gruSize = data.gru_size || 96;
  if (n > data.weights.length) throw new Error('n_weights exceeds weights length');

  const ptr = module._malloc(n * 4);
  for (let i = 0; i < n; i++) module.HEAPF32[(ptr >> 2) + i] = data.weights[i];

  if (state.statePtr) module._nanopitch_free_state(state.statePtr);
  if (state.weightsPtr) module._nanopitch_free_weights(state.weightsPtr);
  if (state.modelDataPtr) module._free(state.modelDataPtr);

  state.weightsPtr = module._nanopitch_load_weights(ptr, n, condSize, gruSize);
  if (!state.weightsPtr) {
    module._free(ptr);
    throw new Error('WASM rejected model weights');
  }

  state.modelDataPtr = ptr;
  state.loadedGruSize = gruSize;
  state.statePtr = module._nanopitch_create_state(gruSize);
  if (!state.statePtr) throw new Error('could not create NanoPitch state');
  state.modelLoaded = true;
}

function resetModelState() {
  if (state.wasmModule && state.statePtr) {
    state.wasmModule._nanopitch_reset_state(state.statePtr, state.loadedGruSize);
  }
}

function processAudioFrame(samples, timeS) {
  const module = state.wasmModule;
  if (!module || !state.weightsPtr || !state.statePtr) return null;

  const inputPtr = module._malloc(NanoPitchAnalyzer.FRAME_SIZE * 4);
  for (let i = 0; i < NanoPitchAnalyzer.FRAME_SIZE; i++) {
    module.HEAPF32[(inputPtr >> 2) + i] = samples[i] || 0;
  }

  const outputPtr = module._malloc((1 + 360 + 1 + 40) * 4);
  const valid = module._nanopitch_process_frame(state.weightsPtr, state.statePtr, inputPtr, outputPtr);

  let result = null;
  if (valid) {
    const base = outputPtr >> 2;
    const pitch = new Float32Array(module.HEAPF32.buffer, outputPtr + 4, 360);
    const mel = new Float32Array(module.HEAPF32.buffer, outputPtr + 362 * 4, 40);
    const vad = module.HEAPF32[base];
    const f0 = module.HEAPF32[base + 361];
    let sumSq = 0;
    let pitchConfidence = 0;
    for (let i = 0; i < NanoPitchAnalyzer.FRAME_SIZE; i++) sumSq += samples[i] * samples[i];
    for (let i = 0; i < pitch.length; i++) pitchConfidence = Math.max(pitchConfidence, pitch[i]);

    result = {
      time_s: timeS,
      vad,
      voiced: vad >= NanoPitchAnalyzer.VAD_THRESHOLD && f0 > 0,
      f0_hz: f0,
      rms_db: 10 * Math.log10(sumSq / NanoPitchAnalyzer.FRAME_SIZE + 1e-10),
      pitch_confidence: pitchConfidence,
      mel: Array.from(mel),
    };
  }

  module._free(inputPtr);
  module._free(outputPtr);
  return result;
}

async function startRecording() {
  if (!state.modelLoaded || state.recording) return;

  resetModelState();
  state.recording = true;
  state.recordedChunks = [];
  state.liveFrames = [];
  state.timelineFrames = [];
  state.timelineOnsets = [];
  state.resampleBuf = new Float32Array(0);
  state.report = null;
  clearTakeUrl();
  renderReport(null);
  drawTimeline([]);

  $('record-btn').textContent = 'Stop';
  $('record-btn').classList.add('recording');
  setStatus('Recording...', 'recording');

  try {
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: NanoPitchAnalyzer.SAMPLE_RATE });
    state.micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: NanoPitchAnalyzer.SAMPLE_RATE,
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
      },
    });

    const source = state.audioCtx.createMediaStreamSource(state.micStream);
    state.processorNode = state.audioCtx.createScriptProcessor(SCRIPT_BUFFER_SIZE, 1, 1);
    state.processorNode.onaudioprocess = handleAudioProcess;
    source.connect(state.processorNode);
    state.processorNode.connect(state.audioCtx.destination);
  } catch (err) {
    stopAudioGraph();
    state.recording = false;
    $('record-btn').textContent = 'Record';
    $('record-btn').classList.remove('recording');
    setStatus(`Microphone error: ${err.message}`, 'bad');
  }
}

function handleAudioProcess(event) {
  const input = event.inputBuffer.getChannelData(0);
  const resampled = resampleChunk(input, state.audioCtx.sampleRate, NanoPitchAnalyzer.SAMPLE_RATE);
  state.resampleBuf = appendFloat32(state.resampleBuf, resampled);

  while (state.resampleBuf.length >= NanoPitchAnalyzer.FRAME_SIZE) {
    const samples = new Float32Array(state.resampleBuf.subarray(0, NanoPitchAnalyzer.FRAME_SIZE));
    state.resampleBuf = state.resampleBuf.slice(NanoPitchAnalyzer.FRAME_SIZE);

    const timeS = state.recordedChunks.length * NanoPitchAnalyzer.HOP_S;
    state.recordedChunks.push(samples);
    const frame = processAudioFrame(samples, timeS);
    if (frame) state.liveFrames.push(frame);
    if (state.liveFrames.length > 900) state.liveFrames.shift();
    state.timelineFrames = state.liveFrames;
    state.timelineOnsets = [];

    if (state.liveFrames.length % 3 === 0) {
      updateLiveMeters(frame);
      drawTimeline(state.liveFrames);
    }
  }
}

async function stopRecording() {
  if (!state.recording) return;
  state.recording = false;
  stopAudioGraph();

  $('record-btn').textContent = 'Record';
  $('record-btn').classList.remove('recording');

  const pcm = concatChunks(state.recordedChunks);
  if (pcm.length < NanoPitchAnalyzer.SAMPLE_RATE * 0.5) {
    setStatus('Take too short. Record at least half a second.', 'bad');
    return;
  }

  const durationS = pcm.length / NanoPitchAnalyzer.SAMPLE_RATE;
  state.takeBlob = NanoPitchAnalyzer.floatToWavBlob(pcm, NanoPitchAnalyzer.SAMPLE_RATE);
  setTakeUrl(state.takeBlob);
  setStatus('Analyzing recorded take...', 'neutral');

  const frames = await analyzePcmTake(pcm);
  let report = NanoPitchAnalyzer.buildLocalReport({ frames, duration_s: durationS });
  const tempoOnsets = report.axes.find(axis => axis.axis === 'tempo')?.timeline || [];
  state.timelineFrames = frames;
  state.timelineOnsets = tempoOnsets;
  state.report = report;
  renderReport(report, { techniquePending: true });
  drawTimeline(frames, tempoOnsets);

  const techniquePayload = await analyzeTechnique(state.takeBlob);
  report = NanoPitchAnalyzer.addTechniqueAxis(report, techniquePayload);
  state.report = report;
  renderReport(report);
  setStatus('Analysis complete.', 'good');
}

function stopAudioGraph() {
  if (state.processorNode) {
    state.processorNode.disconnect();
    state.processorNode = null;
  }
  if (state.micStream) {
    state.micStream.getTracks().forEach(track => track.stop());
    state.micStream = null;
  }
  if (state.audioCtx) {
    state.audioCtx.close();
    state.audioCtx = null;
  }
}

async function analyzePcmTake(pcm) {
  resetModelState();
  const frames = [];
  const frameSize = NanoPitchAnalyzer.FRAME_SIZE;
  for (let offset = 0; offset + frameSize <= pcm.length; offset += frameSize) {
    const timeS = frames.length * NanoPitchAnalyzer.HOP_S;
    const samples = pcm.subarray(offset, offset + frameSize);
    const frame = processAudioFrame(samples, timeS);
    if (frame) frames.push(frame);
    if (frames.length % 250 === 0) await new Promise(resolve => requestAnimationFrame(resolve));
  }
  return frames;
}

async function analyzeTechnique(wavBlob) {
  const target = $('technique-select').value;
  const form = new FormData();
  form.append('audio', wavBlob, 'take.wav');
  if (target) form.append('target_family', target);

  try {
    const response = await fetch(TECHNIQUE_API_URL, { method: 'POST', body: form });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    setBadge('technique-badge', 'Technique connected', 'good');
    return await response.json();
  } catch (err) {
    setBadge('technique-badge', 'Technique offline', 'warn');
    return null;
  }
}

function resampleChunk(input, sourceRate, targetRate) {
  if (sourceRate === targetRate) return new Float32Array(input);
  const ratio = sourceRate / targetRate;
  const outLen = Math.floor(input.length / ratio);
  const output = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = Math.min(input.length - 1, Math.floor(i * ratio));
    output[i] = input[srcIdx];
  }
  return output;
}

function appendFloat32(a, b) {
  const output = new Float32Array(a.length + b.length);
  output.set(a, 0);
  output.set(b, a.length);
  return output;
}

function concatChunks(chunks) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const output = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
}

function setTakeUrl(blob) {
  clearTakeUrl();
  state.takeUrl = URL.createObjectURL(blob);
  const link = $('download-take');
  link.href = state.takeUrl;
  link.download = `nanopitch-take-${new Date().toISOString().replaceAll(':', '-')}.wav`;
  link.removeAttribute('aria-disabled');
}

function clearTakeUrl() {
  if (state.takeUrl) URL.revokeObjectURL(state.takeUrl);
  state.takeUrl = null;
  const link = $('download-take');
  link.removeAttribute('href');
  link.setAttribute('aria-disabled', 'true');
}

function updateLiveMeters(frame) {
  if (!frame) return;
  $('live-vad').textContent = frame.vad > 0.01 ? `${Math.round(frame.vad * 100)}%` : '-';
  $('live-pitch').textContent = frame.f0_hz > 0 ? frame.f0_hz.toFixed(1) : '-';
  $('live-note').textContent = NanoPitchAnalyzer.hzToNote(frame.f0_hz);
  $('live-level').textContent = frame.rms_db > -90 ? frame.rms_db.toFixed(0) : '-';
}

function renderReport(report, options = {}) {
  const grid = $('axis-grid');
  const raw = $('raw-report');

  if (!report) {
    grid.innerHTML = `
      <section class="empty-report">
        <h2>No take analyzed</h2>
        <p>Record a take to generate pitch, tempo, dynamics, and technique detection from the same WAV.</p>
      </section>
    `;
    raw.textContent = '';
    return;
  }

  const axes = [...report.axes];
  if (options.techniquePending && !axes.some(axis => axis.axis === 'technique')) {
    axes.push({
      axis: 'technique',
      mode: 'detection',
      available: false,
      headline: 'Technique analysis pending',
      feedback: 'Sending the recorded WAV to the local technique model.',
      metrics: {},
    });
  }

  grid.innerHTML = axes.map(renderAxisCard).join('');
  raw.textContent = JSON.stringify(report, null, 2);
}

function renderAxisCard(axis) {
  const tone = axis.available ? 'good' : 'warn';
  const metrics = Object.entries(axis.metrics || {})
    .filter(([, value]) => value != null)
    .map(([key, value]) => {
      if (typeof value === 'object') return renderMetricBars(key, value);
      return `
        <div class="metric-row">
          <span>${formatMetricLabel(key)}</span>
          <strong>${formatMetricValue(value)}</strong>
        </div>
      `;
    })
    .join('');

  return `
    <article class="axis-card" data-axis="${axis.axis}">
      <div class="axis-head">
        <span class="axis-name">${formatMetricLabel(axis.axis)}</span>
        <span class="pill" data-tone="${tone}">${axis.mode}</span>
      </div>
      <h2>${axis.headline}</h2>
      <p>${axis.feedback || ''}</p>
      <div class="metric-list">${metrics || '<span class="muted">No metrics available.</span>'}</div>
    </article>
  `;
}

function renderMetricBars(key, values) {
  const entries = Object.entries(values)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, 6);
  const rows = entries.map(([name, value]) => {
    const pct = Math.max(0, Math.min(100, Number(value) * 100));
    return `
      <div class="bar-row">
        <span>${formatMetricLabel(name)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${pct.toFixed(1)}%"></div></div>
        <strong>${pct.toFixed(0)}%</strong>
      </div>
    `;
  }).join('');
  return `<div class="metric-group"><span>${formatMetricLabel(key)}</span>${rows}</div>`;
}

function drawTimeline(frames, onsets = []) {
  const canvas = $('take-canvas');
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(640, Math.floor(rect.width * ratio));
  const height = Math.max(260, Math.floor(rect.height * ratio));
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#151515';
  ctx.fillRect(0, 0, width, height);

  const padL = 44 * ratio;
  const padR = 14 * ratio;
  const padT = 16 * ratio;
  const padB = 28 * ratio;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  if (!frames.length) {
    ctx.fillStyle = '#8d8a82';
    ctx.font = `${12 * ratio}px system-ui`;
    ctx.fillText('Record a take to see the post-take contour.', padL, padT + 24 * ratio);
    return;
  }

  const duration = Math.max(frames[frames.length - 1].time_s, 1);
  const windowRange = activeTimelineWindow(frames, duration);
  const visibleFrames = frames.filter(frame =>
    frame.time_s >= windowRange.start && frame.time_s <= windowRange.end);
  const visibleOnsets = onsets.filter(onset =>
    onset.time_s >= windowRange.start && onset.time_s <= windowRange.end);
  const viewDuration = Math.max(0.2, windowRange.end - windowRange.start);

  const laneGap = 22 * ratio;
  const laneH = (plotH - laneGap) / 2;
  const pitchLane = { y: padT, h: laneH };
  const levelLane = { y: padT + laneH + laneGap, h: laneH };

  drawLaneGrid(ctx, padL, width - padR, pitchLane, ratio);
  drawLaneGrid(ctx, padL, width - padR, levelLane, ratio);

  const pitchMidis = visibleFrames
    .filter(frame => isPitchDisplayFrame(frame))
    .map(frame => NanoPitchAnalyzer.hzToMidi(frame.f0_hz))
    .filter(Number.isFinite);
  const pitchScale = scaleFromValues(pitchMidis, 4, 48, 84);

  const dbValues = visibleFrames
    .map(frame => frame.rms_db)
    .filter(value => Number.isFinite(value));
  const dbScale = scaleFromValues(dbValues, 10, -80, 0);

  const xFor = time => padL + ((time - windowRange.start) / viewDuration) * plotW;
  const yPitch = midi => pitchLane.y + (1 - normalize(midi, pitchScale.min, pitchScale.max)) * pitchLane.h;
  const yDb = db => levelLane.y + (1 - normalize(db, dbScale.min, dbScale.max)) * levelLane.h;

  ctx.strokeStyle = '#e05d43';
  ctx.lineWidth = ratio;
  ctx.globalAlpha = 0.65;
  for (const onset of visibleOnsets) {
    const x = xFor(onset.time_s);
    ctx.beginPath();
    ctx.moveTo(x, pitchLane.y);
    ctx.lineTo(x, levelLane.y + levelLane.h);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  drawPitchTrace(ctx, visibleFrames, xFor, yPitch, ratio);
  drawLevelTrace(ctx, visibleFrames, xFor, yDb, ratio);

  ctx.fillStyle = '#d8d3c8';
  ctx.font = `${11 * ratio}px system-ui`;
  ctx.fillText('pitch', 8 * ratio, pitchLane.y + 14 * ratio);
  ctx.fillText('level', 8 * ratio, levelLane.y + 14 * ratio);
  ctx.fillStyle = '#8d8a82';
  const rangeLabel = windowRange.cropped
    ? `${viewDuration.toFixed(1)}s shown / ${duration.toFixed(1)}s take`
    : `${duration.toFixed(1)}s`;
  ctx.fillText(rangeLabel, width - padR - ctx.measureText(rangeLabel).width, height - 8 * ratio);
}

function activeTimelineWindow(frames, duration) {
  const active = frames.filter(frame =>
    frame && (
      frame.voiced ||
      frame.vad > 0.16 ||
      frame.f0_hz > 0 ||
      frame.rms_db > -55
    ));

  if (!active.length) {
    return { start: 0, end: duration, cropped: false };
  }

  let start = Math.max(0, active[0].time_s - 0.75);
  let end = Math.min(duration, active[active.length - 1].time_s + 0.75);
  if (end - start < 2) {
    const mid = (start + end) / 2;
    start = Math.max(0, mid - 1);
    end = Math.min(duration, mid + 1);
  }
  const cropped = start > 0.05 || end < duration - 0.05;
  return { start, end, cropped };
}

function isPitchDisplayFrame(frame) {
  return frame && frame.f0_hz > 0 && Number.isFinite(frame.f0_hz);
}

function drawLaneGrid(ctx, x0, x1, lane, ratio) {
  ctx.strokeStyle = '#2c2c2c';
  ctx.lineWidth = ratio;
  for (let i = 0; i <= 3; i++) {
    const y = lane.y + (lane.h * i) / 3;
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
    ctx.stroke();
  }
}

function drawPitchTrace(ctx, frames, xFor, yPitch, ratio) {
  const points = [];
  let smoothedMidi = null;
  let lastTime = null;

  for (const frame of frames) {
    if (!isPitchDisplayFrame(frame)) continue;

    const gap = lastTime == null ? 0 : frame.time_s - lastTime;
    const midi = NanoPitchAnalyzer.hzToMidi(frame.f0_hz);
    if (!Number.isFinite(midi)) continue;

    if (smoothedMidi == null || gap > 0.45 || Math.abs(midi - smoothedMidi) > 8) {
      smoothedMidi = midi;
    } else {
      smoothedMidi = 0.35 * midi + 0.65 * smoothedMidi;
    }

    points.push({
      x: xFor(frame.time_s),
      y: yPitch(smoothedMidi),
      time_s: frame.time_s,
      strong: frame.voiced || frame.vad > 0.12,
    });
    lastTime = frame.time_s;
  }

  if (!points.length) return;

  ctx.strokeStyle = '#67d5b5';
  ctx.lineWidth = 2.4 * ratio;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  let started = false;

  for (let i = 0; i < points.length; i++) {
    const point = points[i];
    const prev = points[i - 1];
    const gap = prev ? point.time_s - prev.time_s : 0;
    if (!started || gap > 0.45) {
      ctx.moveTo(point.x, point.y);
      started = true;
    } else {
      ctx.lineTo(point.x, point.y);
    }
  }
  ctx.stroke();

  ctx.fillStyle = '#67d5b5';
  const dotRadius = Math.max(1.8 * ratio, 2);
  for (const point of points) {
    ctx.globalAlpha = point.strong ? 0.9 : 0.45;
    ctx.beginPath();
    ctx.arc(point.x, point.y, point.strong ? dotRadius : dotRadius * 0.8, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function drawLevelTrace(ctx, frames, xFor, yDb, ratio) {
  ctx.strokeStyle = '#e3b04b';
  ctx.lineWidth = 2 * ratio;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  let started = false;
  let smoothedDb = null;
  for (const frame of frames) {
    if (!Number.isFinite(frame.rms_db)) continue;
    smoothedDb = smoothedDb == null
      ? frame.rms_db
      : 0.18 * frame.rms_db + 0.82 * smoothedDb;
    const x = xFor(frame.time_s);
    const y = yDb(smoothedDb);
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    }
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function scaleFromValues(values, minSpan, fallbackMin, fallbackMax) {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!finite.length) return { min: fallbackMin, max: fallbackMax };

  let min = quantileSorted(finite, 0.05);
  let max = quantileSorted(finite, 0.95);
  if (max - min < minSpan) {
    const mid = (min + max) / 2;
    min = mid - minSpan / 2;
    max = mid + minSpan / 2;
  }
  const pad = (max - min) * 0.15;
  return {
    min: Math.max(fallbackMin, min - pad),
    max: Math.min(fallbackMax, max + pad),
  };
}

function quantileSorted(sortedValues, p) {
  const idx = Math.max(0, Math.min(sortedValues.length - 1, Math.round(p * (sortedValues.length - 1))));
  return sortedValues[idx];
}

function normalize(value, min, max) {
  if (max <= min) return 0.5;
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

async function refreshTechniqueHealth() {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 800);
    const response = await fetch(TECHNIQUE_HEALTH_URL, { signal: controller.signal });
    clearTimeout(timer);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    setBadge('technique-badge', 'Technique connected', 'good');
  } catch {
    setBadge('technique-badge', 'Technique offline', 'warn');
  }
}

window.addEventListener('DOMContentLoaded', () => {
  $('record-btn').addEventListener('click', () => {
    if (state.recording) stopRecording();
    else startRecording();
  });
  $('refresh-technique').addEventListener('click', refreshTechniqueHealth);
  window.addEventListener('resize', () => drawTimeline(state.timelineFrames, state.timelineOnsets));

  renderReport(null);
  drawTimeline([]);
  loadModel();
  refreshTechniqueHealth();
});
