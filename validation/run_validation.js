#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const AUDIO_DIR = path.join(ROOT, 'validation', 'audio');
const RESULTS_DIR = path.join(ROOT, 'validation', 'results');
const SAMPLE_RATE = 16000;
const FRAME_SIZE = 160;

global.window = globalThis;
require(path.join(ROOT, 'coach', 'web', 'analyzer.js'));
const Analyzer = globalThis.NanoPitchAnalyzer;
const NanoPitchModule = require(path.join(ROOT, 'deployment', 'web', 'nanopitch.js'));

function ensureDirs() {
  fs.mkdirSync(AUDIO_DIR, { recursive: true });
  fs.mkdirSync(RESULTS_DIR, { recursive: true });
}

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

function round(value, digits = 1) {
  if (value == null || !Number.isFinite(value)) return null;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function centsError(actualHz, expectedHz) {
  if (!(actualHz > 0) || !(expectedHz > 0)) return null;
  return 1200 * Math.log2(actualHz / expectedHz);
}

function appendHarmonicSample(freq, t, harmonicWeights) {
  let value = 0;
  for (let h = 1; h <= harmonicWeights.length; h++) {
    value += harmonicWeights[h - 1] * Math.sin(2 * Math.PI * freq * h * t);
  }
  return value;
}

function fadeEnvelope(localT, duration, attack = 0.025, release = 0.05) {
  const a = attack > 0 ? clamp(localT / attack, 0, 1) : 1;
  const r = release > 0 ? clamp((duration - localT) / release, 0, 1) : 1;
  return Math.min(a, r);
}

function makeHarmonicTone({
  freq,
  duration,
  amp = 0.28,
  ampAt = () => 1,
  vibratoDepthCents = 0,
  vibratoHz = 5.5,
}) {
  const n = Math.round(duration * SAMPLE_RATE);
  const out = new Float32Array(n);
  const weights = [1.0, 0.45, 0.25, 0.14, 0.08];
  const norm = weights.reduce((sum, value) => sum + Math.abs(value), 0);

  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE;
    const vibrato = vibratoDepthCents
      ? 2 ** ((vibratoDepthCents * Math.sin(2 * Math.PI * vibratoHz * t)) / 1200)
      : 1;
    const env = fadeEnvelope(t, duration) * ampAt(t);
    out[i] = (amp * env * appendHarmonicSample(freq * vibrato, t, weights)) / norm;
  }
  return out;
}

function makePulseTrain({ freq, duration, bpm, burstS = 0.16, amp = 0.34 }) {
  const n = Math.round(duration * SAMPLE_RATE);
  const out = new Float32Array(n);
  const interval = 60 / bpm;
  const weights = [1.0, 0.45, 0.2, 0.1];
  const norm = weights.reduce((sum, value) => sum + Math.abs(value), 0);

  for (let start = 0.5; start <= duration - 0.35; start += interval) {
    const startIdx = Math.round(start * SAMPLE_RATE);
    const burstN = Math.round(burstS * SAMPLE_RATE);
    for (let j = 0; j < burstN && startIdx + j < n; j++) {
      const localT = j / SAMPLE_RATE;
      const t = (startIdx + j) / SAMPLE_RATE;
      const env = Math.sin(Math.PI * clamp(localT / burstS, 0, 1));
      out[startIdx + j] += (amp * env * appendHarmonicSample(freq, t, weights)) / norm;
    }
  }
  return out;
}

function writeWav(filePath, samples) {
  const dataSize = samples.length * 2;
  const buffer = Buffer.alloc(44 + dataSize);
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write('WAVE', 8);
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(SAMPLE_RATE, 24);
  buffer.writeUInt32LE(SAMPLE_RATE * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataSize, 40);
  for (let i = 0; i < samples.length; i++) {
    const value = clamp(samples[i], -1, 1);
    const intValue = Math.round(value < 0 ? value * 0x8000 : value * 0x7fff);
    buffer.writeInt16LE(intValue, 44 + i * 2);
  }
  fs.writeFileSync(filePath, buffer);
}

function buildFixtures() {
  return [
    {
      id: 'pitch_a4_harmonic',
      title: 'Pitch: A4 harmonic tone',
      description: 'Four seconds of a voice-like 440 Hz harmonic tone.',
      samples: makeHarmonicTone({
        freq: 440,
        duration: 4,
        amp: 0.3,
        vibratoDepthCents: 8,
      }),
      expected: {
        pitch_hz: 440,
        pitch_tolerance_cents: 75,
        min_voiced_percent: 45,
        max_range_db: 7,
      },
    },
    {
      id: 'tempo_120bpm_pulses',
      title: 'Tempo: 120 BPM pulses',
      description: 'Six seconds of repeated harmonic bursts every 0.5 seconds.',
      samples: makePulseTrain({
        freq: 330,
        duration: 6,
        bpm: 120,
      }),
      expected: {
        bpm: 120,
        bpm_tolerance: 10,
        min_onsets: 7,
        max_onsets: 14,
      },
    },
    {
      id: 'dynamics_constant',
      title: 'Dynamics: constant level',
      description: 'Four seconds of a steady harmonic tone at one amplitude.',
      samples: makeHarmonicTone({
        freq: 220,
        duration: 4,
        amp: 0.22,
      }),
      expected: {
        max_range_db: 5,
      },
    },
    {
      id: 'dynamics_soft_loud_soft',
      title: 'Dynamics: soft/loud/soft',
      description: 'Six seconds with a quiet section, a louder section, then quiet again.',
      samples: makeHarmonicTone({
        freq: 220,
        duration: 6,
        amp: 0.34,
        ampAt: (t) => {
          if (t < 2) return 0.22;
          if (t < 4) return 1.0;
          return 0.28;
        },
      }),
      expected: {
        min_range_db: 9,
      },
    },
  ];
}

async function loadNanoPitch() {
  const wasmBinary = fs.readFileSync(path.join(ROOT, 'deployment', 'web', 'nanopitch.wasm'));
  const module = await NanoPitchModule({ wasmBinary });
  const model = JSON.parse(fs.readFileSync(path.join(ROOT, 'deployment', 'web', 'model.json'), 'utf8'));
  const n = model.n_weights || model.weights.length;
  const condSize = model.cond_size || 64;
  const gruSize = model.gru_size || 96;
  const modelPtr = module._malloc(n * 4);

  for (let i = 0; i < n; i++) {
    module.HEAPF32[(modelPtr >> 2) + i] = model.weights[i];
  }

  const weightsPtr = module._nanopitch_load_weights(modelPtr, n, condSize, gruSize);
  if (!weightsPtr) throw new Error('NanoPitch rejected model weights');
  const statePtr = module._nanopitch_create_state(gruSize);
  if (!statePtr) throw new Error('Could not create NanoPitch state');

  return { module, modelPtr, weightsPtr, statePtr, gruSize };
}

function processFrame(engine, samples, timeS) {
  const inputPtr = engine.module._malloc(FRAME_SIZE * 4);
  for (let i = 0; i < FRAME_SIZE; i++) {
    engine.module.HEAPF32[(inputPtr >> 2) + i] = samples[i] || 0;
  }

  const outputPtr = engine.module._malloc((1 + 360 + 1 + 40) * 4);
  const valid = engine.module._nanopitch_process_frame(
    engine.weightsPtr,
    engine.statePtr,
    inputPtr,
    outputPtr,
  );

  let result = null;
  if (valid) {
    const base = outputPtr >> 2;
    const mel = new Float32Array(engine.module.HEAPF32.buffer, outputPtr + 362 * 4, 40);
    const vad = engine.module.HEAPF32[base];
    const f0 = engine.module.HEAPF32[base + 361];
    let sumSq = 0;
    for (let i = 0; i < FRAME_SIZE; i++) sumSq += samples[i] * samples[i];
    result = {
      time_s: timeS,
      vad,
      voiced: vad >= Analyzer.VAD_THRESHOLD && f0 > 0,
      f0_hz: f0,
      rms_db: 10 * Math.log10(sumSq / FRAME_SIZE + 1e-10),
      mel: Array.from(mel),
    };
  }

  engine.module._free(inputPtr);
  engine.module._free(outputPtr);
  return result;
}

function analyzeFixture(engine, fixture) {
  engine.module._nanopitch_reset_state(engine.statePtr, engine.gruSize);
  const frames = [];
  for (let offset = 0; offset + FRAME_SIZE <= fixture.samples.length; offset += FRAME_SIZE) {
    const samples = fixture.samples.subarray(offset, offset + FRAME_SIZE);
    const frame = processFrame(engine, samples, frames.length * Analyzer.HOP_S);
    if (frame) frames.push(frame);
  }
  const durationS = fixture.samples.length / SAMPLE_RATE;
  const report = Analyzer.buildLocalReport({ frames, duration_s: durationS });
  return { frames, report };
}

function axis(report, name) {
  return report.axes.find((item) => item.axis === name);
}

function evaluateFixture(fixture, report) {
  const checks = [];
  const pitch = axis(report, 'pitch');
  const tempo = axis(report, 'tempo');
  const dynamics = axis(report, 'dynamics');

  if (fixture.expected.pitch_hz) {
    const actual = pitch.metrics.median_f0_hz;
    const err = centsError(actual, fixture.expected.pitch_hz);
    checks.push({
      name: 'pitch median f0',
      expected: `${fixture.expected.pitch_hz} Hz +/- ${fixture.expected.pitch_tolerance_cents} cents`,
      actual: actual == null ? null : `${actual} Hz (${round(err, 1)} cents)`,
      pass: err != null && Math.abs(err) <= fixture.expected.pitch_tolerance_cents,
    });
  }

  if (fixture.expected.min_voiced_percent != null) {
    const actual = pitch.metrics.voiced_percent;
    checks.push({
      name: 'pitch voiced percent',
      expected: `>= ${fixture.expected.min_voiced_percent}%`,
      actual: `${actual}%`,
      pass: actual >= fixture.expected.min_voiced_percent,
    });
  }

  if (fixture.expected.bpm) {
    const actual = tempo.metrics.estimated_bpm;
    checks.push({
      name: 'tempo estimate',
      expected: `${fixture.expected.bpm} BPM +/- ${fixture.expected.bpm_tolerance}`,
      actual: actual == null ? null : `${actual} BPM`,
      pass: actual != null && Math.abs(actual - fixture.expected.bpm) <= fixture.expected.bpm_tolerance,
    });
  }

  if (fixture.expected.min_onsets != null || fixture.expected.max_onsets != null) {
    const actual = tempo.metrics.onset_count;
    checks.push({
      name: 'onset count',
      expected: `${fixture.expected.min_onsets}..${fixture.expected.max_onsets}`,
      actual,
      pass: actual >= fixture.expected.min_onsets && actual <= fixture.expected.max_onsets,
    });
  }

  if (fixture.expected.min_range_db != null) {
    const actual = dynamics.metrics.range_used_db;
    checks.push({
      name: 'dynamic range minimum',
      expected: `>= ${fixture.expected.min_range_db} dB`,
      actual: actual == null ? null : `${actual} dB`,
      pass: actual != null && actual >= fixture.expected.min_range_db,
    });
  }

  if (fixture.expected.max_range_db != null) {
    const actual = dynamics.metrics.range_used_db;
    checks.push({
      name: 'dynamic range maximum',
      expected: `<= ${fixture.expected.max_range_db} dB`,
      actual: actual == null ? null : `${actual} dB`,
      pass: actual != null && actual <= fixture.expected.max_range_db,
    });
  }

  return {
    pass: checks.every((check) => check.pass),
    checks,
  };
}

function svgChart(result) {
  const width = 760;
  const height = 260;
  const pad = { left: 48, right: 18, top: 20, bottom: 28 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const frames = result.frames;
  const duration = Math.max(frames.at(-1)?.time_s || 1, 1);
  const pitchAxis = axis(result.report, 'pitch');
  const tempoAxis = axis(result.report, 'tempo');
  const dynamicsAxis = axis(result.report, 'dynamics');
  const pitchPoints = pitchAxis.timeline || [];
  const dbPoints = dynamicsAxis.timeline || [];
  const onsets = tempoAxis.timeline || [];

  const hzValues = pitchPoints.map((point) => point.f0_hz).filter((value) => value > 0);
  const minHz = Math.max(60, Math.min(...hzValues, 180) * 0.85);
  const maxHz = Math.max(260, Math.max(...hzValues, 520) * 1.15);
  const xFor = (time) => pad.left + (time / duration) * plotW;
  const yPitch = (hz) => pad.top + (1 - (hz - minHz) / Math.max(1, maxHz - minHz)) * plotH * 0.58;
  const yDb = (db) => pad.top + plotH * 0.68 + (1 - ((clamp(db, -70, 0) + 70) / 70)) * plotH * 0.28;

  const pitchPolyline = pitchPoints
    .map((point) => `${round(xFor(point.time_s), 1)},${round(yPitch(point.f0_hz), 1)}`)
    .join(' ');
  const dbPolyline = dbPoints
    .map((point) => `${round(xFor(point.time_s), 1)},${round(yDb(point.dbfs), 1)}`)
    .join(' ');
  const onsetLines = onsets
    .map((onset) => `<line x1="${round(xFor(onset.time_s), 1)}" y1="${pad.top}" x2="${round(xFor(onset.time_s), 1)}" y2="${height - pad.bottom}" class="onset" />`)
    .join('\n');

  return `
<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(result.title)} chart">
  <rect x="0" y="0" width="${width}" height="${height}" class="chart-bg" />
  <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" class="grid-strong" />
  <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" class="grid-strong" />
  <line x1="${pad.left}" y1="${pad.top + plotH * 0.62}" x2="${width - pad.right}" y2="${pad.top + plotH * 0.62}" class="grid" />
  ${onsetLines}
  <polyline points="${pitchPolyline}" class="pitch-line" fill="none" />
  <polyline points="${dbPolyline}" class="db-line" fill="none" />
  <text x="8" y="${pad.top + 12}" class="axis-label">pitch</text>
  <text x="8" y="${pad.top + plotH * 0.78}" class="axis-label">level</text>
  <text x="${width - 62}" y="${height - 8}" class="axis-label">${round(duration, 1)}s</text>
</svg>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function writeHtml(results) {
  const total = results.length;
  const passed = results.filter((result) => result.pass).length;
  const cards = results.map((result) => {
    const pitch = axis(result.report, 'pitch').metrics;
    const tempo = axis(result.report, 'tempo').metrics;
    const dynamics = axis(result.report, 'dynamics').metrics;
    const rows = result.checks.map((check) => `
      <tr class="${check.pass ? 'pass' : 'fail'}">
        <td>${escapeHtml(check.name)}</td>
        <td>${escapeHtml(check.expected)}</td>
        <td>${escapeHtml(check.actual)}</td>
        <td>${check.pass ? 'PASS' : 'FAIL'}</td>
      </tr>`).join('');

    return `
    <section class="fixture ${result.pass ? 'pass' : 'fail'}">
      <div class="fixture-head">
        <div>
          <h2>${escapeHtml(result.title)}</h2>
          <p>${escapeHtml(result.description)}</p>
        </div>
        <span>${result.pass ? 'PASS' : 'FAIL'}</span>
      </div>
      <audio controls src="../audio/${result.id}.wav"></audio>
      <div class="metric-grid">
        <div><strong>${escapeHtml(pitch.median_f0_hz)}</strong><span>median f0 Hz</span></div>
        <div><strong>${escapeHtml(pitch.voiced_percent)}</strong><span>voiced %</span></div>
        <div><strong>${escapeHtml(tempo.estimated_bpm)}</strong><span>estimated BPM</span></div>
        <div><strong>${escapeHtml(tempo.onset_count)}</strong><span>onsets</span></div>
        <div><strong>${escapeHtml(dynamics.range_used_db)}</strong><span>range dB</span></div>
      </div>
      ${svgChart(result)}
      <table>
        <thead><tr><th>Check</th><th>Expected</th><th>Actual</th><th>Status</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </section>`;
  }).join('\n');

  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NanoPitch Axis Validation</title>
  <style>
    :root { color-scheme: light; --ink: #171713; --muted: #6f695f; --line: #d8cfc0; --bg: #f8f4ec; --panel: #fffdf8; --good: #237b60; --bad: #b33d30; --teal: #23876f; --gold: #b87912; }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 28px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; color: var(--ink); background: var(--bg); }
    header { max-width: 1120px; margin: 0 auto 18px; display: flex; justify-content: space-between; gap: 16px; align-items: end; }
    h1, h2 { margin: 0; letter-spacing: 0; }
    header p, .fixture p { color: var(--muted); margin: 6px 0 0; }
    .summary { padding: 10px 14px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); font-weight: 700; }
    main { max-width: 1120px; margin: 0 auto; display: grid; gap: 16px; }
    .fixture { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 16px; }
    .fixture-head { display: flex; align-items: start; justify-content: space-between; gap: 14px; margin-bottom: 12px; }
    .fixture-head span { border-radius: 999px; padding: 6px 10px; color: #fff; background: var(--good); font-size: 0.8rem; font-weight: 700; }
    .fixture.fail .fixture-head span { background: var(--bad); }
    audio { width: 100%; margin: 8px 0 12px; }
    .metric-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); margin-bottom: 12px; }
    .metric-grid div { padding: 12px; background: #fff; min-width: 0; }
    .metric-grid strong { display: block; font-size: 1.35rem; line-height: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .metric-grid span { display: block; margin-top: 5px; color: var(--muted); font-size: 0.78rem; }
    svg { display: block; width: 100%; height: auto; border-radius: 6px; margin: 12px 0; }
    .chart-bg { fill: #171713; }
    .grid { stroke: #3a362e; stroke-width: 1; }
    .grid-strong { stroke: #5a5349; stroke-width: 1; }
    .pitch-line { stroke: var(--teal); stroke-width: 2.2; }
    .db-line { stroke: var(--gold); stroke-width: 2; }
    .onset { stroke: #d05a45; stroke-width: 1; opacity: 0.65; }
    .axis-label { fill: #d8d0c2; font-size: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    th, td { text-align: left; border-top: 1px solid var(--line); padding: 8px; vertical-align: top; }
    th { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; }
    tr.pass td:last-child { color: var(--good); font-weight: 700; }
    tr.fail td:last-child { color: var(--bad); font-weight: 700; }
    @media (max-width: 720px) { body { padding: 16px; } header, .fixture-head { display: block; } .summary { margin-top: 12px; } .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>NanoPitch Axis Validation</h1>
      <p>Generated ${escapeHtml(new Date().toISOString())}. Synthetic known-answer fixtures for pitch, tempo, and dynamics.</p>
    </div>
    <div class="summary">${passed}/${total} fixtures passed</div>
  </header>
  <main>${cards}</main>
</body>
</html>`;

  fs.writeFileSync(path.join(RESULTS_DIR, 'index.html'), html);
}

async function main() {
  ensureDirs();
  const engine = await loadNanoPitch();
  const fixtures = buildFixtures();
  const results = [];

  for (const fixture of fixtures) {
    const wavPath = path.join(AUDIO_DIR, `${fixture.id}.wav`);
    writeWav(wavPath, fixture.samples);
    const { frames, report } = analyzeFixture(engine, fixture);
    const evaluation = evaluateFixture(fixture, report);
    results.push({
      id: fixture.id,
      title: fixture.title,
      description: fixture.description,
      expected: fixture.expected,
      pass: evaluation.pass,
      checks: evaluation.checks,
      report,
      frames,
    });
  }

  const summary = {
    generated_at: new Date().toISOString(),
    passed: results.filter((result) => result.pass).length,
    total: results.length,
    results: results.map((result) => ({
      id: result.id,
      title: result.title,
      description: result.description,
      expected: result.expected,
      pass: result.pass,
      checks: result.checks,
      report: result.report,
    })),
  };

  fs.writeFileSync(path.join(RESULTS_DIR, 'validation_summary.json'), JSON.stringify(summary, null, 2));
  writeHtml(results);

  for (const result of results) {
    console.log(`${result.pass ? 'PASS' : 'FAIL'} ${result.id}`);
    for (const check of result.checks) {
      console.log(`  ${check.pass ? 'ok ' : 'bad'} ${check.name}: expected ${check.expected}, actual ${check.actual}`);
    }
  }
  console.log(`\nVisual report: ${path.join(RESULTS_DIR, 'index.html')}`);

  if (summary.passed !== summary.total) process.exit(1);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
