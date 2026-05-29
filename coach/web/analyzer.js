// analyzer.js - pure free-take analysis helpers.
//
// The MVP is detection-first: one recorded take goes in, one report-shaped
// object comes out. No axis claims a grade unless a target is provided later.

(function () {
  const HOP_S = 0.01;
  const SAMPLE_RATE = 16000;
  const FRAME_SIZE = 160;
  const VAD_THRESHOLD = 0.3;
  const PITCH_BINS = 360;
  const PITCH_FMIN = 31.7;
  const PITCH_CENTS_PER_BIN = 20;
  const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

  function clamp(x, lo, hi) {
    return Math.max(lo, Math.min(hi, x));
  }

  function mean(values) {
    if (!values.length) return null;
    return values.reduce((sum, value) => sum + value, 0) / values.length;
  }

  function percentile(values, p) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const idx = clamp(Math.round(p * (sorted.length - 1)), 0, sorted.length - 1);
    return sorted[idx];
  }

  function median(values) {
    return percentile(values, 0.5);
  }

  function midiToHz(midi) {
    return 440 * Math.pow(2, (midi - 69) / 12);
  }

  function hzToMidi(hz) {
    return 69 + 12 * Math.log2(hz / 440);
  }

  function binToHz(bin) {
    return PITCH_FMIN * Math.pow(2, bin * PITCH_CENTS_PER_BIN / 1200);
  }

  function hzToBin(hz) {
    if (!(hz > 0)) return -1;
    return 1200 * Math.log2(hz / PITCH_FMIN) / PITCH_CENTS_PER_BIN;
  }

  function hzToNote(hz) {
    if (!(hz > 0)) return '-';
    const midi = hzToMidi(hz);
    const note = Math.round(midi);
    const cents = Math.round((midi - note) * 100);
    const name = NOTE_NAMES[((note % 12) + 12) % 12];
    const octave = Math.floor(note / 12) - 1;
    const sign = cents >= 0 ? '+' : '';
    return `${name}${octave} ${sign}${cents}c`;
  }

  function centsOff(f0Hz, midi) {
    if (!(f0Hz > 0)) return null;
    return 1200 * Math.log2(f0Hz / midiToHz(midi));
  }

  function beatsToSeconds(beats, bpm) {
    return beats * 60 / bpm;
  }

  function voicedFrames(frames) {
    return frames.filter(frame => frame && frame.voiced && frame.f0_hz > 0);
  }

  function downsampleTimeline(points, maxPoints = 360) {
    if (points.length <= maxPoints) return points;
    const step = points.length / maxPoints;
    const out = [];
    for (let i = 0; i < maxPoints; i++) {
      out.push(points[Math.floor(i * step)]);
    }
    return out;
  }

  function summarizePitch(frames, durationS) {
    const voiced = voicedFrames(frames);
    const midis = voiced.map(frame => hzToMidi(frame.f0_hz)).filter(Number.isFinite);
    const f0s = voiced.map(frame => frame.f0_hz);
    const deltas = [];

    for (let i = 1; i < voiced.length; i++) {
      const gap = voiced[i].time_s - voiced[i - 1].time_s;
      if (gap <= HOP_S * 3) {
        deltas.push(Math.abs(hzToMidi(voiced[i].f0_hz) - hzToMidi(voiced[i - 1].f0_hz)) * 100);
      }
    }

    const p10 = percentile(midis, 0.1);
    const p90 = percentile(midis, 0.9);
    const medianF0 = median(f0s);
    const stability = median(deltas);
    const voicedPct = frames.length ? voiced.length / frames.length : 0;
    const pitchRange = p10 != null && p90 != null ? Math.max(0, p90 - p10) : null;

    return {
      axis: 'pitch',
      mode: 'detection',
      available: voiced.length > 0,
      headline: voiced.length ? 'Pitch contour detected' : 'No reliable pitch detected',
      feedback: voiced.length
        ? 'This summarizes vocal range and short-term pitch stability without judging against a melody.'
        : 'The take needs more voiced singing before pitch can be summarized.',
      metrics: {
        voiced_percent: Math.round(voicedPct * 1000) / 10,
        median_f0_hz: medianF0 != null ? Math.round(medianF0 * 10) / 10 : null,
        median_note: medianF0 != null ? hzToNote(medianF0) : null,
        pitch_range_semitones: pitchRange != null ? Math.round(pitchRange * 10) / 10 : null,
        stability_cents: stability != null ? Math.round(stability) : null,
      },
      timeline: downsampleTimeline(voiced.map(frame => ({
        time_s: round(frame.time_s, 3),
        f0_hz: round(frame.f0_hz, 2),
      }))),
      duration_s: round(durationS, 3),
    };
  }

  function computeNovelty(frames) {
    const novelty = [];
    let prevMel = null;
    let prevCents = null;
    let prevVad = 0;
    let prevRms = null;

    for (const frame of frames) {
      if (!frame) {
        novelty.push(0);
        continue;
      }

      let flux = 0;
      if (prevMel && frame.mel) {
        for (let b = 0; b < frame.mel.length; b++) {
          const d = frame.mel[b] - prevMel[b];
          if (d > 0) flux += d;
        }
      }
      if (frame.mel) prevMel = frame.mel;

      let pitchJump = 0;
      if (frame.f0_hz > 0) {
        const cents = 1200 * Math.log2(frame.f0_hz / 440);
        if (prevCents != null) pitchJump = Math.min(Math.abs(cents - prevCents), 1200);
        prevCents = cents;
      } else {
        prevCents = null;
      }

      const vadRise = Math.max(0, frame.vad - prevVad);
      prevVad = frame.vad;

      let levelRise = 0;
      if (prevRms != null) levelRise = Math.max(0, frame.rms_db - prevRms);
      prevRms = frame.rms_db;

      let value = 0.8 * flux + 0.025 * pitchJump + 2.5 * vadRise + 0.12 * levelRise;
      if (frame.vad < 0.15 && frame.rms_db < -55) value *= 0.2;
      novelty.push(Math.max(0, value));
    }

    return novelty;
  }

  function detectOnsets(frames, novelty) {
    const onsets = [];
    const refractoryFrames = 12;
    let sinceOnset = refractoryFrames;

    for (let i = 0; i < novelty.length; i++) {
      const start = Math.max(0, i - 120);
      const recent = novelty.slice(start, i).filter(value => value > 0);
      const localMean = mean(recent) || 0;
      const localStd = recent.length
        ? Math.sqrt(mean(recent.map(value => Math.pow(value - localMean, 2))))
        : 0;
      const threshold = Math.max(1.2, localMean + 1.6 * localStd);
      const frame = frames[i];
      const active = frame && (frame.vad > 0.15 || frame.rms_db > -50);
      const isOnset = active && novelty[i] > threshold && sinceOnset >= refractoryFrames;

      if (isOnset) {
        onsets.push({ time_s: round(frame.time_s, 3), strength: round(novelty[i], 3) });
        sinceOnset = 0;
      } else {
        sinceOnset++;
      }
    }

    return onsets;
  }

  function estimateBpm(novelty) {
    const values = novelty.filter(Number.isFinite);
    if (values.length < 180) return { bpm: null, confidence: null };

    const minBpm = 50;
    const maxBpm = 200;
    const lagMin = Math.round(60 / maxBpm / HOP_S);
    const lagMax = Math.round(60 / minBpm / HOP_S);
    const signal = values.slice(-Math.min(values.length, 900));
    const avg = mean(signal) || 0;
    let bestLag = null;
    let bestValue = -Infinity;
    let secondValue = -Infinity;

    for (let lag = lagMin; lag <= lagMax; lag++) {
      if (lag >= signal.length - 1) continue;
      let sum = 0;
      let energyA = 0;
      let energyB = 0;
      for (let i = 0; i < signal.length - lag; i++) {
        const a = signal[i] - avg;
        const b = signal[i + lag] - avg;
        sum += a * b;
        energyA += a * a;
        energyB += b * b;
      }
      const bpm = 60 / (lag * HOP_S);
      const prior = Math.exp(-Math.pow(bpm - 100, 2) / (2 * 45 * 45));
      const value = (sum / (Math.sqrt(energyA * energyB) + 1e-6)) * prior;
      if (value > bestValue) {
        secondValue = bestValue;
        bestValue = value;
        bestLag = lag;
      } else if (value > secondValue) {
        secondValue = value;
      }
    }

    if (bestLag == null || bestValue < 0.08) return { bpm: null, confidence: null };
    const confidence = clamp(bestValue - Math.max(0, secondValue), 0, 1);
    return {
      bpm: round(60 / (bestLag * HOP_S), 1),
      confidence: round(confidence * 100, 1),
    };
  }

  function summarizeTempo(frames, durationS) {
    const novelty = computeNovelty(frames);
    const onsets = detectOnsets(frames, novelty);
    const tempo = estimateBpm(novelty);
    const onsetGaps = [];

    for (let i = 1; i < onsets.length; i++) {
      onsetGaps.push(onsets[i].time_s - onsets[i - 1].time_s);
    }

    const avgGap = mean(onsetGaps);
    const gapSpread = onsetGaps.length && avgGap
      ? Math.sqrt(mean(onsetGaps.map(gap => Math.pow(gap - avgGap, 2))))
      : null;

    return {
      axis: 'tempo',
      mode: 'detection',
      available: onsets.length > 0,
      headline: onsets.length ? 'Rhythmic onsets detected' : 'No clear onsets detected',
      feedback: onsets.length
        ? 'This reports the take-internal pulse and onset activity without comparing against a score.'
        : 'Try a take with clearer note attacks or syllable starts for tempo detection.',
      metrics: {
        estimated_bpm: tempo.bpm,
        bpm_confidence_percent: tempo.confidence,
        onset_count: onsets.length,
        onset_rate_per_second: durationS > 0 ? round(onsets.length / durationS, 2) : null,
        onset_gap_std_s: gapSpread != null ? round(gapSpread, 3) : null,
      },
      timeline: downsampleTimeline(onsets, 240),
      duration_s: round(durationS, 3),
    };
  }

  function summarizeDynamics(frames, durationS) {
    const active = frames.filter(frame => frame && frame.rms_db > -80 && (frame.voiced || frame.rms_db > -55));
    const dbs = active.map(frame => frame.rms_db);
    const p5 = percentile(dbs, 0.05);
    const p95 = percentile(dbs, 0.95);
    const avg = mean(dbs);
    const peak = percentile(dbs, 0.99);
    const range = p5 != null && p95 != null ? Math.max(0, p95 - p5) : null;

    return {
      axis: 'dynamics',
      mode: 'detection',
      available: active.length > 0,
      headline: active.length ? 'Loudness contour detected' : 'No usable loudness contour detected',
      feedback: active.length
        ? 'This measures how much loud/soft contrast the take used, independent of microphone gain calibration.'
        : 'The take was too quiet or too short for a dynamics summary.',
      metrics: {
        average_dbfs: avg != null ? round(avg, 1) : null,
        peak_dbfs: peak != null ? round(peak, 1) : null,
        p5_dbfs: p5 != null ? round(p5, 1) : null,
        p95_dbfs: p95 != null ? round(p95, 1) : null,
        range_used_db: range != null ? round(range, 1) : null,
      },
      timeline: downsampleTimeline(active.map(frame => ({
        time_s: round(frame.time_s, 3),
        dbfs: round(frame.rms_db, 1),
      }))),
      duration_s: round(durationS, 3),
    };
  }

  function normalizeTechniqueResult(payload) {
    if (!payload || payload.ok === false) {
      return {
        axis: 'technique',
        mode: 'detection',
        available: false,
        headline: payload?.error || 'Technique service unavailable',
        feedback: "Start the local technique API to include Brady's GT Singer model output.",
        metrics: {
          status: 'service_unavailable',
        },
        timeline: [],
      };
    }
    if (payload.axis_result) return payload.axis_result;
    const summary = payload.summary || payload.prediction || payload;
    const dominantTechniques = Array.isArray(summary.dominant_techniques)
      ? Object.fromEntries(summary.dominant_techniques
        .filter(item => item && item.technique)
        .map(item => [item.technique, item.score]))
      : null;
    return {
      axis: 'technique',
      mode: summary.target_family ? 'targeted_detection' : 'multi_label_detection',
      available: Boolean(summary.detected_family)
        && summary.detection_status !== 'not_enough_voice'
        && summary.detection_status !== 'no_clear_technique',
      headline: summary.detected_family
        ? `Detected ${String(summary.detected_family).replaceAll('_', ' ')}`
        : 'Technique result received',
      feedback: summary.feedback || 'Technique probabilities are reported as detected model activity.',
      metrics: {
        status: summary.detection_status || null,
        detected_family: summary.detected_family || null,
        primary_technique: summary.primary_technique || null,
        primary_technique_score_percent: summary.primary_technique_score != null
          ? round(summary.primary_technique_score * 100, 1)
          : null,
        confidence_percent: summary.detected_confidence != null
          ? round(summary.detected_confidence * 100, 1)
          : null,
        voiced_percent: summary.voiced_ratio != null ? round(summary.voiced_ratio * 100, 1) : null,
        family_margin: summary.family_margin != null ? round(summary.family_margin, 3) : null,
        dominant_techniques: dominantTechniques,
        technique_scores: summary.technique_scores || null,
      },
      timeline: summary.technique_timeline || [],
    };
  }

  function buildLocalReport({ frames, duration_s }) {
    return {
      mode: 'free_take_detection',
      generated_at: new Date().toISOString(),
      input: {
        sample_rate: SAMPLE_RATE,
        duration_s: round(duration_s, 3),
        frame_count: frames.length,
      },
      axes: [
        summarizePitch(frames, duration_s),
        summarizeTempo(frames, duration_s),
        summarizeDynamics(frames, duration_s),
      ],
    };
  }

  function addTechniqueAxis(report, techniquePayload) {
    const next = { ...report, axes: [...report.axes] };
    const existingIndex = next.axes.findIndex(axis => axis.axis === 'technique');
    const axis = normalizeTechniqueResult(techniquePayload);
    if (existingIndex >= 0) next.axes[existingIndex] = axis;
    else next.axes.push(axis);
    return next;
  }

  function scoreNote(note, bpm, frames) {
    const startS = beatsToSeconds(note.start_beat, bpm);
    const endS = beatsToSeconds(note.start_beat + note.duration_beats, bpm);
    const windowFrames = frames.filter(frame => frame.time_s >= startS && frame.time_s < endS && frame.voiced);
    const cents = windowFrames
      .map(frame => centsOff(frame.f0_hz, note.midi))
      .filter(value => value != null && Number.isFinite(value));

    if (!cents.length) {
      return {
        midi: note.midi,
        start_s: startS,
        end_s: endS,
        duration_s: endS - startS,
        n_voiced_frames: 0,
        n_in_tune_frames: 0,
        in_tune_pct: null,
        mean_abs_cents: null,
        status: 'missed',
      };
    }

    const inTune = cents.filter(value => Math.abs(value) < 50).length;
    const signedMean = mean(cents);
    const inTunePct = inTune / cents.length;
    return {
      midi: note.midi,
      start_s: startS,
      end_s: endS,
      duration_s: endS - startS,
      n_voiced_frames: cents.length,
      n_in_tune_frames: inTune,
      in_tune_pct: inTunePct,
      mean_abs_cents: mean(cents.map(value => Math.abs(value))),
      status: inTunePct > 0.6 ? 'hit' : signedMean < 0 ? 'flat' : 'sharp',
    };
  }

  function aggregateScore(noteResults) {
    const totalDuration = noteResults.reduce((sum, result) => sum + result.duration_s, 0);
    if (!noteResults.length || totalDuration <= 0) {
      return {
        overall_in_tune_pct: null,
        overall_mean_abs_cents: null,
        n_notes: 0,
        n_hit: 0,
        n_missed: 0,
        n_flat: 0,
        n_sharp: 0,
      };
    }

    let weightedTune = 0;
    let weightedCents = 0;
    let centsWeight = 0;
    for (const result of noteResults) {
      weightedTune += (result.in_tune_pct || 0) * result.duration_s;
      if (result.mean_abs_cents != null) {
        weightedCents += result.mean_abs_cents * result.duration_s;
        centsWeight += result.duration_s;
      }
    }

    return {
      overall_in_tune_pct: weightedTune / totalDuration,
      overall_mean_abs_cents: centsWeight > 0 ? weightedCents / centsWeight : null,
      n_notes: noteResults.length,
      n_hit: noteResults.filter(result => result.status === 'hit').length,
      n_missed: noteResults.filter(result => result.status === 'missed').length,
      n_flat: noteResults.filter(result => result.status === 'flat').length,
      n_sharp: noteResults.filter(result => result.status === 'sharp').length,
    };
  }

  function floatToWavBlob(samples, sampleRate = SAMPLE_RATE) {
    const bytesPerSample = 2;
    const dataSize = samples.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    writeAscii(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeAscii(view, 8, 'WAVE');
    writeAscii(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * bytesPerSample, true);
    view.setUint16(32, bytesPerSample, true);
    view.setUint16(34, 8 * bytesPerSample, true);
    writeAscii(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
      const value = clamp(samples[i], -1, 1);
      view.setInt16(offset, value < 0 ? value * 0x8000 : value * 0x7fff, true);
      offset += 2;
    }

    return new Blob([view], { type: 'audio/wav' });
  }

  function writeAscii(view, offset, text) {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  }

  function round(value, digits = 0) {
    if (value == null || !Number.isFinite(value)) return null;
    const scale = Math.pow(10, digits);
    return Math.round(value * scale) / scale;
  }

  window.NanoPitchAnalyzer = {
    HOP_S,
    SAMPLE_RATE,
    FRAME_SIZE,
    VAD_THRESHOLD,
    PITCH_BINS,
    PITCH_FMIN,
    PITCH_CENTS_PER_BIN,
    midiToHz,
    hzToMidi,
    binToHz,
    hzToBin,
    hzToNote,
    centsOff,
    beatsToSeconds,
    scoreNote,
    aggregateScore,
    summarizePitch,
    summarizeTempo,
    summarizeDynamics,
    buildLocalReport,
    addTechniqueAxis,
    normalizeTechniqueResult,
    floatToWavBlob,
  };
})();
