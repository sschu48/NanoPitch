"""Serve a small local browser demo for the GT Singer grader."""

from __future__ import annotations

import argparse
import html
import os
import tempfile
import threading
import warnings
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, message="'cgi' is deprecated.*")
    import cgi

from .constants import FAMILY_NAMES
from .feedback import build_demo_assessment, summary_to_json
from .infer import LoadedPredictor, load_predictor, predict_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local GT Singer technique-demo server")
    parser.add_argument("--checkpoint", default=None, help="Path to the checkpoint to load")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-upload-mb", type=int, default=25)
    parser.add_argument("--nanopitch-vad-checkpoint", default=None)
    parser.add_argument("--disable-nanopitch-vad", action="store_true")
    parser.add_argument("--open-browser", action="store_true", help="Open the local demo in your browser")
    return parser.parse_args()


def resolve_default_checkpoint() -> str:
    root = Path(__file__).resolve().parent
    candidates = [
        root / "models" / "technique_demo_best.pth",
        root / "runs" / "exp2_detail" / "checkpoints" / "best.pth",
        root / "runs" / "exp1" / "checkpoints" / "best.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("no demo checkpoint found under gt_singer_grader/models or gt_singer_grader/runs")


def _display_name(label: str) -> str:
    return label.replace("_", " ")


def _render_score_rows(
    scores: dict[str, float],
    *,
    highlight: str | None = None,
) -> str:
    rows = []
    for name, value in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        bar_class = "score-fill score-fill-hot" if name == highlight else "score-fill"
        rows.append(
            f"""
            <div class="score-row">
              <div class="score-header">
                <span>{html.escape(_display_name(name).title())}</span>
                <span>{value * 100.0:.1f}%</span>
              </div>
              <div class="score-track">
                <div class="{bar_class}" style="width: {max(0.0, min(100.0, value * 100.0)):.1f}%"></div>
              </div>
            </div>
            """
        )
    return "".join(rows)


def _render_results(filename: str, summary: dict[str, object], assessment: dict[str, object]) -> str:
    status_class = {
        "well_done": "status-good",
        "developing": "status-warn",
        "needs_work": "status-bad",
        "uncertain": "status-warn",
        "not_enough_voice": "status-warn",
        "detected_only": "status-neutral",
    }.get(str(assessment.get("status", "")), "status-neutral")

    stats = [
        ("Detected technique", str(assessment["detected_family_display"]).title()),
        ("Confidence", f"{float(assessment.get('confidence_percent', 0.0)):.1f}%"),
        ("Runner-up", str(assessment.get("runner_up_family_display", "")).title()),
        ("Margin", f"{float(assessment.get('family_margin', 0.0)) * 100.0:.1f} pts"),
        ("Voiced audio", f"{float(assessment.get('voiced_ratio_percent', 0.0)):.1f}%"),
    ]
    if "target_family_display" in assessment and assessment["target_family_display"]:
        stats.append(("Target", str(assessment["target_family_display"]).title()))
    if "grade" in assessment:
        stats.append(("Technique grade", f"{float(assessment['grade']):.1f}/100"))
        stats.append(("Target strength", f"{float(assessment['target_strength']) * 100.0:.1f}%"))
        stats.append(("Off-target", f"{float(assessment['off_target_strength']) * 100.0:.1f}%"))

    stat_cards = "".join(
        f"""
        <div class="stat-card">
          <span class="stat-label">{html.escape(label)}</span>
          <strong>{html.escape(value)}</strong>
        </div>
        """
        for label, value in stats
    )

    details_json = summary_to_json({"assessment": assessment, "prediction": summary})

    return f"""
    <section class="panel result-panel">
      <div class="result-hero">
        <div>
          <span class="badge {status_class}">{html.escape(str(assessment.get("badge", "Result")))}</span>
          <h2>{html.escape(str(assessment.get("headline", "Prediction ready")))}</h2>
          <p class="lede">{html.escape(str(assessment.get("feedback", "")))}</p>
        </div>
        <div class="file-chip">{html.escape(filename)}</div>
      </div>
      <div class="stat-grid">
        {stat_cards}
      </div>
      <div class="score-grid">
        <div class="panel inset">
          <h3>Uploaded WAV Technique Ranking</h3>
          {_render_score_rows(summary["family_probabilities"], highlight=str(summary["detected_family"]))}
        </div>
        <div class="panel inset">
          <h3>Frame-Level Technique Strength</h3>
          {_render_score_rows(summary["technique_scores"])}
        </div>
      </div>
      <details class="details-block">
        <summary>Show raw prediction JSON</summary>
        <pre>{html.escape(details_json)}</pre>
      </details>
    </section>
    """


def _render_page(
    *,
    checkpoint_path: str,
    max_seconds: float,
    checkpoint_epoch: int | None,
    val_metrics: dict[str, float],
    selected_target: str | None = None,
    error_message: str | None = None,
    results_html: str = "",
) -> str:
    options = ['<option value="">Auto-detect only</option>']
    for family in FAMILY_NAMES:
        selected = " selected" if selected_target == family else ""
        options.append(f'<option value="{html.escape(family)}"{selected}>{html.escape(_display_name(family).title())}</option>')

    error_block = (
        f'<div class="panel error-panel">{html.escape(error_message)}</div>'
        if error_message
        else ""
    )

    metric_chips = []
    if checkpoint_epoch is not None:
        metric_chips.append(f'<span class="chip">Checkpoint epoch: {checkpoint_epoch}</span>')
    clip_acc = val_metrics.get("clip_acc")
    tech_f1 = val_metrics.get("tech_macro_f1")
    if clip_acc is not None:
        metric_chips.append(f'<span class="chip">Clip acc: {float(clip_acc) * 100.0:.1f}%</span>')
    if tech_f1 is not None:
        metric_chips.append(f'<span class="chip">Technique F1: {float(tech_f1) * 100.0:.1f}%</span>')
    if val_metrics.get("_nanopitch_vad_active"):
        metric_chips.append('<span class="chip">VAD: NanoPitch</span>')
    else:
        metric_chips.append('<span class="chip">VAD: Technique model</span>')

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Technique Coach Demo</title>
  <style>
    :root {{
      --bg-top: #10212f;
      --bg-mid: #18374a;
      --bg-bottom: #f2ead9;
      --panel: rgba(255, 248, 238, 0.92);
      --panel-strong: #fff9f1;
      --ink: #16232b;
      --muted: #56656e;
      --line: rgba(22, 35, 43, 0.12);
      --accent: #d56b43;
      --accent-soft: #f3b995;
      --good: #2c7d58;
      --warn: #b66b18;
      --bad: #a33f3f;
      --neutral: #49687c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255, 198, 140, 0.18), transparent 28%),
        linear-gradient(180deg, var(--bg-top) 0%, var(--bg-mid) 42%, var(--bg-bottom) 100%);
      font-family: "Trebuchet MS", "Gill Sans", sans-serif;
    }}
    .shell {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      display: grid;
      gap: 18px;
      margin-bottom: 24px;
      color: #fff6eb;
      animation: fadeUp 0.45s ease-out;
    }}
    h1, h2, h3 {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      letter-spacing: 0.01em;
    }}
    .hero h1 {{
      font-size: clamp(2rem, 5vw, 3.6rem);
      line-height: 1.02;
      max-width: 11ch;
    }}
    .hero p {{
      margin: 0;
      max-width: 64ch;
      color: rgba(255, 246, 235, 0.86);
      font-size: 1rem;
      line-height: 1.55;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      font-size: 0.92rem;
      color: rgba(255, 246, 235, 0.75);
    }}
    .chip {{
      padding: 8px 12px;
      border: 1px solid rgba(255, 246, 235, 0.16);
      border-radius: 999px;
      background: rgba(255, 246, 235, 0.07);
      backdrop-filter: blur(10px);
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 22px 60px rgba(12, 28, 36, 0.18);
      animation: fadeUp 0.45s ease-out;
    }}
    .panel.inset {{
      background: var(--panel-strong);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.55);
    }}
    .form-panel {{
      display: grid;
      gap: 18px;
      padding: 24px;
    }}
    .form-grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }}
    label {{
      display: grid;
      gap: 8px;
      font-weight: 700;
      font-size: 0.96rem;
    }}
    input[type="file"], select {{
      width: 100%;
      padding: 14px 14px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.78);
      color: var(--ink);
      font: inherit;
    }}
    .hint {{
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.45;
    }}
    button {{
      align-self: start;
      border: none;
      border-radius: 999px;
      padding: 14px 22px;
      background: linear-gradient(135deg, var(--accent) 0%, #b84737 100%);
      color: white;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
      box-shadow: 0 12px 24px rgba(181, 71, 55, 0.28);
    }}
    button:hover {{
      transform: translateY(-1px);
      box-shadow: 0 16px 28px rgba(181, 71, 55, 0.34);
    }}
    button:disabled {{
      cursor: not-allowed;
      opacity: 0.52;
      transform: none;
      box-shadow: none;
    }}
    .secondary-button {{
      background: #335f71;
      box-shadow: 0 12px 24px rgba(51, 95, 113, 0.22);
    }}
    .danger-button {{
      background: #a33f3f;
      box-shadow: 0 12px 24px rgba(163, 63, 63, 0.22);
    }}
    .record-panel {{
      display: grid;
      gap: 16px;
      padding: 20px;
      background: rgba(255,255,255,0.55);
      border: 1px solid var(--line);
      border-radius: 18px;
    }}
    .record-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    .record-status {{
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .record-status.recording {{
      color: var(--bad);
      font-weight: 700;
    }}
    audio {{
      width: 100%;
      max-width: 520px;
    }}
    .error-panel {{
      margin-top: 18px;
      padding: 16px 18px;
      color: #7f1e1e;
      background: rgba(255, 237, 237, 0.95);
    }}
    .result-panel {{
      margin-top: 22px;
      padding: 24px;
      display: grid;
      gap: 22px;
    }}
    .result-hero {{
      display: grid;
      gap: 16px;
      grid-template-columns: 1fr auto;
      align-items: start;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 12px;
      margin-bottom: 12px;
      color: white;
      font-size: 0.84rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .status-good {{ background: var(--good); }}
    .status-warn {{ background: var(--warn); }}
    .status-bad {{ background: var(--bad); }}
    .status-neutral {{ background: var(--neutral); }}
    .lede {{
      margin-top: 10px;
      color: var(--muted);
      line-height: 1.6;
      max-width: 62ch;
    }}
    .file-chip {{
      border-radius: 18px;
      padding: 10px 14px;
      background: rgba(213, 107, 67, 0.1);
      color: #8c4024;
      font-size: 0.9rem;
      max-width: min(100%, 280px);
      overflow-wrap: anywhere;
    }}
    .stat-grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    }}
    .stat-card {{
      padding: 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.65);
      border: 1px solid var(--line);
      display: grid;
      gap: 6px;
    }}
    .stat-label {{
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .score-grid {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }}
    .score-grid .panel {{
      padding: 20px;
    }}
    .score-row {{
      display: grid;
      gap: 8px;
      margin-top: 14px;
    }}
    .score-header {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-size: 0.95rem;
    }}
    .score-track {{
      height: 10px;
      border-radius: 999px;
      background: rgba(24, 55, 74, 0.12);
      overflow: hidden;
    }}
    .score-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #5b8aa5 0%, #3b6b80 100%);
    }}
    .score-fill-hot {{
      background: linear-gradient(90deg, #f1a66d 0%, #d56b43 100%);
    }}
    .details-block {{
      border-top: 1px solid var(--line);
      padding-top: 16px;
    }}
    .details-block summary {{
      cursor: pointer;
      font-weight: 700;
    }}
    pre {{
      margin: 12px 0 0;
      padding: 14px;
      border-radius: 16px;
      overflow-x: auto;
      background: rgba(17, 31, 39, 0.95);
      color: #e7f4ff;
      font-size: 0.88rem;
      line-height: 1.45;
    }}
    @keyframes fadeUp {{
      from {{ opacity: 0; transform: translateY(10px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @media (max-width: 720px) {{
      .result-hero {{ grid-template-columns: 1fr; }}
      .shell {{ padding: 20px 14px 36px; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>Technique Coach Demo</h1>
      <p>
        Record a live take or upload a <code>.wav</code> singing clip, pick the intended technique if you want a pass/fail style verdict,
        and the model will estimate what technique is present and whether it came through well enough.
      </p>
      <div class="meta">
        <span class="chip">Checkpoint: {html.escape(os.path.basename(checkpoint_path))}</span>
        <span class="chip">Listening window: {max_seconds:.1f}s chunks</span>
        <span class="chip">Standalone from NanoPitch</span>
        {''.join(metric_chips)}
      </div>
    </section>
    <section class="panel form-panel">
      <div>
        <h2>Run A Clip</h2>
        <p class="hint">
          Record in the browser or choose an existing WAV file. NanoPitch VAD gates voiced frames before the technique model summarizes the take. Leave the target blank for pure technique detection. Choose a target technique if you want the demo
          to decide whether that technique was executed well or still needs work.
        </p>
      </div>
      <div class="record-panel">
        <div>
          <h3>Record Live Singing</h3>
          <p class="hint">Start recording, sing a short phrase, then stop and analyze the captured take.</p>
        </div>
        <div class="record-controls">
          <button type="button" class="secondary-button" id="start-recording">Start Recording</button>
          <button type="button" class="danger-button" id="stop-recording" disabled>Stop Recording</button>
          <span class="record-status" id="record-status">Microphone idle</span>
        </div>
        <audio id="recording-preview" controls hidden></audio>
      </div>
      <form method="post" enctype="multipart/form-data" class="form-grid">
        <label>
          Upload WAV
          <input type="file" name="audio" id="audio-input" accept=".wav,audio/wav" required>
        </label>
        <label>
          Intended Technique
          <select name="target_family">
            {''.join(options)}
          </select>
        </label>
        <div>
          <button type="submit">Analyze Clip</button>
        </div>
      </form>
    </section>
    {error_block}
    {results_html}
  </main>
  <script>
    const startButton = document.getElementById('start-recording');
    const stopButton = document.getElementById('stop-recording');
    const statusText = document.getElementById('record-status');
    const audioInput = document.getElementById('audio-input');
    const preview = document.getElementById('recording-preview');

    let audioContext = null;
    let mediaStream = null;
    let sourceNode = null;
    let processorNode = null;
    let recordedChunks = [];
    let recordedSampleRate = 44100;

    function setRecordingStatus(text, isRecording) {{
      statusText.textContent = text;
      statusText.classList.toggle('recording', Boolean(isRecording));
    }}

    function flattenChunks(chunks) {{
      const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
      const samples = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) {{
        samples.set(chunk, offset);
        offset += chunk.length;
      }}
      return samples;
    }}

    function writeString(view, offset, text) {{
      for (let i = 0; i < text.length; i += 1) {{
        view.setUint8(offset + i, text.charCodeAt(i));
      }}
    }}

    function encodeWav(samples, sampleRate) {{
      const bytesPerSample = 2;
      const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
      const view = new DataView(buffer);
      writeString(view, 0, 'RIFF');
      view.setUint32(4, 36 + samples.length * bytesPerSample, true);
      writeString(view, 8, 'WAVE');
      writeString(view, 12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * bytesPerSample, true);
      view.setUint16(32, bytesPerSample, true);
      view.setUint16(34, 8 * bytesPerSample, true);
      writeString(view, 36, 'data');
      view.setUint32(40, samples.length * bytesPerSample, true);

      let offset = 44;
      for (let i = 0; i < samples.length; i += 1) {{
        const sample = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
        offset += bytesPerSample;
      }}
      return new Blob([view], {{ type: 'audio/wav' }});
    }}

    function attachRecordedFile(blob) {{
      const file = new File([blob], 'live-technique-take.wav', {{ type: 'audio/wav' }});
      const transfer = new DataTransfer();
      transfer.items.add(file);
      audioInput.files = transfer.files;
      preview.src = URL.createObjectURL(blob);
      preview.hidden = false;
    }}

    async function startRecording() {{
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
        setRecordingStatus('This browser does not expose microphone recording.', false);
        return;
      }}

      mediaStream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
      audioContext = new AudioContext();
      recordedSampleRate = audioContext.sampleRate;
      recordedChunks = [];
      sourceNode = audioContext.createMediaStreamSource(mediaStream);
      processorNode = audioContext.createScriptProcessor(4096, 1, 1);
      processorNode.onaudioprocess = (event) => {{
        recordedChunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      }};
      sourceNode.connect(processorNode);
      processorNode.connect(audioContext.destination);
      startButton.disabled = true;
      stopButton.disabled = false;
      setRecordingStatus('Recording...', true);
    }}

    async function stopRecording() {{
      stopButton.disabled = true;
      if (processorNode) {{
        processorNode.disconnect();
        processorNode.onaudioprocess = null;
      }}
      if (sourceNode) {{
        sourceNode.disconnect();
      }}
      if (mediaStream) {{
        mediaStream.getTracks().forEach((track) => track.stop());
      }}
      if (audioContext) {{
        await audioContext.close();
      }}
      const samples = flattenChunks(recordedChunks);
      if (samples.length === 0) {{
        setRecordingStatus('No audio was captured. Try recording again.', false);
      }} else {{
        const blob = encodeWav(samples, recordedSampleRate);
        attachRecordedFile(blob);
        setRecordingStatus('Live take ready. Click Analyze Clip.', false);
      }}
      startButton.disabled = false;
    }}

    startButton.addEventListener('click', () => {{
      startRecording().catch((error) => {{
        setRecordingStatus(`Microphone error: ${{error.message}}`, false);
        startButton.disabled = false;
        stopButton.disabled = true;
      }});
    }});
    stopButton.addEventListener('click', () => {{
      stopRecording().catch((error) => setRecordingStatus(`Recording error: ${{error.message}}`, false));
    }});
  </script>
</body>
</html>
"""


@dataclass
class DemoState:
    predictor: LoadedPredictor
    max_upload_bytes: int
    lock: threading.Lock


class DemoServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], state: DemoState) -> None:
        super().__init__(server_address, DemoHandler)
        self.state = state


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_HEAD(self) -> None:
        if self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self._send_html(
            _render_page(
                checkpoint_path=self.server.state.predictor.checkpoint_path,
                max_seconds=self.server.state.predictor.max_seconds,
                checkpoint_epoch=self.server.state.predictor.checkpoint_epoch,
                val_metrics=self.server.state.predictor.val_metrics,
            ),
            head_only=True,
        )

    def do_GET(self) -> None:
        if self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self._send_html(
            _render_page(
                checkpoint_path=self.server.state.predictor.checkpoint_path,
                max_seconds=self.server.state.predictor.max_seconds,
                checkpoint_epoch=self.server.state.predictor.checkpoint_epoch,
                val_metrics=self.server.state.predictor.val_metrics,
            )
        )

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length > self.server.state.max_upload_bytes:
            self._send_html(
                _render_page(
                    checkpoint_path=self.server.state.predictor.checkpoint_path,
                    max_seconds=self.server.state.predictor.max_seconds,
                    checkpoint_epoch=self.server.state.predictor.checkpoint_epoch,
                    val_metrics=self.server.state.predictor.val_metrics,
                    error_message="That file is too large for the demo upload limit.",
                ),
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                },
            )
            target_family = str(form.getfirst("target_family", "")).strip() or None
            upload = form["audio"] if "audio" in form else None
            if upload is None or not getattr(upload, "filename", ""):
                raise ValueError("Please choose a WAV file first.")

            filename = os.path.basename(str(upload.filename))
            suffix = Path(filename).suffix.lower()
            if suffix != ".wav":
                raise ValueError("This demo currently expects a .wav file.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                temp_path = handle.name
                handle.write(upload.file.read())

            try:
                with self.server.state.lock:
                    summary = predict_summary(
                        self.server.state.predictor,
                        temp_path,
                        target_family=target_family,
                    )
                assessment = build_demo_assessment(summary)
                results_html = _render_results(filename, summary, assessment)
                body = _render_page(
                    checkpoint_path=self.server.state.predictor.checkpoint_path,
                    max_seconds=self.server.state.predictor.max_seconds,
                    checkpoint_epoch=self.server.state.predictor.checkpoint_epoch,
                    val_metrics=self.server.state.predictor.val_metrics,
                    selected_target=target_family,
                    results_html=results_html,
                )
                self._send_html(body)
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
        except Exception as exc:
            self._send_html(
                _render_page(
                    checkpoint_path=self.server.state.predictor.checkpoint_path,
                    max_seconds=self.server.state.predictor.max_seconds,
                    checkpoint_epoch=self.server.state.predictor.checkpoint_epoch,
                    val_metrics=self.server.state.predictor.val_metrics,
                    error_message=str(exc),
                ),
                status=HTTPStatus.BAD_REQUEST,
            )

    def _send_html(
        self,
        body: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        head_only: bool = False,
    ) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        if not head_only:
            self.wfile.write(encoded)


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint or resolve_default_checkpoint()
    predictor = load_predictor(
        checkpoint_path,
        device_name=args.device,
        nanopitch_vad_checkpoint=args.nanopitch_vad_checkpoint,
        use_nanopitch_vad=not args.disable_nanopitch_vad,
    )
    predictor.val_metrics["_nanopitch_vad_active"] = predictor.nanopitch_vad_model is not None
    state = DemoState(
        predictor=predictor,
        max_upload_bytes=max(1, args.max_upload_mb) * 1024 * 1024,
        lock=threading.Lock(),
    )
    server = DemoServer((args.host, args.port), state)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving GT Singer demo at {url}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    if predictor.nanopitch_vad_model is not None:
        print(f"Using NanoPitch VAD: {predictor.nanopitch_vad_path}")
    else:
        print("Using technique model VAD")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
