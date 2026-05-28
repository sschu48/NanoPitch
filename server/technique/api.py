"""Local JSON API for Brady's GT Singer technique model.

Run from the repo root:

    python3 server/technique/api.py --port 8765

The browser app posts the recorded WAV to /analyze and receives the same
axis-result shape used by the rest of the Project 2 report.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, message="'cgi' is deprecated.*")
    import cgi

from gt_singer_grader.constants import FAMILY_NAMES
from gt_singer_grader.feedback import build_demo_assessment
from gt_singer_grader.infer import LoadedPredictor, load_predictor, predict_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Project 2 technique analysis API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-upload-mb", type=int, default=25)
    return parser.parse_args()


def default_checkpoint() -> Path:
    return Path(__file__).resolve().parent / "gt_singer_grader" / "models" / "technique_demo_best.pth"


def display_name(value: str) -> str:
    return value.replace("_", " ")


def build_axis_result(summary: dict[str, Any], assessment: dict[str, Any]) -> dict[str, Any]:
    target_family = summary.get("target_family")
    detected_family = str(summary.get("detected_family", "unknown"))
    available = assessment.get("status") != "not_enough_voice"

    metrics: dict[str, Any] = {
        "detected_family": detected_family,
        "confidence_percent": round(float(summary.get("detected_confidence", 0.0)) * 100.0, 1),
        "voiced_percent": round(float(summary.get("voiced_ratio", 0.0)) * 100.0, 1),
        "family_margin": round(float(summary.get("family_margin", 0.0)), 3),
        "technique_scores": summary.get("technique_scores", {}),
        "family_probabilities": summary.get("family_probabilities", {}),
    }

    if target_family:
        metrics.update(
            {
                "target_family": target_family,
                "target_strength_percent": round(float(summary.get("target_strength", 0.0)) * 100.0, 1),
                "off_target_percent": round(float(summary.get("off_target_strength", 0.0)) * 100.0, 1),
                "target_match_score": round(float(summary.get("grade", 0.0)), 1),
            }
        )

    return {
        "axis": "technique",
        "mode": "targeted_detection" if target_family else "detection",
        "available": bool(available),
        "headline": assessment.get("headline") or f"Detected {display_name(detected_family)}",
        "feedback": assessment.get("feedback") or "Technique probabilities are reported as model activity.",
        "metrics": metrics,
        "timeline": [],
    }


class TechniqueApiHandler(BaseHTTPRequestHandler):
    predictor: LoadedPredictor
    max_upload_bytes: int

    server_version = "NanoPitchTechniqueAPI/0.1"

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return

        self.send_json(
            {
                "ok": True,
                "checkpoint": self.predictor.checkpoint_path,
                "device": str(self.predictor.device),
                "families": FAMILY_NAMES,
                "checkpoint_epoch": self.predictor.checkpoint_epoch,
                "val_metrics": self.predictor.val_metrics,
            }
        )

    def do_POST(self) -> None:
        if self.path != "/analyze":
            self.send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return

        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "missing request body")
            return
        if content_length > self.max_upload_bytes:
            self.send_error_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "upload is too large")
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
            file_item = form["audio"] if "audio" in form else None
            if file_item is None or not getattr(file_item, "file", None):
                self.send_error_json(HTTPStatus.BAD_REQUEST, "missing audio file")
                return

            target_family = form.getfirst("target_family") or None
            if target_family == "":
                target_family = None
            if target_family is not None and target_family not in FAMILY_NAMES:
                self.send_error_json(HTTPStatus.BAD_REQUEST, f"unknown target family: {target_family}")
                return

            suffix = Path(getattr(file_item, "filename", "") or "take.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                temp_path = Path(handle.name)
                handle.write(file_item.file.read())

            try:
                summary = predict_summary(self.predictor, str(temp_path), target_family=target_family)
                assessment = build_demo_assessment(summary)
                self.send_json(
                    {
                        "ok": True,
                        "summary": summary,
                        "assessment": assessment,
                        "axis_result": build_axis_result(summary, assessment),
                    }
                )
            finally:
                temp_path.unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover - returned to browser for local demo debugging
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"ok": False, "error": message}, status=status)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[technique-api] {self.address_string()} - {format % args}")


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    TechniqueApiHandler.predictor = load_predictor(str(checkpoint), device_name=args.device)
    TechniqueApiHandler.max_upload_bytes = args.max_upload_mb * 1024 * 1024

    server = ThreadingHTTPServer((args.host, args.port), TechniqueApiHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Technique API listening at {url}")
    print(f"Health check: {url}/health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping technique API")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
