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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Project 2 technique analysis API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-upload-mb", type=int, default=25)
    return parser.parse_args()


def default_checkpoint() -> Path:
    return Path(__file__).resolve().parent / "gt_singer_grader" / "models" / "technique_demo_best.pth"


def default_metadata() -> Path:
    return Path(__file__).resolve().parent / "gt_singer_grader" / "models" / "technique_demo_metadata.json"


def display_name(value: str) -> str:
    return value.replace("_", " ")


def load_package_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "exists": False,
            "path": str(path),
            "release_ready": False,
            "promotion_eligible": False,
            "app_validation_ready": False,
            "candidate_kind": None,
        }

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    promotion = payload.get("promotion") if isinstance(payload.get("promotion"), dict) else {}
    app_validation = (
        payload.get("app_validation_audit") if isinstance(payload.get("app_validation_audit"), dict) else {}
    )
    app_domain_comparison = (
        payload.get("app_domain_comparison") if isinstance(payload.get("app_domain_comparison"), dict) else {}
    )
    contract = payload.get("model_contract") if isinstance(payload.get("model_contract"), dict) else {}
    candidate_kind = contract.get("candidate_kind")
    promotion_eligible = promotion.get("eligible") is True
    app_validation_ready = app_validation.get("ready_for_mvp_validation") is True
    app_domain_comparison_ready = app_domain_comparison.get("ok") is True
    release_ready = (
        candidate_kind == "app_adapted"
        and promotion_eligible
        and app_validation_ready
        and app_domain_comparison_ready
    )
    return {
        "exists": True,
        "path": str(path),
        "packaged_at": payload.get("packaged_at"),
        "candidate_kind": candidate_kind,
        "contract": contract,
        "release_ready": release_ready,
        "promotion_failed_gates": promotion.get("failed_gates") or [],
        "promotion_unknown_gates": promotion.get("unknown_gates") or [],
        "promotion_eligible": promotion_eligible,
        "app_validation_ready": app_validation_ready,
        "app_domain_comparison_ready": app_domain_comparison_ready,
        "app_domain_comparison_failed_checks": app_domain_comparison.get("failed_checks") or [],
        "metrics": payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
        "operating_point": payload.get("operating_point") if isinstance(payload.get("operating_point"), dict) else {},
    }


def build_axis_result(summary: dict[str, Any], assessment: dict[str, Any]) -> dict[str, Any]:
    target_family = summary.get("target_family")
    detected_family = str(summary.get("detected_family", "unknown"))
    status = str(assessment.get("status") or summary.get("detection_status") or "")
    available = status not in {"not_enough_voice", "no_clear_technique"}
    dominant_techniques = summary.get("dominant_techniques") or []
    dominant_score_map = {
        str(item.get("technique")): float(item.get("score", 0.0))
        for item in dominant_techniques
        if isinstance(item, dict) and item.get("technique")
    }

    metrics: dict[str, Any] = {
        "status": status or None,
        "detected_family": detected_family,
        "primary_technique": summary.get("primary_technique"),
        "primary_technique_score_percent": round(float(summary.get("primary_technique_score", 0.0)) * 100.0, 1),
        "confidence_percent": round(float(summary.get("detected_confidence", 0.0)) * 100.0, 1),
        "voiced_percent": round(float(summary.get("voiced_ratio", 0.0)) * 100.0, 1),
        "family_margin": round(float(summary.get("family_margin", 0.0)), 3),
        "dominant_techniques": dominant_score_map or None,
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
        "timeline": summary.get("technique_timeline", []),
    }


class TechniqueApiHandler(BaseHTTPRequestHandler):
    predictor: Any
    package_metadata: dict[str, Any]
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
                "package_metadata": self.package_metadata,
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
                from gt_singer_grader.feedback import build_demo_assessment
                from gt_singer_grader.infer import predict_summary

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
    metadata = Path(args.metadata) if args.metadata else default_metadata()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    from gt_singer_grader.infer import load_predictor

    TechniqueApiHandler.predictor = load_predictor(str(checkpoint), device_name=args.device)
    TechniqueApiHandler.package_metadata = load_package_metadata(metadata)
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
