"""Download the GT Singer English subset from Hugging Face."""

from __future__ import annotations

import argparse
from typing import Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download GT Singer English data")
    parser.add_argument("--output-dir", default="./data/GTSinger")
    parser.add_argument("--language", default="English")
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download every file for the language instead of only training-required WAV/JSON files.",
    )
    parser.add_argument(
        "--include-speech",
        action="store_true",
        help="Include Paired_Speech_Group files. The baseline training command leaves these out.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel Hugging Face download workers. Keep low for unauthenticated downloads.",
    )
    return parser.parse_args()


def load_snapshot_download() -> Callable[..., str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install training dependencies with:\n"
            "  python3 -m pip install -r gt_singer_grader/requirements-training.txt"
        ) from exc
    return snapshot_download


def allow_patterns(language: str, *, all_files: bool = False) -> list[str]:
    if all_files:
        return [f"{language}/**"]
    return [
        f"{language}/**/*.wav",
        f"{language}/**/*.json",
    ]


def ignore_patterns(language: str, *, include_speech: bool = False) -> list[str] | None:
    if include_speech:
        return None
    return [f"{language}/**/Paired_Speech_Group/**"]


def main() -> None:
    args = parse_args()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be >= 1")
    snapshot_download = load_snapshot_download()

    snapshot_download(
        repo_id="GTSinger/GTSinger",
        repo_type="dataset",
        local_dir=args.output_dir,
        allow_patterns=allow_patterns(args.language, all_files=args.all_files),
        ignore_patterns=ignore_patterns(args.language, include_speech=args.include_speech),
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
