"""Download the GT Singer English subset from Hugging Face."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download GT Singer English data")
    parser.add_argument("--output-dir", default="./data/GTSinger")
    parser.add_argument("--language", default="English")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="GTSinger/GTSinger",
        repo_type="dataset",
        local_dir=args.output_dir,
        allow_patterns=[f"{args.language}/**"],
    )


if __name__ == "__main__":
    main()
