"""Download VocalSet from Zenodo.

VocalSet 1.2 is large, so this script streams the zip to disk and leaves
extraction as an explicit opt-in step.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from urllib.request import urlopen

from tqdm import tqdm


VOCALSET_12_URL = "https://zenodo.org/records/1442513/files/VocalSet1-2.zip?download=1"


def download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urlopen(url)
    total = int(request.headers.get("Content-Length", "0") or "0")
    mode = "ab" if output_path.exists() else "wb"
    existing = output_path.stat().st_size if output_path.exists() else 0
    if existing and total and existing >= total:
        print(f"Already downloaded: {output_path}")
        return
    headers = {}
    if existing:
        # urllib cannot add Range after urlopen, so restart if partial support is not available.
        print(f"Existing partial file found ({existing} bytes); remove it to restart cleanly if needed.")
    with open(output_path, mode) as handle, tqdm(total=total or None, initial=existing, unit="B", unit_scale=True, desc=output_path.name) as progress:
        while True:
            chunk = request.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            progress.update(len(chunk))


def extract(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download VocalSet 1.2 from Zenodo")
    parser.add_argument("--output-dir", default="./gt_singer_grader/data/VocalSet")
    parser.add_argument("--zip-name", default="VocalSet1-2.zip")
    parser.add_argument("--url", default=VOCALSET_12_URL)
    parser.add_argument("--extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    zip_path = output_dir / args.zip_name
    download(args.url, zip_path)
    if args.extract:
        extract(zip_path, output_dir)
    print(os.path.abspath(zip_path))


if __name__ == "__main__":
    main()
