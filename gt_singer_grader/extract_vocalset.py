"""Extract VocalSet with Python's ZIP64-capable reader."""

from __future__ import annotations

import argparse
import os
import struct
import zipfile
from pathlib import Path
import zlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract VocalSet1-2.zip")
    parser.add_argument("--zip-path", default="./gt_singer_grader/data/VocalSet/VocalSet1-2.zip")
    parser.add_argument("--output-dir", default="./gt_singer_grader/data/VocalSet")
    return parser.parse_args()


def _safe_output_path(output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    resolved_output = output_dir.resolve()
    resolved_path = path.resolve()
    if resolved_output != resolved_path and resolved_output not in resolved_path.parents:
        raise ValueError(f"unsafe archive path: {filename}")
    return path


def _extract_sequential(zip_path: Path, output_dir: Path) -> int:
    extracted = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(zip_path, "rb") as handle:
        while True:
            signature = handle.read(4)
            if not signature:
                break
            if signature != b"PK\x03\x04":
                # Stop at central-directory/end records.
                if signature in {b"PK\x01\x02", b"PK\x05\x06", b"PK\x06\x06", b"PK\x06\x07"}:
                    break
                raise zipfile.BadZipFile(f"unexpected signature at {handle.tell() - 4}: {signature!r}")

            header = handle.read(26)
            if len(header) != 26:
                break
            (
                _version,
                flag_bits,
                method,
                _mtime,
                _mdate,
                _crc,
                compressed_size,
                uncompressed_size,
                filename_len,
                extra_len,
            ) = struct.unpack("<HHHHHIIIHH", header)
            filename = handle.read(filename_len).decode("utf-8", errors="replace")
            handle.seek(extra_len, os.SEEK_CUR)

            is_dir = filename.endswith("/")
            skip = filename.startswith("__MACOSX/") or Path(filename).name in {".DS_Store"}
            output_path = _safe_output_path(output_dir, filename)
            if is_dir:
                if not skip:
                    output_path.mkdir(parents=True, exist_ok=True)
                continue

            if not skip:
                output_path.parent.mkdir(parents=True, exist_ok=True)

            uses_descriptor = bool(flag_bits & 0x08)
            if method == 0 and not uses_descriptor:
                remaining = compressed_size
                with open(output_path, "wb") if not skip else open(os.devnull, "wb") as out:
                    while remaining:
                        chunk = handle.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise zipfile.BadZipFile(f"unexpected EOF in {filename}")
                        out.write(chunk)
                        remaining -= len(chunk)
            elif method == 8:
                decompressor = zlib.decompressobj(-15)
                with open(output_path, "wb") if not skip else open(os.devnull, "wb") as out:
                    while True:
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            raise zipfile.BadZipFile(f"unexpected EOF in {filename}")
                        data = decompressor.decompress(chunk)
                        if data:
                            out.write(data)
                        if decompressor.eof:
                            if decompressor.unused_data:
                                handle.seek(-len(decompressor.unused_data), os.SEEK_CUR)
                            break
                if uses_descriptor:
                    descriptor = handle.read(16)
                    if len(descriptor) < 12:
                        raise zipfile.BadZipFile(f"truncated data descriptor after {filename}")
                    if descriptor[:4] != b"PK\x07\x08":
                        handle.seek(-4, os.SEEK_CUR)
                elif compressed_size:
                    # The decompressor already positioned the stream at the next member.
                    pass
            else:
                raise NotImplementedError(f"unsupported compression method {method} for {filename}")

            if not skip:
                extracted += 1
                if extracted % 500 == 0:
                    print(f"extracted {extracted}", flush=True)

    return extracted


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path)
    output_dir = Path(args.output_dir)
    marker = output_dir / ".extracted-vocalset-1-2"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path) as archive:
            infos = [info for info in archive.infolist() if not info.filename.startswith("__MACOSX/")]
            total = len(infos)
            for index, info in enumerate(infos, 1):
                archive.extract(info, output_dir)
                if index % 500 == 0 or index == total:
                    print(f"extracted {index}/{total}", flush=True)
    except zipfile.BadZipFile as exc:
        print(f"standard extraction failed ({exc}); retrying sequential extraction", flush=True)
        total = _extract_sequential(zip_path, output_dir)
        print(f"sequential extraction wrote {total} files", flush=True)

    marker.write_text("ok\n", encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
