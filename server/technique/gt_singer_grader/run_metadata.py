"""Capture lightweight reproducibility metadata for technique-model runs."""

from __future__ import annotations

import platform
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    return {
        "path": str(path_obj),
        "exists": path_obj.is_file(),
        "bytes": path_obj.stat().st_size if path_obj.is_file() else None,
        "sha256": sha256_file(path_obj) if path_obj.is_file() else None,
    }


def _git_value(args: list[str], *, cwd: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def git_metadata(cwd: str | Path = ".") -> dict[str, Any]:
    root = _git_value(["rev-parse", "--show-toplevel"], cwd=cwd)
    commit = _git_value(["rev-parse", "HEAD"], cwd=cwd)
    branch = _git_value(["branch", "--show-current"], cwd=cwd)
    status = _git_value(["status", "--short"], cwd=cwd)
    return {
        "root": root,
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
    }


def collect_run_metadata(cwd: str | Path = ".") -> dict[str, Any]:
    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "git": git_metadata(cwd),
    }
