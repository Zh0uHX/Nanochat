"""Reproducibility metadata for training checkpoints and reports."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Any


def configuration_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(project_root: str, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _content_manifest_sha256(
    project_root: str,
    relative_paths: list[str],
) -> str:
    """Hash paths, executable bits, symlink targets, and working-file contents."""
    digest = hashlib.sha256()
    for relative_path in sorted(set(relative_paths)):
        encoded_path = relative_path.encode("utf-8", errors="surrogateescape")
        digest.update(encoded_path)
        digest.update(b"\0")
        absolute_path = os.path.join(project_root, relative_path)
        if os.path.islink(absolute_path):
            digest.update(b"symlink\0")
            digest.update(os.readlink(absolute_path).encode("utf-8"))
        elif os.path.isfile(absolute_path):
            executable = bool(os.stat(absolute_path).st_mode & 0o111)
            digest.update(b"file+x\0" if executable else b"file\0")
            with open(absolute_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            # A tracked deletion must affect the manifest.
            digest.update(b"missing\0")
        digest.update(b"\0")
    return digest.hexdigest()


def _working_tree_sha256(project_root: str) -> str | None:
    listing = _git(
        project_root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
    )
    if listing is None:
        return None
    relative_paths = listing.splitlines() if listing else []
    return _content_manifest_sha256(project_root, relative_paths)


def collect_run_provenance(
    config: dict[str, Any],
    project_root: str | None = None,
) -> dict[str, Any]:
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    status = _git(project_root, "status", "--porcelain")
    diff = _git(project_root, "diff", "--binary", "HEAD")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": configuration_hash(config),
        "git": {
            "commit": _git(project_root, "rev-parse", "HEAD"),
            "branch": _git(project_root, "branch", "--show-current"),
            "dirty": bool(status),
            "diff_sha256": (
                hashlib.sha256(diff.encode("utf-8")).hexdigest()
                if diff is not None
                else None
            ),
            # Unlike git diff, this also covers non-ignored untracked source
            # files, which matters before a portfolio refactor is committed.
            "working_tree_sha256": _working_tree_sha256(project_root),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
