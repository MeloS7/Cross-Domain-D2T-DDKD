"""Filesystem helpers shared by evaluation scripts."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_case_insensitive_path(path: str | os.PathLike[str]) -> str | None:
    """Resolve a path despite case-only filename differences.

    Exact matches are preferred. If a component differs only by case, the
    unique matching directory entry is returned. Ambiguous matches raise
    instead of silently selecting one.
    """
    requested = os.path.abspath(os.fspath(path))
    drive, tail = os.path.splitdrive(requested)
    current = drive + os.sep if tail.startswith(os.sep) else (drive or ".")
    for part in (segment for segment in tail.split(os.sep) if segment):
        exact = os.path.join(current, part)
        if os.path.exists(exact):
            current = exact
            continue
        if not os.path.isdir(current):
            return None
        matches = [
            os.path.join(current, entry)
            for entry in os.listdir(current)
            if entry.casefold() == part.casefold()
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous case-insensitive path component '{part}' in '{current}': "
                f"{matches}"
            )
        if not matches:
            return None
        current = matches[0]
    return current
