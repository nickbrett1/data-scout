"""``file_census`` — magic/size/mtime histogram over a tree. Tier 0, exhaustive.

The first instrument of the tier-0 sweep (``data-scout-recipe`` phase 2). It
answers "what files have I got" without opening any of them, so a source that is
already closed (empty, wrong format, a hundred copies of the same JPEG) is
closed here for free.

``file`` decides the type, not the extension: a corpus of ``.pdf`` files that
are really scans, or ``.txt`` files that are really PostScript, is a normal
finding and the extension cannot see it.
"""

from __future__ import annotations

import os
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from data_scout.tools._common import (
    capped,
    iter_files,
    measurement,
    require_binary,
    share,
)

_BATCH = 200
_MAX_DETAIL = 25


def _mime_types(paths: list[Path]) -> dict[Path, str]:
    """Run ``file --mime-type`` over ``paths`` in batches, one path per argument."""
    binary = require_binary("file_census", "file", package="file (with libmagic1)")
    found: dict[Path, str] = {}
    for start in range(0, len(paths), _BATCH):
        batch = paths[start : start + _BATCH]
        try:
            completed = subprocess.run(  # noqa: S603 - argv list, no shell
                [binary, "-b", "--mime-type", "--", *[str(item) for item in batch]],
                capture_output=True,
                text=True,
                check=False,
                timeout=300,
            )
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
            raise RuntimeError(f"file_census: `file` failed on {len(batch)} paths: {exc}") from exc
        lines = completed.stdout.splitlines()
        for item, mime in zip(batch, lines, strict=False):
            found[item] = mime.strip() or "application/octet-stream"
    return found


def file_census(
    path: str | os.PathLike[str],
    *,
    max_files: int | None = None,
    limit_details: bool = True,
) -> dict[str, Any]:
    """Census a file or a tree.

    ``max_files`` caps the walk for a corpus too large to census exhaustively;
    ``n`` reports how many were actually examined, so a capped census is never
    mistaken for a complete one.
    """
    files = list(iter_files(path, limit=max_files))
    if not files:
        return measurement(
            value=0.0,
            n=0,
            unit_notes="share of files in the largest mime bucket",
            files=0,
            bytes_total=0,
            by_mime={},
            by_extension={},
            size_bytes={},
            zero_byte=0,
            unreadable=0,
            earliest_mtime=None,
            latest_mtime=None,
        )

    by_mime: Counter[str] = Counter()
    by_extension: Counter[str] = Counter()
    sizes: list[int] = []
    mtimes: list[float] = []
    zero_byte = 0
    unreadable: list[str] = []

    for item in files:
        try:
            stat = item.stat()
        except OSError:
            unreadable.append(str(item))
            continue
        sizes.append(stat.st_size)
        mtimes.append(stat.st_mtime)
        if stat.st_size == 0:
            zero_byte += 1
        by_extension[(item.suffix.lower() or "<none>")] += 1

    mimes = _mime_types([item for item in files if str(item) not in set(unreadable)])
    by_mime.update(mimes.values())

    sizes.sort()
    top_mime, top_count = by_mime.most_common(1)[0] if by_mime else ("<unknown>", 0)
    mime_detail, mime_trimmed = capped(by_mime.most_common(), _MAX_DETAIL) if limit_details else (by_mime.most_common(), False)
    ext_detail, ext_trimmed = capped(by_extension.most_common(), _MAX_DETAIL) if limit_details else (by_extension.most_common(), False)

    return measurement(
        value=share(top_count, sum(by_mime.values())),
        n=len(sizes),
        unit_notes="share of files in the largest mime bucket",
        files=len(sizes),
        bytes_total=sum(sizes),
        dominant_mime=top_mime,
        by_mime=dict(mime_detail),
        by_mime_total=len(by_mime),
        by_mime_trimmed=mime_trimmed,
        by_extension=dict(ext_detail),
        by_extension_total=len(by_extension),
        by_extension_trimmed=ext_trimmed,
        size_bytes={
            "min": sizes[0],
            "median": int(statistics.median(sizes)),
            "mean": int(statistics.fmean(sizes)),
            "p90": sizes[min(len(sizes) - 1, int(len(sizes) * 0.9))],
            "max": sizes[-1],
        },
        zero_byte=zero_byte,
        unreadable=len(unreadable),
        unreadable_examples=unreadable[:5],
        earliest_mtime=_iso(min(mtimes)) if mtimes else None,
        latest_mtime=_iso(max(mtimes)) if mtimes else None,
        truncated_walk=max_files is not None and len(files) >= max_files,
    )


def _iso(timestamp: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")
