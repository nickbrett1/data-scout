"""What is available to scout: the datasets under the read-only root.

The reports index answers "what has been measured". This module answers the
question before that one — **what could be measured** — without opening any
dataset for writing and without inferring anything from a name alone.

Two facts shape the scan:

* **The mount can be enormous.** A dataset may hold millions of files, and the
  index page is rendered on every request, so the walk is *bounded*: it stops
  after :data:`MAX_SCAN_ENTRIES` entries and says so (``truncated: true``).
  A partial count that announces itself is honest; a five-minute page load is
  not.
* **Not every directory is a dataset.** Hidden directories are skipped, and the
  per-dataset subdirectories the workspace writes are never under this root, so
  no extra filtering is invented here.

The summary is display state, not a measurement: it never lands in
``findings.jsonl``. Counts, bytes and the extension mix are cheap facts for
triage; the instruments still decide what any of it means.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data_scout.workspace import (
    FRAME_FILENAME,
    WorkspaceError,
    datasets_root,
    validate_dataset_id,
)

#: Stop the walk after this many entries (files + directories). Enough for a
#: real triage of a large dataset, small enough that the index page stays a
#: page rather than a job.
MAX_SCAN_ENTRIES = 50_000

#: How many extensions the summary carries. The tail is long and uninformative.
TOP_EXTENSIONS = 5

_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_FRAME_TITLE_BYTES = 4096


@dataclass(frozen=True)
class DatasetSummary:
    """One dataset directory, as the index needs to show it."""

    dataset_id: str
    path: Path
    files: int = 0
    directories: int = 0
    bytes: int = 0
    extensions: tuple[tuple[str, int], ...] = ()
    newest: str | None = None
    frame: bool = False
    title: str | None = None
    truncated: bool = False
    error: str | None = None

    @property
    def name(self) -> str:
        """The frame's heading when it has one, else the dataset id.

        The id is the identity; the heading is a label. Never the reverse — a
        title must not become a second id.
        """
        return self.title or self.dataset_id

    def as_record(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "files": self.files,
            "directories": self.directories,
            "bytes": self.bytes,
            "extensions": dict(self.extensions),
            "newest": self.newest,
            "frame": self.frame,
            "truncated": self.truncated,
            "error": self.error,
            "in_root": True,
        }


@dataclass(frozen=True)
class WorkspaceOnlyDataset:
    """A workspace directory with no counterpart under the root.

    The mount may be unavailable or the dataset moved, but the frozen artifacts
    still exist under the workspace. Hiding them would make a report disappear
    because a mount was down, so they are listed and the absence is stated.
    """

    dataset_id: str

    def as_record(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.dataset_id,
            "files": None,
            "directories": None,
            "bytes": None,
            "extensions": {},
            "newest": None,
            "frame": False,
            "truncated": False,
            "error": None,
            "in_root": False,
        }


def summarize(dataset_id: str, path: Path, *, max_entries: int = MAX_SCAN_ENTRIES) -> DatasetSummary:
    """Count one dataset without following symlinks and without writing.

    An unreadable subdirectory is skipped rather than fatal: one permission
    error must not blank the whole inventory. If the dataset root itself is
    unreadable, ``error`` carries the reason and the counts stay zero.
    """
    validate_dataset_id(dataset_id)
    files = directories = total = 0
    extensions: Counter[str] = Counter()
    newest: float | None = None
    truncated = False
    error: str | None = None

    stack: list[Path] = [path]
    while stack and not truncated:
        current = stack.pop()
        try:
            entries = os.scandir(current)
        except OSError as exc:
            if current == path:
                error = str(exc)
            continue
        with entries:
            for entry in entries:
                if files + directories >= max_entries:
                    truncated = True
                    break
                try:
                    if entry.is_dir(follow_symlinks=False):
                        directories += 1
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        stat = entry.stat(follow_symlinks=False)
                        files += 1
                        total += stat.st_size
                        newest = stat.st_mtime if newest is None else max(newest, stat.st_mtime)
                        suffix = Path(entry.name).suffix.lower() or "(none)"
                        extensions[suffix] += 1
                except OSError:
                    # A file can vanish or be unreadable mid-walk; the count is
                    # a triage fact, not a record, so skipping it is honest.
                    continue

    frame_file = path / FRAME_FILENAME
    return DatasetSummary(
        dataset_id=dataset_id,
        path=path,
        files=files,
        directories=directories,
        bytes=total,
        extensions=tuple(extensions.most_common(TOP_EXTENSIONS)),
        newest=_iso_date(newest),
        frame=frame_file.is_file(),
        title=_frame_title(frame_file),
        truncated=truncated,
        error=error,
    )


def discover(
    root: Path | None = None, *, max_entries: int = MAX_SCAN_ENTRIES
) -> tuple[list[DatasetSummary], str | None]:
    """Every visible dataset under the root, sorted by id.

    Returns ``(datasets, error)`` where ``error`` names an unreadable root. An
    empty list with no error is a real answer: the mount works and holds
    nothing yet.
    """
    root = root or datasets_root()
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        return [], f"{root} is not readable: {exc}"
    datasets: list[DatasetSummary] = []
    for entry in sorted(entries, key=lambda item: item.name):
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        try:
            validate_dataset_id(entry.name)
        except WorkspaceError:
            continue
        datasets.append(summarize(entry.name, entry, max_entries=max_entries))
    return datasets, None


def _frame_title(frame_file: Path) -> str | None:
    try:
        with frame_file.open("rb") as handle:
            text = handle.read(_FRAME_TITLE_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return None
    match = _HEADING_RE.search(text)
    return match.group(1).strip() if match else None


def _iso_date(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=UTC).date().isoformat()


def format_bytes(size: int | None) -> str:
    """A short human-readable size. ``None`` means "not on this mount"."""
    if size is None:
        return "—"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def format_extensions(extensions: Any) -> str:
    """``{'.pdf': 3, '.txt': 1}`` -> ``"pdf 3, txt 1"``."""
    items = extensions.items() if isinstance(extensions, dict) else extensions
    return ", ".join(f"{str(ext).lstrip('.')} {count}" for ext, count in items)
