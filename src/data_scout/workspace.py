"""The workspace layout and the phase-1 frame validation.

Two roots, and nothing else (``data-scout-convention`` §8):

``DATA_SCOUT_DATASETS_ROOT``
    Data. One read-only mount of the whole datasets root, default ``/datasets``.
    data-scout is told a root and a dataset id and never learns a host path.

``DATA_SCOUT_WORKSPACE``
    State. Default ``/workspace``; a bind under ``/srv`` on ryzen so the nightly
    restic job picks it up. Per dataset::

        workspace/<dataset-id>/
          frame.md  notebooks/  scratch/  findings.jsonl  samples/
          cache/  staging/  report-YYYY-MM-DD.md  report-YYYY-MM-DD.html

**Fail loudly, never infer.** The single broad dataset mount cannot prove that a
session is scoped to the dataset it named, so this module checks instead of
inferring:

* a dataset id missing from the root raises :class:`DatasetNotFound` **naming the
  resolved root**, so a misconfigured mount is diagnosable from the error;
* a dataset without ``frame.md`` raises :class:`DatasetUnprofiled` rather than
  being profiled under an inferred id.

**Write no function here that opens the dataset for writing.** The only writable
tree is the workspace, and every helper in this module that creates anything
creates it under that tree.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATASETS_ROOT_ENV = "DATA_SCOUT_DATASETS_ROOT"
WORKSPACE_ENV = "DATA_SCOUT_WORKSPACE"
DATASET_ID_ENV = "DATA_SCOUT_DATASET_ID"
SESSION_ENV = "DATA_SCOUT_SESSION"

DEFAULT_DATASETS_ROOT = "/datasets"
DEFAULT_WORKSPACE = "/workspace"

FRAME_FILENAME = "frame.md"
LEDGER_FILENAME = "findings.jsonl"

#: The per-dataset subdirectories, in the order the convention lists them.
WORKSPACE_SUBDIRS = ("notebooks", "scratch", "samples", "cache", "staging")

_DATASET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class WorkspaceError(RuntimeError):
    """Base class for workspace and frame errors."""


class DatasetNotFound(WorkspaceError):
    """The dataset id does not exist under the resolved datasets root."""


class DatasetUnprofiled(WorkspaceError):
    """The dataset exists but has no frame.md, so it has no dataset id."""


class NotScoped(WorkspaceError):
    """No dataset id was supplied and none is in the environment."""


def datasets_root() -> Path:
    """The read-only datasets root, from the environment, default ``/datasets``."""
    return Path(os.environ.get(DATASETS_ROOT_ENV) or DEFAULT_DATASETS_ROOT)


def workspace_root() -> Path:
    """The writable state root, from the environment, default ``/workspace``."""
    return Path(os.environ.get(WORKSPACE_ENV) or DEFAULT_WORKSPACE)


def validate_dataset_id(dataset_id: str) -> str:
    """Reject anything that is not a single, plain path component.

    A dataset id is one directory name. Anything containing a separator -- or a
    ``..`` -- is a scoping error, and it must not silently become a path.
    """
    if not isinstance(dataset_id, str) or not dataset_id:
        raise WorkspaceError(f"dataset id must be a non-empty string, got {dataset_id!r}")
    if not _DATASET_ID_RE.match(dataset_id):
        raise WorkspaceError(
            f"invalid dataset id {dataset_id!r}: expected a single path component "
            "matching [A-Za-z0-9][A-Za-z0-9._-]*"
        )
    return dataset_id


def dataset_path(dataset_id: str) -> Path:
    """Resolve ``dataset_id`` under the datasets root, or fail naming the root."""
    validate_dataset_id(dataset_id)
    root = datasets_root()
    path = root / dataset_id
    if not path.is_dir():
        raise DatasetNotFound(
            f"dataset id {dataset_id!r} does not exist under the resolved datasets "
            f"root {root} (from {DATASETS_ROOT_ENV}). Check the mount, or the id: "
            f"the ids present are {sorted(_visible_dirs(root)) or 'none'}"
        )
    if not (path / FRAME_FILENAME).is_file():
        raise DatasetUnprofiled(
            f"dataset {dataset_id!r} at {path} has no {FRAME_FILENAME}: it is "
            "unprofiled and must not be profiled under an inferred id"
        )
    return path


def _visible_dirs(root: Path) -> list[str]:
    try:
        return [entry.name for entry in root.iterdir() if entry.is_dir()]
    except OSError:
        return []


# --------------------------------------------------------------------------
# frame.md
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """The dataset frame written in phase 1 (``data-scout-recipe`` phase 1).

    ``purpose`` is deliberately not required and not invented: purpose arrives
    late, and the frame's job is triage, not motive.
    """

    dataset_id: str
    path: Path
    raw: str
    path_hint: str | None = None
    lenses: tuple[str, ...] = ()
    unit: str | None = None
    tier_ceiling: int | None = None
    created: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def _parse_scalar(value: str) -> Any:
    text = value.strip().strip('"').strip("'")
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return text


def _parse_frontmatter(block: str) -> dict[str, Any]:
    """A deliberately tiny YAML subset: ``key: value`` and ``- item`` lists.

    No PyYAML: a frame is written by the recipe, not by a stranger, and an
    unparseable frame should fail loudly rather than half-load through a parser
    with more grammar than the format needs.
    """
    data: dict[str, Any] = {}
    current: str | None = None
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.lstrip().startswith("- ") and current is not None:
            data.setdefault(current, [])
            if isinstance(data[current], list):
                data[current].append(_parse_scalar(line.lstrip()[2:]))
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            current = key
            if value.strip():
                data[key] = _parse_scalar(value)
            else:
                data[key] = []
    return data


def read_frame(dataset_id: str) -> Frame:
    """Read and parse ``frame.md``; the dataset id in it must match ``dataset_id``.

    A mismatch is a scoping error and is refused rather than reconciled: two
    ids for one directory is exactly the "wrong number with a stamp" the ledger
    exists to prevent.
    """
    directory = dataset_path(dataset_id)
    frame_file = directory / FRAME_FILENAME
    raw = frame_file.read_text(encoding="utf-8", errors="replace")
    meta: dict[str, Any] = {}
    match = _FRONTMATTER_RE.match(raw)
    if match:
        meta = _parse_frontmatter(match.group(1))
    declared = meta.get("dataset_id") or meta.get("id")
    if not declared:
        heading = _HEADING_RE.search(raw)
        declared = heading.group(1).strip() if heading else None
    if not declared:
        raise DatasetUnprofiled(
            f"{frame_file} declares no dataset id: add `dataset_id:` to its "
            "frontmatter or a `# <dataset-id>` heading. The directory name is "
            "not the dataset id (fail loudly, never infer)."
        )
    declared = str(declared).strip()
    if declared != dataset_id:
        raise WorkspaceError(
            f"{frame_file} declares dataset id {declared!r} but the dataset was "
            f"addressed as {dataset_id!r}; refusing to profile under two ids"
        )
    lenses = meta.get("lenses") or ()
    if isinstance(lenses, str):
        lenses = (lenses,)
    ceiling = meta.get("tier_ceiling")
    return Frame(
        dataset_id=declared,
        path=directory,
        raw=raw,
        path_hint=meta.get("path"),
        lenses=tuple(str(item) for item in lenses),
        unit=meta.get("unit"),
        tier_ceiling=int(ceiling) if isinstance(ceiling, int) else None,
        created=meta.get("created"),
        extra={k: v for k, v in meta.items() if k not in {"dataset_id", "id", "lenses", "unit", "tier_ceiling", "created", "path"}},
    )


# --------------------------------------------------------------------------
# the workspace
# --------------------------------------------------------------------------


def workspace_dir(dataset_id: str) -> Path:
    """The per-dataset workspace directory. Not created here."""
    validate_dataset_id(dataset_id)
    return workspace_root() / dataset_id


def ensure_workspace(dataset_id: str) -> Path:
    """Create the workspace tree for ``dataset_id`` if needed and return it.

    Idempotent, because half-done is the normal case: re-running an order must
    never fail on state that already exists.
    """
    directory = workspace_dir(dataset_id)
    directory.mkdir(parents=True, exist_ok=True)
    for name in WORKSPACE_SUBDIRS:
        (directory / name).mkdir(exist_ok=True)
    return directory


def notebook_dir(dataset_id: str) -> Path:
    return workspace_dir(dataset_id) / "notebooks"


def scratch_dir(dataset_id: str) -> Path:
    return workspace_dir(dataset_id) / "scratch"


def samples_dir(dataset_id: str) -> Path:
    return workspace_dir(dataset_id) / "samples"


def staging_dir(dataset_id: str) -> Path:
    return workspace_dir(dataset_id) / "staging"


def cache_dir(dataset_id: str | None = None) -> Path:
    """The observation cache for a dataset.

    ``rm -rf cache/`` must remain the entire recovery procedure, so this is
    always a plain directory under the workspace.
    """
    if dataset_id is None:
        dataset_id = try_current_dataset_id()
    if dataset_id is None:
        return workspace_root() / ".cache"
    return workspace_dir(dataset_id) / "cache"


def ledger_path(dataset_id: str | None = None) -> Path:
    """The append-only ledger path for a dataset."""
    if dataset_id is None:
        dataset_id = current_dataset_id()
    return workspace_dir(dataset_id) / LEDGER_FILENAME


def try_current_dataset_id() -> str | None:
    value = os.environ.get(DATASET_ID_ENV)
    return value.strip() or None if value else None


def current_dataset_id() -> str:
    """The dataset this process is scoped to, from ``DATA_SCOUT_DATASET_ID``."""
    dataset_id = try_current_dataset_id()
    if dataset_id is None:
        raise NotScoped(
            f"{DATASET_ID_ENV} is not set, so the ledger and cache paths are "
            "ambiguous. Pass dataset_id explicitly, or set the environment "
            "variable (report-render sets it for the kernel)."
        )
    return validate_dataset_id(dataset_id)


def current_session() -> int | None:
    value = os.environ.get(SESSION_ENV)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        raise WorkspaceError(f"{SESSION_ENV}={value!r} is not an integer") from None
