"""The workspace layout and the phase-1 frame validation.

Two roots, and nothing else (``data-scout-convention`` §8):

``DATA_SCOUT_CORPORA_ROOT``
    Data. One read-only mount of the whole corpora root, default ``/corpora``.
    data-scout is told a root and a corpus id and never learns a host path.

``DATA_SCOUT_WORKSPACE``
    State. Default ``/workspace``; a bind under ``/srv`` on ryzen so the nightly
    restic job picks it up. Per corpus::

        workspace/<corpus-id>/
          frame.md  notebooks/  scratch/  findings.jsonl  samples/
          cache/  staging/  report-YYYY-MM-DD.md  report-YYYY-MM-DD.html

**Fail loudly, never infer.** The single broad corpora mount cannot prove that a
session is scoped to the corpus it named, so this module checks instead of
inferring:

* a corpus id missing from the root raises :class:`CorpusNotFound` **naming the
  resolved root**, so a misconfigured mount is diagnosable from the error;
* a corpus without ``frame.md`` raises :class:`CorpusUnprofiled` rather than
  being profiled under an inferred id.

**Write no function here that opens the corpus for writing.** The only writable
tree is the workspace, and every helper in this module that creates anything
creates it under that tree.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CORPORA_ROOT_ENV = "DATA_SCOUT_CORPORA_ROOT"
WORKSPACE_ENV = "DATA_SCOUT_WORKSPACE"
CORPUS_ID_ENV = "DATA_SCOUT_CORPUS_ID"
SESSION_ENV = "DATA_SCOUT_SESSION"

DEFAULT_CORPORA_ROOT = "/corpora"
DEFAULT_WORKSPACE = "/workspace"

FRAME_FILENAME = "frame.md"
LEDGER_FILENAME = "findings.jsonl"

#: The per-corpus subdirectories, in the order the convention lists them.
WORKSPACE_SUBDIRS = ("notebooks", "scratch", "samples", "cache", "staging")

_CORPUS_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class WorkspaceError(RuntimeError):
    """Base class for workspace and frame errors."""


class CorpusNotFound(WorkspaceError):
    """The corpus id does not exist under the resolved corpora root."""


class CorpusUnprofiled(WorkspaceError):
    """The corpus exists but has no frame.md, so it has no corpus id."""


class NotScoped(WorkspaceError):
    """No corpus id was supplied and none is in the environment."""


def corpora_root() -> Path:
    """The read-only corpora root, from the environment, default ``/corpora``."""
    return Path(os.environ.get(CORPORA_ROOT_ENV) or DEFAULT_CORPORA_ROOT)


def workspace_root() -> Path:
    """The writable state root, from the environment, default ``/workspace``."""
    return Path(os.environ.get(WORKSPACE_ENV) or DEFAULT_WORKSPACE)


def validate_corpus_id(corpus_id: str) -> str:
    """Reject anything that is not a single, plain path component.

    A corpus id is one directory name. Anything containing a separator -- or a
    ``..`` -- is a scoping error, and it must not silently become a path.
    """
    if not isinstance(corpus_id, str) or not corpus_id:
        raise WorkspaceError(f"corpus id must be a non-empty string, got {corpus_id!r}")
    if not _CORPUS_ID_RE.match(corpus_id):
        raise WorkspaceError(
            f"invalid corpus id {corpus_id!r}: expected a single path component "
            "matching [A-Za-z0-9][A-Za-z0-9._-]*"
        )
    return corpus_id


def corpus_path(corpus_id: str) -> Path:
    """Resolve ``corpus_id`` under the corpora root, or fail naming the root."""
    validate_corpus_id(corpus_id)
    root = corpora_root()
    path = root / corpus_id
    if not path.is_dir():
        raise CorpusNotFound(
            f"corpus id {corpus_id!r} does not exist under the resolved corpora "
            f"root {root} (from {CORPORA_ROOT_ENV}). Check the mount, or the id: "
            f"the ids present are {sorted(_visible_dirs(root)) or 'none'}"
        )
    if not (path / FRAME_FILENAME).is_file():
        raise CorpusUnprofiled(
            f"corpus {corpus_id!r} at {path} has no {FRAME_FILENAME}: it is "
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
    """The corpus frame written in phase 1 (``data-scout-recipe`` phase 1).

    ``purpose`` is deliberately not required and not invented: purpose arrives
    late, and the frame's job is triage, not motive.
    """

    corpus_id: str
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


def read_frame(corpus_id: str) -> Frame:
    """Read and parse ``frame.md``; the corpus id in it must match ``corpus_id``.

    A mismatch is a scoping error and is refused rather than reconciled: two
    ids for one directory is exactly the "wrong number with a stamp" the ledger
    exists to prevent.
    """
    directory = corpus_path(corpus_id)
    frame_file = directory / FRAME_FILENAME
    raw = frame_file.read_text(encoding="utf-8", errors="replace")
    meta: dict[str, Any] = {}
    match = _FRONTMATTER_RE.match(raw)
    if match:
        meta = _parse_frontmatter(match.group(1))
    declared = meta.get("corpus_id") or meta.get("id")
    if not declared:
        heading = _HEADING_RE.search(raw)
        declared = heading.group(1).strip() if heading else None
    if not declared:
        raise CorpusUnprofiled(
            f"{frame_file} declares no corpus id: add `corpus_id:` to its "
            "frontmatter or a `# <corpus-id>` heading. The directory name is "
            "not the corpus id (fail loudly, never infer)."
        )
    declared = str(declared).strip()
    if declared != corpus_id:
        raise WorkspaceError(
            f"{frame_file} declares corpus id {declared!r} but the corpus was "
            f"addressed as {corpus_id!r}; refusing to profile under two ids"
        )
    lenses = meta.get("lenses") or ()
    if isinstance(lenses, str):
        lenses = (lenses,)
    ceiling = meta.get("tier_ceiling")
    return Frame(
        corpus_id=declared,
        path=directory,
        raw=raw,
        path_hint=meta.get("path"),
        lenses=tuple(str(item) for item in lenses),
        unit=meta.get("unit"),
        tier_ceiling=int(ceiling) if isinstance(ceiling, int) else None,
        created=meta.get("created"),
        extra={k: v for k, v in meta.items() if k not in {"corpus_id", "id", "lenses", "unit", "tier_ceiling", "created", "path"}},
    )


# --------------------------------------------------------------------------
# the workspace
# --------------------------------------------------------------------------


def workspace_dir(corpus_id: str) -> Path:
    """The per-corpus workspace directory. Not created here."""
    validate_corpus_id(corpus_id)
    return workspace_root() / corpus_id


def ensure_workspace(corpus_id: str) -> Path:
    """Create the workspace tree for ``corpus_id`` if needed and return it.

    Idempotent, because half-done is the normal case: re-running an order must
    never fail on state that already exists.
    """
    directory = workspace_dir(corpus_id)
    directory.mkdir(parents=True, exist_ok=True)
    for name in WORKSPACE_SUBDIRS:
        (directory / name).mkdir(exist_ok=True)
    return directory


def notebook_dir(corpus_id: str) -> Path:
    return workspace_dir(corpus_id) / "notebooks"


def scratch_dir(corpus_id: str) -> Path:
    return workspace_dir(corpus_id) / "scratch"


def samples_dir(corpus_id: str) -> Path:
    return workspace_dir(corpus_id) / "samples"


def staging_dir(corpus_id: str) -> Path:
    return workspace_dir(corpus_id) / "staging"


def cache_dir(corpus_id: str | None = None) -> Path:
    """The observation cache for a corpus.

    ``rm -rf cache/`` must remain the entire recovery procedure, so this is
    always a plain directory under the workspace.
    """
    if corpus_id is None:
        corpus_id = try_current_corpus_id()
    if corpus_id is None:
        return workspace_root() / ".cache"
    return workspace_dir(corpus_id) / "cache"


def ledger_path(corpus_id: str | None = None) -> Path:
    """The append-only ledger path for a corpus."""
    if corpus_id is None:
        corpus_id = current_corpus_id()
    return workspace_dir(corpus_id) / LEDGER_FILENAME


def try_current_corpus_id() -> str | None:
    value = os.environ.get(CORPUS_ID_ENV)
    return value.strip() or None if value else None


def current_corpus_id() -> str:
    """The corpus this process is scoped to, from ``DATA_SCOUT_CORPUS_ID``."""
    corpus_id = try_current_corpus_id()
    if corpus_id is None:
        raise NotScoped(
            f"{CORPUS_ID_ENV} is not set, so the ledger and cache paths are "
            "ambiguous. Pass corpus_id explicitly, or set the environment "
            "variable (report-render sets it for the kernel)."
        )
    return validate_corpus_id(corpus_id)


def current_session() -> int | None:
    value = os.environ.get(SESSION_ENV)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        raise WorkspaceError(f"{SESSION_ENV}={value!r} is not an integer") from None
