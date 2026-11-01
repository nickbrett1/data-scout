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

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATASETS_ROOT_ENV = "DATA_SCOUT_DATASETS_ROOT"
WORKSPACE_ENV = "DATA_SCOUT_WORKSPACE"
DATASET_ID_ENV = "DATA_SCOUT_DATASET_ID"
SESSION_ENV = "DATA_SCOUT_SESSION"

#: Redirect the ledger to a staging file for the duration of one kernel.
#:
#: ``report-render`` sets this for the artifact execution so a notebook's
#: appends land somewhere that can be discarded: the live ``findings.jsonl`` is
#: only touched after both gates pass. It is not a general-purpose knob and no
#: other component sets it.
LEDGER_OVERRIDE_ENV = "DATA_SCOUT_LEDGER"

DEFAULT_DATASETS_ROOT = "/datasets"
DEFAULT_WORKSPACE = "/workspace"

FRAME_FILENAME = "frame.md"
LEDGER_FILENAME = "findings.jsonl"

#: The per-dataset subdirectories, in the order the convention lists them.
WORKSPACE_SUBDIRS = ("notebooks", "scratch", "samples", "cache", "staging")

_DATASET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: A rendered artifact is one of two things, and they do **not** share a
#: directory: the deliverable (``report-*``) sits at the dataset root beside
#: ``frame.md``/``findings.jsonl``, while the evidence appendix (``session-*``)
#: sits with its notebook under ``notebooks/`` (convention §1, §4.1). The
#: pattern lives here, beside :func:`artifact_path`, so the writer and the HTTP
#: reader share one definition of "servable" as well as one definition of where.
SERVABLE_ARTIFACT_RE = re.compile(r"^(report|session)-[A-Za-z0-9._-]*\.html$")


class WorkspaceError(RuntimeError):
    """Base class for workspace and frame errors."""


class DatasetNotFound(WorkspaceError):
    """The dataset id does not exist under the resolved datasets root."""


class DatasetUnprofiled(WorkspaceError):
    """The dataset exists but has no frame.md, so it has no dataset id."""


class NotScoped(WorkspaceError):
    """No dataset id was supplied and none is in the environment."""


class PinUnreadable(WorkspaceError):
    """A pin file exists but cannot be understood. Not the same fact as no pin.

    An absent pin is ``None`` — a dataset may be unversioned by owner decision.
    A pin that is present and malformed, missing a key, of the wrong type,
    written under an unknown ``pin_schema``, or naming a different dataset is
    this error. Conflating the two is how a corrupt store silently reads as an
    unpinned one.
    """


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
    """Resolve ``dataset_id`` under the datasets root, or fail naming the root.

    This answers exactly one question — *does this dataset exist on the mount* —
    and deliberately says nothing about frames. The frame is state and lives in
    the workspace (:func:`frame_path`); a path resolver that also asserted where
    a frame must be is what let two modules answer "where is the frame" with two
    different rules.
    """
    validate_dataset_id(dataset_id)
    root = datasets_root()
    path = root / dataset_id
    if not path.is_dir():
        raise DatasetNotFound(
            f"dataset id {dataset_id!r} does not exist under the resolved datasets "
            f"root {root} (from {DATASETS_ROOT_ENV}). Check the mount, or the id: "
            f"the ids present are {sorted(_visible_dirs(root)) or 'none'}"
        )
    return path


def _visible_dirs(root: Path) -> list[str]:
    try:
        return [entry.name for entry in root.iterdir() if entry.is_dir()]
    except OSError:
        return []


# --------------------------------------------------------------------------
# pin.json — the host's claim about the payload, read only here
# --------------------------------------------------------------------------

#: The pin file's name, at ``<datasets_root>/<dataset-id>/pin.json``. Written by
#: the host (``/srv/bin/datasets pin``); never by anything in this image.
PIN_FILENAME = "pin.json"

#: The only pin format this reader understands. Unknown → :class:`PinUnreadable`.
PIN_SCHEMA = 1


@dataclass(frozen=True)
class DatasetPin:
    """A pin read from disk: which bytes the host says this dataset is.

    ``dataset_files`` / ``dataset_bytes`` are load-bearing, not decoration: the
    frame measurement cross-checks them so the image never records a pin it did
    not verify against the payload it just measured. ``tag``, ``dvc_version``
    and ``written_at`` are provenance for a human; nothing reads them
    programmatically.
    """

    dataset_id: str
    dataset_commit: str
    dataset_manifest_hash: str
    dataset_files: int
    dataset_bytes: int
    tag: str | None = None
    dvc_version: str | None = None
    written_at: str | None = None


def pin_path(dataset_id: str) -> Path:
    """``<datasets_root>/<dataset_id>/pin.json``. Beside :func:`dataset_path`.

    Unlike :func:`dataset_path`, this does **not** require the dataset to exist:
    a pin check is meaningful precisely when a mount is misconfigured, and
    "there is no pin at this path" must be answerable without first proving the
    dataset is present.
    """
    return datasets_root() / validate_dataset_id(dataset_id) / PIN_FILENAME


def read_pin(dataset_id: str, root: Path | None = None) -> DatasetPin | None:
    """The pin recorded for this dataset, or ``None`` if none was ever recorded.

    ``root`` defaults to :func:`datasets_root`. ``measure_frame`` passes its own
    root so a fixture pin is visible beside a fixture payload; that is the only
    reason the parameter exists. Reading is one file in one direction: no git,
    no subprocess, no DVC.

    Rules, each of which is a test: an absent file is ``None`` (an unversioned
    dataset is legitimate); a present file that is malformed, missing a required
    key, of the wrong type, under an unknown ``pin_schema``, or naming a
    different ``dataset_id`` raises :class:`PinUnreadable` naming the path and
    the problem. This function never writes.
    """
    dataset_id = validate_dataset_id(dataset_id)
    base = Path(root) if root is not None else datasets_root()
    path = base / dataset_id / PIN_FILENAME
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PinUnreadable(f"{path}: pin file could not be read ({exc})") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PinUnreadable(f"{path}: pin file is not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise PinUnreadable(
            f"{path}: pin file must be a JSON object, got {type(data).__name__}"
        )

    schema = data.get("pin_schema")
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise PinUnreadable(
            f"{path}: pin_schema must be an integer, got {schema!r}"
        )
    if schema != PIN_SCHEMA:
        raise PinUnreadable(
            f"{path}: pin_schema {schema} is not one this reader understands "
            f"(known: {PIN_SCHEMA})"
        )

    required = (
        "dataset_id",
        "dataset_commit",
        "dataset_manifest_hash",
        "dataset_files",
        "dataset_bytes",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise PinUnreadable(
            f"{path}: pin file is missing required keys: {', '.join(missing)}"
        )

    for key in ("dataset_id", "dataset_commit", "dataset_manifest_hash"):
        if not isinstance(data[key], str) or not data[key]:
            raise PinUnreadable(
                f"{path}: {key} must be a non-empty string, got {data[key]!r}"
            )
    for key in ("dataset_files", "dataset_bytes"):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise PinUnreadable(
                f"{path}: {key} must be an integer, got {value!r}"
            )
    for key in ("tag", "dvc_version", "written_at"):
        value = data.get(key)
        if value is not None and not isinstance(value, str):
            raise PinUnreadable(
                f"{path}: {key} must be a string or null, got {value!r}"
            )

    if data["dataset_id"] != dataset_id:
        raise PinUnreadable(
            f"{path}: pin names dataset {data['dataset_id']!r} but the dataset "
            f"was addressed as {dataset_id!r}; the pin describes different bytes"
        )

    return DatasetPin(
        dataset_id=data["dataset_id"],
        dataset_commit=data["dataset_commit"],
        dataset_manifest_hash=data["dataset_manifest_hash"],
        dataset_files=data["dataset_files"],
        dataset_bytes=data["dataset_bytes"],
        tag=data.get("tag"),
        dvc_version=data.get("dvc_version"),
        written_at=data.get("written_at"),
    )


# --------------------------------------------------------------------------
# frame.md
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameIdentity:
    """One measured scope recorded in a v2 frame: scope, count, bytes, manifest."""

    scope: str
    files: int
    bytes: int
    manifest: str


@dataclass(frozen=True)
class FramePin:
    """The pin as a v3 frame records it, read back without a datasets mount.

    Deliberately **not** :class:`DatasetPin`: reading a frame is a workspace
    read, and must not require the read-only datasets root to be mounted. A
    frame written for an unversioned dataset records ``pinned=False`` with null
    fields — "unpinned" is a record, not a gap.
    """

    pinned: bool
    dataset_commit: str | None = None
    dataset_manifest_hash: str | None = None
    dataset_files: int | None = None
    dataset_bytes: int | None = None


@dataclass(frozen=True)
class Frame:
    """The dataset frame written in phase 1 (``data-scout-recipe`` phase 1).

    ``purpose`` is deliberately not required and not invented: purpose arrives
    late, and the frame's job is triage, not motive.

    ``schema_version`` is ``1`` when the frame carries no ``frame_schema`` key
    (every phase-1 frame, including the real one on disk). A v2 frame records a
    payload ``identity`` and a ``checkout`` figure separately, plus the named
    exclusions in force. The two are not interchangeable, which is why
    ``frame.compare_frames`` refuses to diff across versions.
    """

    dataset_id: str
    path: Path
    raw: str
    path_hint: str | None = None
    lenses: tuple[str, ...] = ()
    unit: str | None = None
    tier_ceiling: int | None = None
    created: str | None = None
    schema_version: int = 1
    exclusions: tuple[str, ...] = ()
    identity: FrameIdentity | None = None
    checkout: FrameIdentity | None = None
    pin: FramePin | None = None
    extra: dict[str, Any] = field(default_factory=dict)


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def _parse_scalar(value: str) -> Any:
    """One frontmatter scalar: bool, int, null, or the string as written.

    ``null``, ``~`` and an empty value are YAML's three spellings of *absent*,
    and they parse to ``None``. They are not the strings ``"null"`` / ``"~"``:
    a frame renders an unpinned dataset's fields as literal ``null`` (§3.2), and
    reading that back as a string would turn "there is no pin" into a pin whose
    commit is the word "null".
    """
    text = value.strip().strip('"').strip("'")
    if not text or text.lower() in {"null", "~"}:
        return None
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
    frame_file = frame_path(dataset_id)
    if not frame_file.is_file():
        raise DatasetUnprofiled(
            f"dataset {dataset_id!r} has no {FRAME_FILENAME} at {frame_file}: it is "
            "unprofiled and must not be profiled under an inferred id"
        )
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

    from data_scout.frame import SUPPORTED_FRAME_SCHEMAS, FrameSchemaError

    # An absent key is version 1: every frame written before the schema carried
    # a version. It is not "unknown" — the rule it was measured under is known.
    schema_version = meta.get("frame_schema", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise FrameSchemaError(
            f"{frame_file} declares frame_schema {schema_version!r}, which is not "
            f"an integer. Known versions: {SUPPORTED_FRAME_SCHEMAS}."
        )
    if schema_version not in SUPPORTED_FRAME_SCHEMAS:
        raise FrameSchemaError(
            f"{frame_file} was written under frame schema v{schema_version}; this "
            f"reader understands {SUPPORTED_FRAME_SCHEMAS}. Refusing to read a "
            "frame under rules it was not measured with."
        )

    declared_exclusions = meta.get("frame_exclusions") or ()
    if isinstance(declared_exclusions, str):
        declared_exclusions = (declared_exclusions,)

    known = {
        "dataset_id",
        "id",
        "lenses",
        "unit",
        "tier_ceiling",
        "created",
        "path",
        "frame_schema",
        "frame_exclusions",
        "identity_scope",
        "identity_files",
        "identity_bytes",
        "identity_manifest",
        "checkout_scope",
        "checkout_files",
        "checkout_bytes",
        "checkout_manifest",
        "manifest_rule",
        "dataset_pinned",
        "dataset_commit",
        "dataset_manifest_hash",
        "dataset_files",
        "dataset_bytes",
    }
    return Frame(
        dataset_id=declared,
        path=frame_file.parent,
        raw=raw,
        path_hint=meta.get("path"),
        lenses=tuple(str(item) for item in lenses),
        unit=meta.get("unit"),
        tier_ceiling=int(ceiling) if isinstance(ceiling, int) else None,
        created=meta.get("created"),
        schema_version=schema_version,
        exclusions=tuple(str(item) for item in declared_exclusions),
        identity=_frame_identity(meta, "identity"),
        checkout=_frame_identity(meta, "checkout"),
        pin=_frame_pin(meta, frame_file),
        extra={k: v for k, v in meta.items() if k not in known},
    )


def _frame_identity(meta: dict[str, Any], prefix: str) -> FrameIdentity | None:
    """Read one ``<prefix>_{scope,files,bytes,manifest}`` block, or ``None``."""
    keys = (f"{prefix}_scope", f"{prefix}_files", f"{prefix}_bytes", f"{prefix}_manifest")
    if not any(key in meta for key in keys):
        return None
    files = meta.get(f"{prefix}_files")
    size = meta.get(f"{prefix}_bytes")
    return FrameIdentity(
        scope=str(meta.get(f"{prefix}_scope") or ""),
        files=files if isinstance(files, int) and not isinstance(files, bool) else 0,
        bytes=size if isinstance(size, int) and not isinstance(size, bool) else 0,
        manifest=str(meta.get(f"{prefix}_manifest") or ""),
    )


def _frame_pin(meta: dict[str, Any], frame_file: Path) -> FramePin | None:
    """Read the v3 frame's pin block, or ``None`` for a frame that has none.

    A v1/v2 frame carries no pin keys, and that is a real difference: it was
    measured before the pin existed. A v3 frame **always** writes
    ``dataset_pinned`` so that "unpinned" is a record rather than a gap.
    """
    if "dataset_pinned" not in meta:
        return None
    pinned = meta.get("dataset_pinned")
    if not isinstance(pinned, bool):
        raise WorkspaceError(
            f"{frame_file} declares dataset_pinned {pinned!r}, which is not a "
            "boolean; a frame cannot say half of whether it was pinned"
        )
    commit = meta.get("dataset_commit")
    manifest = meta.get("dataset_manifest_hash")
    files = meta.get("dataset_files")
    size = meta.get("dataset_bytes")
    for key, value in (
        ("dataset_commit", commit),
        ("dataset_manifest_hash", manifest),
    ):
        if value is not None and not isinstance(value, str):
            raise WorkspaceError(
                f"{frame_file} declares {key} {value!r}, which is not a string or null"
            )
    for key, value in (("dataset_files", files), ("dataset_bytes", size)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise WorkspaceError(
                f"{frame_file} declares {key} {value!r}, which is not an integer or null"
            )
    if pinned and (not commit or files is None or size is None):
        raise WorkspaceError(
            f"{frame_file} says dataset_pinned: true but has no complete pin "
            "(dataset_commit, dataset_files and dataset_bytes are required when "
            "pinned); refusing to read a pin that does not name its bytes"
        )
    return FramePin(
        pinned=pinned,
        dataset_commit=commit,
        dataset_manifest_hash=manifest,
        dataset_files=files,
        dataset_bytes=size,
    )


# --------------------------------------------------------------------------
# the workspace
# --------------------------------------------------------------------------


def workspace_dir(dataset_id: str) -> Path:
    """The per-dataset workspace directory. Not created here."""
    validate_dataset_id(dataset_id)
    return workspace_root() / dataset_id


def frame_path(dataset_id: str) -> Path:
    """The **one** place the dataset frame's location is decided.

    The frame is state, so it lives under the writable workspace at
    ``workspace/<dataset-id>/frame.md`` (convention §1) — never inside the
    dataset, which is mounted read-only by design. Every caller that asks where
    a frame is (``read_frame``, the inventory's ``frame`` flag and its title)
    goes through here so the two answers cannot drift apart again.
    """
    return workspace_dir(dataset_id) / FRAME_FILENAME


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


def artifact_kind(filename: str) -> str | None:
    """``"report"``, ``"session"``, or ``None`` if the name is not an artifact."""
    match = SERVABLE_ARTIFACT_RE.fullmatch(filename)
    return match.group(1) if match else None


def artifact_dir(dataset_id: str, kind: str) -> Path:
    """The **one** place a rendered artifact's directory is decided.

    Both servable kinds — ``report-*.html`` and ``session-*.html`` — live at the
    dataset root beside ``frame.md`` and ``findings.jsonl`` (convention §1 and
    §4.1: ``report-*.md/.html`` at the dataset root, and a session's HTML is the
    evidence appendix to the same deliverable). The notebook itself stays under
    ``notebooks/``; only its rendered form is served. Both the writer
    (``render``) and the reader (``__main__``, for ``/report/``) call this, so
    a correct render cannot be invisible to the index by construction.
    """
    if kind in ("report", "session"):
        return workspace_dir(dataset_id)
    raise WorkspaceError(f"unknown artifact kind {kind!r}; expected 'report' or 'session'")


def artifact_path(dataset_id: str, filename: str) -> Path:
    """Resolve one servable artifact name through :func:`artifact_dir`."""
    kind = artifact_kind(filename)
    if kind is None:
        raise WorkspaceError(
            f"{filename!r} is not a servable artifact: expected report-*.html or "
            "session-*.html"
        )
    return artifact_dir(dataset_id, kind) / filename


def artifact_paths(dataset_id: str) -> list[Path]:
    """Every rendered artifact that exists, in the convention's two locations.

    A missing directory is an empty set, not an error: a workspace that has
    never rendered anything is a normal state, not a broken one.
    """
    found: list[Path] = []
    for kind in ("report", "session"):
        try:
            entries = sorted(artifact_dir(dataset_id, kind).iterdir())
        except OSError:
            continue
        found.extend(
            path for path in entries if path.is_file() and artifact_kind(path.name) == kind
        )
    return found


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
    """The append-only ledger path for a dataset.

    ``DATA_SCOUT_LEDGER`` overrides the location when set, so ``report-render``
    can execute a notebook against a staged ledger and commit the appends only
    after its gates pass. Unset — the normal case — the answer is the per-dataset
    ``findings.jsonl``.
    """
    override = os.environ.get(LEDGER_OVERRIDE_ENV)
    if override:
        return Path(override)
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
