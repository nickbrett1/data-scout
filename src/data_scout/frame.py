"""The dataset frame — an identity claim about the data, and the rules behind it.

``frame.md`` used to record a single figure over the whole checkout. DVC
rewrites ``.dvc/tmp/lock`` and ``.dvc/tmp/rwlock.lock`` whenever it runs, so
that figure moved by two bytes 3 h 44 m after it was measured, with the payload
provably untouched. The number looked like a measurement of the data and was
partly a measurement of the tooling — item 2's defect one layer up.

The fix is structural, not a filter:

* **The identity is the payload.** ``data/`` is the scope: names and sizes,
  sorted, hashed. That is what "which corpus is this" means.
* **The whole-checkout figure is kept, but it is not the identity.** It is
  recorded as ``checkout`` beside the payload identity, and it is measured
  under the same declared exclusions so it cannot drift on VCS scratch either.
* **Exclusions are an explicit, named list** (:data:`FRAME_EXCLUSIONS`) — a
  property of this module, not a per-call option. A caller cannot ask for a
  different rule, because a number measured under a rule nobody can see is the
  defect this module exists to remove.
* **The schema version is emitted here**, by the code that measures the frame
  (:data:`FRAME_SCHEMA_VERSION`), never supplied by a caller. A v1 frame (no
  ``frame_schema`` key) and a v2 frame are *not comparable*, and
  :func:`compare_frames` says so instead of reporting a byte delta.
* **``unit`` does not move.** The frame's unit is a file-level identity claim
  and stays ``files`` (phase 2a settled this); ``document`` is the unit for
  measurement.

The exclusion set is deliberately not configurable. If a future dataset needs a
different rule, that is a new schema version, measured by code that records it.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data_scout import workspace

#: The exclusion set lives in :mod:`data_scout.exclusions` — the frame, the
#: toolbelt's walks, the inventory and the report all have to agree on what is
#: *not* the dataset, and a second copy of the rule is a second answer.
#: (Re-exported here because the ledger already names these.)
from data_scout.exclusions import (  # noqa: F401
    EXCLUSION_NAMES,
    FRAME_EXCLUSIONS,
    Exclusion,
)

#: The version of the *frame schema* this module emits. A frame without a
#: ``frame_schema`` key is version 1 (the phase-1 record on ``vg-books-2026-10``
#: is one). Version 2 adds the payload identity, the declared exclusion set and
#: the checkout figure as separate, labelled facts. Version 3 adds the dataset
#: pin the measurement was taken under, and the ``dataset-pin`` exclusion that
#: keeps ``pin.json`` out of the checkout figure.
FRAME_SCHEMA_VERSION = 3

#: Versions this reader understands. A frame from a newer schema fails loudly
#: rather than being read under rules it was not measured with.
SUPPORTED_FRAME_SCHEMAS = (1, 2, 3)

#: The tool name on a frame identity line in ``findings.jsonl``. Not a belt
#: instrument — the frame is part of the substrate, not a sixteenth name.
FRAME_TOOL = "frame"

#: How the manifest is computed, recorded in the frame so a reader never has to
#: guess. Names and sizes only: this is an identity claim, not a content hash.
MANIFEST_RULE = (
    "sha256 over 'relpath<TAB>size<LF>' for every included file, sorted by "
    "relpath; symlinks skipped; names and sizes only, not content"
)

#: The two scopes a v2 frame records. ``identity`` is the claim about the data;
#: ``checkout`` is the claim about the tree on disk.
IDENTITY_SCOPE = "data"
CHECKOUT_SCOPE = "."


class FrameSchemaError(workspace.WorkspaceError):
    """A frame declares a schema this reader does not understand."""


class FrameMeasureError(RuntimeError):
    """A frame cannot be written without silently replacing an existing one."""


class PinStale(FrameMeasureError):
    """The pin's files/bytes disagree with the payload just measured.

    A pin that does not describe the bytes in front of it is not a weaker pin,
    it is a false claim, and both the frame and the ledger are append-only. Fail
    before writing, never after (D5).

    The residual hole, stated rather than papered over: two commits whose trees
    have the *same* file count and total bytes are indistinguishable to this
    check (``frame.py`` deliberately does not hash content — ``MANIFEST_RULE``
    is names and sizes only). So a re-commit mid-session that changes bytes
    without changing shape would pass. Mitigations, all already in place: §6.6
    of the store order forbids recording a pin from a dirty tree;
    ``write_frame`` refuses to replace an existing frame without an
    ``overwrite_reason``; and a re-commit during a session is a human act that
    ought to be followed by a frame re-measure.
    """

@dataclass(frozen=True)
class TreeIdentity:
    """One scope, measured: how many files, how many bytes, which manifest."""

    scope: str
    files: int
    bytes: int
    manifest: str
    exclusions: tuple[str, ...] = ()
    excluded_files: int = 0
    excluded_bytes: int = 0

    def as_record(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "files": self.files,
            "bytes": self.bytes,
            "manifest": self.manifest,
            "exclusions": list(self.exclusions),
            "excluded_files": self.excluded_files,
            "excluded_bytes": self.excluded_bytes,
        }


@dataclass(frozen=True)
class FrameMeasurement:
    """A frame measured under this schema: the payload identity and the checkout."""

    dataset_id: str
    schema_version: int
    unit: str
    identity: TreeIdentity
    checkout: TreeIdentity
    measured: str
    pin: workspace.DatasetPin | None

    @property
    def exclusions(self) -> tuple[str, ...]:
        # Both scopes are measured under the same declared set; recording it
        # once keeps "which exclusions were in force" a single answer.
        return self.identity.exclusions

    def as_record(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "frame_schema": self.schema_version,
            "unit": self.unit,
            "exclusions": list(self.exclusions),
            "identity": self.identity.as_record(),
            "checkout": self.checkout.as_record(),
            "manifest_rule": MANIFEST_RULE,
            "measured": self.measured,
        }


# --------------------------------------------------------------------------
# measuring
# --------------------------------------------------------------------------


def _excluded(relative: str, exclusions: Sequence[Exclusion]) -> Exclusion | None:
    for item in exclusions:
        if relative == item.path or relative.startswith(item.path + "/"):
            return item
    return None


def _tally(root: Path) -> tuple[int, int]:
    """Files and bytes under ``root``, symlinks skipped."""
    files = total = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            try:
                total += path.stat().st_size
            except OSError:
                continue
            files += 1
    return files, total


def measure_tree(
    root: Path,
    *,
    scope: str,
    exclusions: Sequence[Exclusion] | None = None,
) -> TreeIdentity:
    """Measure one scope under the declared exclusions.

    Private in spirit: the public entry point is :func:`measure_frame`, which
    fixes the scopes and the exclusion set. This function takes them because a
    measurement should be *reused*, not because a caller may choose a rule.
    """
    if exclusions is None:
        exclusions = FRAME_EXCLUSIONS
    base = root if scope in {"", ".", "./"} else root / scope
    records: list[tuple[str, int]] = []
    excluded_files = excluded_bytes = 0

    if base.is_dir():
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            directory = Path(dirpath)
            relative_dir = directory.relative_to(root).as_posix()
            if relative_dir == ".":
                relative_dir = ""

            keep: list[str] = []
            for name in sorted(dirnames):
                relative = f"{relative_dir}/{name}" if relative_dir else name
                if _excluded(relative, exclusions):
                    kept_files, kept_bytes = _tally(directory / name)
                    excluded_files += kept_files
                    excluded_bytes += kept_bytes
                else:
                    keep.append(name)
            dirnames[:] = keep

            for name in sorted(filenames):
                path = directory / name
                if path.is_symlink():
                    continue
                relative = f"{relative_dir}/{name}" if relative_dir else name
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if _excluded(relative, exclusions):
                    excluded_files += 1
                    excluded_bytes += size
                    continue
                records.append((relative, size))

    records.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    for relative, size in records:
        digest.update(f"{relative}\t{size}\n".encode())

    return TreeIdentity(
        scope=scope,
        files=len(records),
        bytes=sum(size for _relative, size in records),
        manifest=digest.hexdigest(),
        exclusions=tuple(item.name for item in exclusions),
        excluded_files=excluded_files,
        excluded_bytes=excluded_bytes,
    )


def measure_frame(dataset_id: str, *, root: Path | None = None) -> FrameMeasurement:
    """Measure a dataset's frame under this schema.

    ``root`` exists so a test can measure a fixture without an environment
    mount; the scopes and the exclusion set are **not** parameters. That is the
    point: the rule belongs to the schema, not to the caller.
    """
    workspace.validate_dataset_id(dataset_id)
    if root is None:
        root = workspace.dataset_path(dataset_id)
    root = Path(root)

    identity = measure_tree(root, scope=IDENTITY_SCOPE)
    checkout = measure_tree(root, scope=CHECKOUT_SCOPE)

    # The one place a pin is read for a frame. ``read_pin`` substitutes its
    # ``root`` for the datasets root, and ``root`` here is the dataset
    # directory, so the pin is read from its parent — the fixture's pin beside
    # the fixture's payload, the real pin beside the real payload.
    pin = workspace.read_pin(dataset_id, root=root.parent)
    if pin is not None and (
        pin.dataset_files != identity.files or pin.dataset_bytes != identity.bytes
    ):
        raise PinStale(
            f"pin at {root / workspace.PIN_FILENAME} says {pin.dataset_files} "
            f"files / {pin.dataset_bytes} B but the payload measured "
            f"{identity.files} files / {identity.bytes} B; the pin describes "
            "different bytes than this session measured"
        )

    return FrameMeasurement(
        dataset_id=dataset_id,
        # Emitted here, by the code that measured it. Never a caller field.
        schema_version=FRAME_SCHEMA_VERSION,
        unit="files",
        identity=identity,
        checkout=checkout,
        measured=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        pin=pin,
    )


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def render_frame(
    measurement: FrameMeasurement,
    *,
    lenses: Sequence[str] = (),
    tier_ceiling: int = 1,
    created: str | None = None,
    supersedes_reason: str | None = None,
) -> str:
    """Render a v3 ``frame.md``. The exclusion set and the pin are written down,
    not implied.

    ``supersedes_reason`` is written only when an existing frame is replaced, so
    the destruction of the old text is documented on the same disk as the new
    text. It is a record, not a verdict: it says why the previous frame stopped
    being the frame, not whether the new one is better.
    """
    created = created or measurement.measured[:10]
    pin = measurement.pin
    lines = [
        "---",
        f"dataset_id: {measurement.dataset_id}",
        f"frame_schema: {measurement.schema_version}",
        f"unit: {measurement.unit}",
        "frame_exclusions:",
        *(f"  - {name}" for name in measurement.exclusions),
        f"identity_scope: {measurement.identity.scope}",
        f"identity_files: {measurement.identity.files}",
        f"identity_bytes: {measurement.identity.bytes}",
        f"identity_manifest: {measurement.identity.manifest}",
        f"checkout_scope: {measurement.checkout.scope}",
        f"checkout_files: {measurement.checkout.files}",
        f"checkout_bytes: {measurement.checkout.bytes}",
        f"checkout_manifest: {measurement.checkout.manifest}",
        f"manifest_rule: {MANIFEST_RULE}",
        f"created: {created}",
        f"dataset_pinned: {'true' if pin else 'false'}",
        f"dataset_commit: {pin.dataset_commit if pin else 'null'}",
        f"dataset_manifest_hash: {pin.dataset_manifest_hash if pin else 'null'}",
        f"dataset_files: {pin.dataset_files if pin else 'null'}",
        f"dataset_bytes: {pin.dataset_bytes if pin else 'null'}",
    ]
    if supersedes_reason:
        lines.append(f"supersedes_reason: {supersedes_reason}")
    if lenses:
        lines.append("lenses:")
        lines.extend(f"  - {lens}" for lens in lenses)
    lines.append(f"tier_ceiling: {tier_ceiling}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {measurement.dataset_id}")
    lines.append("")
    lines.append(
        f"Identity is the payload under `{measurement.identity.scope}/` "
        f"({measurement.identity.files} "
        f"{'file' if measurement.identity.files == 1 else 'files'}, "
        f"{measurement.identity.bytes} B). "
        f"The checkout figure includes the rest of the tree except the declared "
        f"exclusions ({', '.join(measurement.exclusions)}), which are VCS "
        "transient state and not part of the corpus."
    )
    lines.append("")
    return "\n".join(lines)


def write_frame(
    dataset_id: str,
    measurement: FrameMeasurement,
    *,
    lenses: Sequence[str] = (),
    tier_ceiling: int = 1,
    created: str | None = None,
    overwrite: bool = False,
    overwrite_reason: str | None = None,
) -> Path:
    """Write a **new** frame. An existing frame is a record, not a draft.

    ``frame.md`` says what was measured under the rules in force at the time.
    Re-measuring an existing frame is a deliberate decision, so this refuses to
    replace one unless the caller says so explicitly.

    ``overwrite=True`` is not enough on its own: destroying the old text with no
    trail is the same defect shape as a measurement without its parameters.
    Replacing a frame therefore requires a non-empty ``overwrite_reason``, which
    is written into the new frame as ``supersedes_reason``. A reason without a
    replacement is refused too — it would document a destruction that did not
    happen.
    """
    path = workspace.frame_path(dataset_id)
    if path.exists() and not overwrite:
        raise FrameMeasureError(
            f"{path} already exists. A frame records what was measured under the "
            "rules in force then; re-measuring it is a decision, not an overwrite. "
            "Pass overwrite=True and overwrite_reason=... only when that decision "
            "has been made."
        )
    if overwrite:
        if not overwrite_reason or not overwrite_reason.strip():
            raise FrameMeasureError(
                "overwrite=True requires a non-empty overwrite_reason: replacing "
                f"{path} destroys the previous frame's text, and the record of why "
                "must survive on disk (it is written as supersedes_reason)."
            )
        if "\n" in overwrite_reason or "\r" in overwrite_reason:
            raise FrameMeasureError(
                "overwrite_reason must be one line: it is written into the frame's "
                "frontmatter, and a newline would turn the rest of the reason into "
                "fields nobody declared."
            )
    elif overwrite_reason:
        raise FrameMeasureError(
            "overwrite_reason was given without overwrite=True; no frame is being "
            "replaced, so there is no destruction to document."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_frame(
            measurement,
            lenses=lenses,
            tier_ceiling=tier_ceiling,
            created=created,
            supersedes_reason=overwrite_reason if overwrite else None,
        ),
        encoding="utf-8",
    )
    return path


def identity_finding(
    measurement: FrameMeasurement,
    *,
    dataset_id: str,
    session: int,
    promoted: bool,
    script: str | None = None,
    retire_reason: str | None = None,
) -> dict[str, Any]:
    """The ledger line for a frame, with the schema version emitted, not supplied.

    ``value`` is the payload identity's byte count and ``n`` its file count —
    the claim about the data. The checkout figure is recorded in ``params`` so
    it is visible without being mistaken for the identity.
    """
    params: dict[str, Any] = {
        "frame_schema": measurement.schema_version,
        "frame_exclusions": list(measurement.exclusions),
        "manifest_rule": MANIFEST_RULE,
        "identity_scope": measurement.identity.scope,
        "identity_files": measurement.identity.files,
        "identity_manifest": measurement.identity.manifest,
        "checkout_scope": measurement.checkout.scope,
        "checkout_files": measurement.checkout.files,
        "checkout_bytes": measurement.checkout.bytes,
        "checkout_manifest": measurement.checkout.manifest,
        "checkout_excluded_files": measurement.checkout.excluded_files,
        "dataset_pinned": measurement.pin is not None,
        "dataset_commit": measurement.pin.dataset_commit if measurement.pin else None,
        "dataset_manifest_hash": (
            measurement.pin.dataset_manifest_hash if measurement.pin else None
        ),
        "dataset_files": measurement.pin.dataset_files if measurement.pin else None,
        "dataset_bytes": measurement.pin.dataset_bytes if measurement.pin else None,
    }
    record: dict[str, Any] = {
        "dataset_id": dataset_id,
        "session": session,
        "unit": measurement.unit,
        "tool": FRAME_TOOL,
        # Emitted by the measurer, never a caller argument.
        "tool_version": str(measurement.schema_version),
        "params": params,
        "value": measurement.identity.bytes,
        "n": measurement.identity.files,
        "promoted": promoted,
    }
    if promoted:
        record["script"] = script or "data_scout.frame"
    if retire_reason is not None:
        record["retire_reason"] = retire_reason
    return record


# --------------------------------------------------------------------------
# append-once at promotion (frame lines only)
# --------------------------------------------------------------------------

#: The five pin fields that say *which bytes* a frame measurement describes.
#: They are part of :func:`frame_dedup_key` on purpose — see its docstring.
FRAME_PIN_FIELDS: tuple[str, ...] = (
    "dataset_pinned",
    "dataset_commit",
    "dataset_manifest_hash",
    "dataset_files",
    "dataset_bytes",
)


def frame_dedup_key(line: str | dict[str, Any]) -> str:
    """The append-once identity of a frame line: content minus ``ts``, plus the pin.

    This is **not** ``ledger.normalized_line``'s default key, and must not be
    made one. That key deliberately drops ``dataset_commit`` so the §9
    reproduction gate does not read a legitimate re-pin as a failure to
    reproduce (§3.5a). Reusing that drop set here would invert this rule: a
    genuine mid-session re-pin can produce identical normalized content, so the
    second frame line — the one that says the bytes moved — would be silently
    suppressed while ``frame.md`` and the checkout had moved. That is the exact
    defect this feature exists to close, mirrored.

    So the key is the whole line with only ``ts`` removed (every other field,
    ``dataset_commit`` included, is identity), and the five pin fields are
    lifted into it explicitly rather than trusted to be present in ``params``.
    """
    data = json.loads(line) if isinstance(line, str) else dict(line)
    params = data.get("params")
    if not isinstance(params, dict):
        params = {}
    data.pop("ts", None)
    data["frame_pin"] = {
        name: data.get(name, params.get(name)) for name in FRAME_PIN_FIELDS
    }
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def is_frame_line(line: str | dict[str, Any]) -> bool:
    """True when this ledger line is the frame identity line."""
    if isinstance(line, str):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return False
    else:
        record = line
    return isinstance(record, dict) and record.get("tool") == FRAME_TOOL


def promote_once(
    appended: Sequence[str], existing: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Split staged appends into ``(to_append, skipped)`` under append-once.

    **Frame lines only.** The ledger holds at most one current frame statement
    per distinct measurement, so re-rendering an unchanged notebook does not
    grow a duplicate frame line. The rule is deliberately not applied to other
    tools: two identical findings from two executions are two events, and both
    stay.

    "Distinct measurement" is :func:`frame_dedup_key` — content minus ``ts``
    plus the pin. A re-pin is therefore a new measurement and appends; so is a
    genuine re-measure that changes any identity or measurement field.
    """
    seen = {frame_dedup_key(line) for line in existing if is_frame_line(line)}
    keep: list[str] = []
    skipped: list[str] = []
    for line in appended:
        if not is_frame_line(line):
            keep.append(line)
            continue
        key = frame_dedup_key(line)
        if key in seen:
            skipped.append(line)
            continue
        seen.add(key)
        keep.append(line)
    return keep, skipped


# --------------------------------------------------------------------------
# comparability
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """The result of comparing two frames.

    ``delta`` is only present when the frames are comparable. A not-comparable
    result must never carry a byte delta: the delta would be between rules, not
    between data.
    """

    comparable: bool
    left_schema: int
    right_schema: int
    reason: str | None = None
    delta: dict[str, Any] | None = None
    #: Populated only when both frames are pinned to different commits. It is a
    #: separate field, never folded into ``reason``: a reason is prose for a
    #: human, and a pin move is a fact a future prune must be able to read.
    pin_changed: dict[str, str] | None = None


def _pin_commit(pin: workspace.FramePin | None) -> str | None:
    """The commit a frame's pin names, or ``None`` when the frame is unpinned."""
    if pin is None or not pin.pinned:
        return None
    return pin.dataset_commit


def compare_frames(left: workspace.Frame, right: workspace.Frame) -> Comparison:
    """Compare two frames, refusing a delta when they were measured differently.

    Different schema versions, or different exclusion sets, mean the numbers are
    claims about different rules. Reporting a byte delta there would be drift
    fiction.
    """
    if left.schema_version != right.schema_version:
        return Comparison(
            comparable=False,
            left_schema=left.schema_version,
            right_schema=right.schema_version,
            reason=(
                "not comparable: these frames were measured under different frame "
                f"rules (schema v{left.schema_version} and v{right.schema_version}); "
                "a byte delta would be a delta between rules, not data"
            ),
        )
    if left.exclusions != right.exclusions:
        return Comparison(
            comparable=False,
            left_schema=left.schema_version,
            right_schema=right.schema_version,
            reason=(
                "not comparable: different exclusions were in force "
                f"({list(left.exclusions)} vs {list(right.exclusions)}); a byte "
                "delta would be a delta between rules, not data"
            ),
        )
    if left.identity is None or right.identity is None:
        return Comparison(
            comparable=False,
            left_schema=left.schema_version,
            right_schema=right.schema_version,
            reason=(
                "not comparable: at least one frame records no payload identity "
                "(an identity scope and manifest are required)"
            ),
        )
    left_commit = _pin_commit(left.pin)
    right_commit = _pin_commit(right.pin)
    if (left_commit is None) != (right_commit is None):
        return Comparison(
            comparable=False,
            left_schema=left.schema_version,
            right_schema=right.schema_version,
            reason=(
                "not comparable: one frame is pinned and the other is not, so one "
                "side does not know which bytes it measured; a delta would be "
                "between provenance states, not data"
            ),
        )
    pin_changed: dict[str, str] | None = None
    if left_commit is not None and right_commit is not None and left_commit != right_commit:
        pin_changed = {"from": left_commit, "to": right_commit}
    return Comparison(
        comparable=True,
        left_schema=left.schema_version,
        right_schema=right.schema_version,
        pin_changed=pin_changed,
        delta={
            "scope": left.identity.scope,
            "files": right.identity.files - left.identity.files,
            "bytes": right.identity.bytes - left.identity.bytes,
            "manifest_same": left.identity.manifest == right.identity.manifest,
        },
    )


def _frame_lines(lines: Sequence[str | dict]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for line in lines:
        if isinstance(line, str):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
        else:
            record = dict(line)
        if record.get("tool") == FRAME_TOOL:
            found.append(record)
    return found


def _line_schema(record: dict[str, Any]) -> int:
    params = record.get("params")
    value = params.get("frame_schema", 1) if isinstance(params, dict) else 1
    # An unreadable version is not v1; it is unknown, and unknown is not comparable.
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def ledger_frame_schemas(lines: Sequence[str | dict]) -> list[int]:
    """The set of frame schema versions present in a ledger, sorted.

    ``tool: "frame"`` with no ``params.frame_schema`` is v1 (the phase-1 record);
    a version that cannot be read is ``-1``, never silently dropped. An empty
    list means **no frame line is present**, which is a different fact from
    "a frame was measured" — that is why the inventory reports this beside
    ``frame: true`` rather than folding it into it.

    Multiple versions in one ledger are normal and not an error: the ledger is
    append-only, so a v2 line sits beside the v1 line it makes incomparable.
    A reader who sees the set can tell which rules the numbers beside it were
    measured under; ``frame: true`` alone cannot.
    """
    return sorted({_line_schema(record) for record in _frame_lines(lines)})


def comparability_problems(
    left: Sequence[str | dict], right: Sequence[str | dict]
) -> list[str]:
    """Problems that make two runs' frame lines incomparable. Empty means none.

    This is the ledger-side counterpart of :func:`compare_frames`: the frame is
    a line in ``findings.jsonl``, so ``report-render``'s diff is where a v1 line
    and a v2 line would otherwise be read as drift. Absence of a frame line on
    either side makes no claim, so it is not a problem.
    """
    left_versions = sorted({_line_schema(record) for record in _frame_lines(left)})
    right_versions = sorted({_line_schema(record) for record in _frame_lines(right)})
    if not left_versions or not right_versions:
        return []
    if left_versions != right_versions:
        return [
            "frame identity not comparable: the two runs measured it under "
            f"different frame rules (frame schema {left_versions} vs "
            f"{right_versions}). A byte delta here would be a delta between "
            "rules, not data; no drift verdict is possible."
        ]
    return []
