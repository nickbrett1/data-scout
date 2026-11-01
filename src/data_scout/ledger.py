"""``findings.jsonl`` — the append-only ledger, one record per measurement.

The design is in the absence as much as in the presence: there is **no update
method and no delete method**, because a correction is a new line and never an
edit. The history of *changed my mind* is itself the record.

Two invariants are enforced by the pydantic model rather than by advice, so a
session that did not close cannot be written down:

* every line is **decided** — ``promoted: true`` with a resolvable ``script``,
  or ``promoted: false`` with a ``retire_reason``. A line with neither is a
  session that did not close, and the model refuses it.
* ``script`` is a **module path**, not a file path — it resolves from anywhere,
  which is what a record for an importable thing should be.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utcnow() -> datetime:
    return datetime.now(UTC)


def _script_resolvable(script: str) -> bool:
    try:
        return importlib.util.find_spec(script) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


class Finding(BaseModel):
    """One line of ``findings.jsonl``.

    ``value`` is the number, complete and typed; ``n`` is what it is a number
    *of*; ``params`` is there because a value without its parameters restates
    the number instead of answering "what did I get last time".
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    ts: datetime = Field(default_factory=utcnow)
    dataset_id: str
    #: The pin the measurement was taken under, or ``None`` for an unversioned
    #: dataset. **Absent ≠ null**: a line written before this field existed has
    #: no key and its provenance is dataset_id-only ("we did not know"); a line
    #: written since has the key and may say null ("there is nothing to know").
    #: A reader must not conflate the two. ``model_dump`` fills every declared
    #: field, so re-serialising an old line *adds* ``"dataset_commit": null`` —
    #: that is why ``VOLATILE_FIELD_KEYS`` drops it from the §9 diff.
    dataset_commit: str | None = None
    session: int = Field(ge=1)
    unit: str
    tool: str
    tool_version: str
    params: dict[str, Any] = Field(default_factory=dict)
    value: Any
    n: int = Field(ge=0)
    sample: dict[str, Any] | None = None
    promoted: bool
    script: str | None = None
    retire_reason: str | None = None

    @field_validator("ts")
    @classmethod
    def _ts_is_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("ts must be timezone-aware (it is compared modulo ts)")
        return value.astimezone(UTC)

    @field_validator("value")
    @classmethod
    def _value_is_a_number_or_a_string(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("value must be a number or a string, not a bool")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"value {value!r} is not finite: a NaN in the ledger is a measurement that did not happen")
        if not isinstance(value, int | float | str):
            raise ValueError(
                f"value must be int, float or str, got {type(value).__name__}: "
                "a blob is not a measurement — put the number in `value` and "
                "the detail in a dedicated first-class field"
            )
        return value

    @field_validator("params")
    @classmethod
    def _params_are_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"params must be JSON-serializable: {exc}") from None
        return value

    @field_validator("script")
    @classmethod
    def _script_is_a_module_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value.endswith(".py") or "/" in value or os.sep in value:
            raise ValueError(
                f"script {value!r} is a file path; it must be a module path "
                "(e.g. data_scout.tools.candidate_density) so it resolves from anywhere"
            )
        if not _script_resolvable(value):
            raise ValueError(
                f"script {value!r} does not resolve to an importable module: a "
                "promoted finding must name code that exists"
            )
        return value

    @model_validator(mode="after")
    def _is_decided(self) -> Finding:
        if self.promoted:
            if not self.script:
                raise ValueError("promoted: true requires a script (module path)")
            if self.retire_reason:
                raise ValueError("promoted: true must not carry a retire_reason")
        else:
            if not self.retire_reason:
                raise ValueError(
                    "promoted: false requires a retire_reason: 'unpromoted' must "
                    "not be indistinguishable from 'forgotten'"
                )
            if self.script:
                raise ValueError("promoted: false must not carry a script")
        return self

    # ---- serialization ---------------------------------------------------

    def to_json(self) -> str:
        """One line, no trailing newline, keys in a stable order."""
        data = self.model_dump(mode="json")
        data["ts"] = self.ts.astimezone(UTC).isoformat().replace("+00:00", "Z")
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> Finding:
        return cls.model_validate_json(line)


#: Fields that legitimately differ between two executions of the same notebook
#: and are therefore not part of the measurement:
#:
#: ``observation_source`` records whether an observation came from cache or was
#: re-fetched — which the convention *requires* on the line for a
#: ``purity: network`` tool — so it must not be able to fail the diff it exists
#: to describe.
VOLATILE_PARAM_KEYS = ("observation_source",)

#: **Field-level** volatile keys, distinct from ``VOLATILE_PARAM_KEYS`` (which is
#: stripped only inside ``params``). ``ts`` always differed between executions.
#: ``dataset_commit`` legitimately varies when the store is re-pinned between
#: two runs of the same notebook: the measurement can reproduce byte for byte
#: while the pin moves. Dropping it from the multiset diff keeps the §9 gate
#: asking "did the measurement reproduce", not "was the store still"; the pin
#: movement is reported as its own verdict by :func:`pin_moved`, never as a
#: diff failure and never by silently deleting the field from the line.
VOLATILE_FIELD_KEYS = ("ts", "dataset_commit")


def normalized_line(line: str | dict, *, drop: tuple[str, ...] = VOLATILE_FIELD_KEYS) -> str:
    """A ledger line with the volatile fields dropped, for the §9 diff.

    The reproducibility claim is scoped to these lines and to nothing else:
    a rendered artifact is *not* byte-reproducible and must not be tested as if
    it were.
    """
    data = json.loads(line) if isinstance(line, str) else dict(line)
    for key in drop:
        data.pop(key, None)
    params = data.get("params")
    if isinstance(params, dict):
        for key in VOLATILE_PARAM_KEYS:
            params.pop(key, None)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def diff_lines(
    left: Sequence[str | dict],
    right: Sequence[str | dict],
    *,
    drop: tuple[str, ...] = VOLATILE_FIELD_KEYS,
) -> list[str]:
    """Multiset difference of two ledgers, modulo ``drop``.

    Returned as human-readable strings so a failed gate can print *what*
    differed rather than only that something did.
    """
    def bag(lines: Sequence[str | dict]) -> Counter[str]:
        return Counter(
            normalized_line(line, drop=drop)
            for line in lines
            if (line.strip() if isinstance(line, str) else line)
        )

    left_bag, right_bag = bag(left), bag(right)
    problems: list[str] = []
    for missing in sorted((right_bag - left_bag).elements()):
        problems.append(f"only in the re-execution: {missing}")
    for missing in sorted((left_bag - right_bag).elements()):
        problems.append(f"only in the ledger: {missing}")
    return problems


def _distinct_commits(lines: Sequence[str | dict]) -> set[str | None]:
    """The distinct ``dataset_commit`` values in a sequence of lines.

    An absent key and an explicit null are both ``None`` here: for the purpose
    of "did the pin move" they are the same provenance state (no pin named).
    """
    values: set[str | None] = set()
    for line in lines:
        if isinstance(line, str) and not line.strip():
            continue
        data = json.loads(line) if isinstance(line, str) else dict(line)
        values.add(data.get("dataset_commit"))
    return values


def pin_moved(left: Sequence[str | dict], right: Sequence[str | dict]) -> list[str]:
    """Distinct ``dataset_commit`` values on each side, when they differ.

    Empty means the two runs name the same pin (or both name none). A value on
    both sides is a move. More than one value on **one** side also reports:
    that is a mid-session move, not two runs. ``None`` is not returned as a
    commit — it is the absence of one, and the caller reads "moved" from the
    non-empty result, not from a sentinel.
    """
    left_values = _distinct_commits(left)
    right_values = _distinct_commits(right)
    if left_values == right_values:
        return []
    return sorted(value for value in (left_values | right_values) if value is not None)


def ledger_pins(findings: Sequence[Finding]) -> list[str | None]:
    """The distinct pins recorded in a ledger, sorted. More than one is a mid-session move.

    ``None`` (a line with no pin) is kept as a distinct provenance state and
    sorts first, so a ledger that mixes a pinned line and an unpinned one does
    not report a single pin by hiding the other.
    """
    values = {finding.dataset_commit for finding in findings}
    return sorted(values, key=lambda value: (value is not None, value or ""))


class Ledger:
    """The append-only ledger file.

    Deliberately narrow: ``append``, ``lines``, ``findings``, ``verify``. No
    update, no delete, no truncate. That absence is the design.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def __repr__(self) -> str:
        return f"Ledger({str(self._path)!r})"

    def lines(self) -> list[str]:
        """Every line, in file order. An absent file is an empty ledger."""
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        return [line for line in text.splitlines() if line.strip()]

    def findings(self) -> list[Finding]:
        return [Finding.from_json(line) for line in self.lines()]

    def append(self, finding: Finding | dict[str, Any]) -> int:
        """Append one decided finding; return its 1-based line number.

        Opens the file with ``"a"`` only — the single write mode this module
        ever uses. Never rewrites, never reorders, never reads-modifies-writes.
        """
        record = finding if isinstance(finding, Finding) else Finding.model_validate(finding)
        line = record.to_json()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return len(self.lines())

    def append_line(self, line: str) -> int:
        """Append one already-serialized line **byte for byte**, after validating.

        ``report-render`` uses this to promote appends a notebook made into a
        staged ledger: the bytes the kernel wrote are the bytes that land, so a
        promoted line is indistinguishable from the one the notebook would have
        appended directly. Still ``"a"`` only — the line is not rewritten, only
        checked that it is a decided finding.
        """
        Finding.from_json(line)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip("\n") + "\n")
        return len(self.lines())

    def verify(self) -> list[str]:
        """Re-validate every line. Empty means the ledger is well-formed.

        A hand-edited ledger is a ledger that has been rewritten, which is the
        one thing this file exists to rule out — so verification is a real
        re-parse, not a line count.
        """
        problems: list[str] = []
        for number, line in enumerate(self.lines(), start=1):
            try:
                Finding.from_json(line)
            except Exception as exc:  # noqa: BLE001 - report every shape of bad line
                problems.append(f"line {number}: {exc}")
        return problems
