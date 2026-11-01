"""``findings_append`` — the append *is* the promotion decision, made visible.

There is no update and no delete: a correction is a new line, and the history of
*changed my mind* is itself the record. The pydantic model behind this is what
refuses a line that was never decided (``promoted: true`` with a resolvable
``script``, or ``promoted: false`` with a ``retire_reason``), so a session that
did not close cannot be written down.

``ts``, ``dataset_id`` and ``session`` are filled from the environment when the
record omits them: ``report-render`` sets ``DATA_SCOUT_DATASET_ID`` and
``DATA_SCOUT_SESSION`` for the kernel, so a cell can append a complete line
without repeating the frame in every call.

:func:`derive_finding` is the other half: it **derives** one line from a
measurement instead of copying one that already exists. It is ledger-blind by
construction — no path, no ``Ledger``, no read — so a session can write its
first line into a workspace where ``findings.jsonl`` has never existed.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from data_scout import registry, workspace
from data_scout.ledger import Finding, Ledger, utcnow
from data_scout.tools._common import _stable, measurement


def _calling_instrument_name() -> str:
    """The belt instrument that called :func:`derive_finding`, from its frame.

    ``tool`` is a property of the code that measured, so it is *read* from the
    caller (``__name__``) rather than typed by hand at every call site: a
    caller that has to spell its own name is a caller that can spell it wrong,
    and a wrong tool name on a line is a wrong provenance claim.

    The walk skips this module's own frames, so the answer is the first caller
    outside it. A caller that is not in the belt — a notebook cell, whose
    ``__name__`` is ``__main__``, or a helper named ``measure`` — is refused by
    name rather than guessed at: passing ``tool=`` is the explicit fix, and a
    guessed identity is exactly what this function exists to prevent.
    """
    frame = inspect.currentframe()
    frame = frame.f_back if frame is not None else None
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
    if frame is None:
        raise ValueError(
            "derive_finding: no `tool` given and no calling frame could be "
            "inspected. Pass tool= explicitly."
        )
    name = frame.f_code.co_name
    module = str(frame.f_globals.get("__name__", "<unknown>"))
    if name in registry.BELT:
        return name
    raise ValueError(
        f"derive_finding: no `tool` given, and the calling instrument "
        f"{module}.{name} is not in the belt ({', '.join(registry.names())}). "
        "Call it from the instrument that measured, or pass tool= — a tool "
        "identity that has to be guessed is not a derived one."
    )


def derive_finding(
    *,
    tool: str | None = None,
    unit: str | None = None,
    value: Any,
    n: int | None = None,
    params: Mapping[str, Any] | None = None,
    promoted: bool = False,
    retire_reason: str | None = None,
    sample: Mapping[str, Any] | None = None,
    dataset_id: str | None = None,
    session: int | None = None,
) -> dict[str, Any]:
    """Derive one decided finding line from a measurement. **Ledger-blind.**

    Takes no ledger, reads no file and consults no existing record: the line it
    returns is a function of the arguments and the frame stamp, which is why it
    works the first time, in a workspace where ``findings.jsonl`` has never
    existed. Appending is somebody else's job (``Ledger.append`` or
    :func:`findings_append`); this function only decides.

    **Identity is derived, not declared.** ``tool`` names the instrument (and
    is read from the calling frame when omitted); ``tool_version`` and
    ``script`` come from that instrument's entry in
    :mod:`data_scout.registry`, because they are facts about the code that
    produced the number and never a caller's opinion. ``script`` is therefore
    set exactly when ``promoted`` is true, and it is the module path the
    registry already publishes — the same string that resolves from anywhere.

    **``params`` must carry what the measurement actually used**, including the
    observation scope (``params["scope"]``: ``"."``, ``"data/"``, or the file
    measured) and, where the measurement pinned a file set, its manifest. A
    value without its parameters restates the number instead of answering
    *what did I get last time*; a value without its scope does not even say
    which part of the dataset it is about.

    ``n`` has no default: a finding is a number *of* something, and "not
    supplied" is not a count. ``0`` is a real count and passes.

    The returned mapping is the ledger line as ``Finding.to_json`` writes it
    (sorted keys, ``ts`` rendered with ``Z``), so it round-trips through
    ``Ledger.append`` and through ``normalized_line`` unchanged. ``value`` and
    ``params`` are rounded by the belt's one rounding rule, so a derived line
    re-derives byte for byte.
    """
    spec = registry.get(tool if tool is not None else _calling_instrument_name())

    if n is None:
        raise ValueError(
            f"derive_finding({spec.name}): `n` is required — a finding is a "
            "number *of* something, and 'not supplied' is not a count. Pass the "
            "population the value was measured over (0 is a legitimate count)."
        )
    if params is None:
        raise ValueError(
            f"derive_finding({spec.name}): `params` is required — a value "
            "without its parameters restates the number instead of saying what "
            "it was measured on. It must carry the observation scope as "
            "params['scope'] and, where the file set is pinned, its manifest."
        )
    resolved_params = dict(params)
    scope = resolved_params.get("scope")
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError(
            f"derive_finding({spec.name}): params['scope'] must name the "
            f"observation scope ('.' for the dataset root, 'data/' for a "
            f"subtree, or the file measured); got {scope!r}. A number without "
            "its scope is a number about an unknown part of the dataset."
        )

    payload: dict[str, Any] = {
        "dataset_id": dataset_id or workspace.current_dataset_id(),
        "session": session if session is not None else _session_or_fail(),
        "unit": unit if unit is not None else spec.unit,
        "tool": spec.name,
        "tool_version": spec.version,
        "params": _stable(resolved_params),
        "value": _stable(value),
        "n": n,
        "sample": _stable(dict(sample)) if sample is not None else None,
        "promoted": promoted,
        "script": spec.module if promoted else None,
        "retire_reason": retire_reason,
    }
    finding = Finding.model_validate(payload)
    return json.loads(finding.to_json())


def _session_or_fail() -> int:
    """The session from the environment, or a loud refusal (as ``findings_append``)."""
    resolved = workspace.current_session()
    if resolved is None:
        raise ValueError(
            "derive_finding: no `session` given and "
            f"{workspace.SESSION_ENV} is not set. A row has to tie back to the "
            "notebook that produced it."
        )
    return resolved


def findings_append(
    record: Mapping[str, Any],
    *,
    ledger_path: str | os.PathLike[str] | None = None,
    dataset_id: str | None = None,
    session: int | None = None,
) -> dict[str, Any]:
    """Append one decided finding line.

    ``value`` is 1 (one record), ``n`` is 1. The detail is where it landed.
    """
    payload: dict[str, Any] = dict(record)
    payload.setdefault("ts", utcnow())
    if "dataset_id" not in payload:
        payload["dataset_id"] = dataset_id or workspace.current_dataset_id()
    if "session" not in payload:
        resolved_session = session or workspace.current_session()
        if resolved_session is None:
            raise ValueError(
                "findings_append: no `session` in the record and "
                f"{workspace.SESSION_ENV} is not set. A row has to tie back to the "
                "notebook that produced it."
            )
        payload["session"] = resolved_session

    finding = Finding.model_validate(payload)
    path = Path(ledger_path) if ledger_path is not None else workspace.ledger_path(finding.dataset_id)
    line_number = Ledger(path).append(finding)
    return measurement(
        value=1,
        n=1,
        unit_notes="record appended",
        ledger=str(path),
        line=line_number,
        tool=finding.tool,
        promoted=finding.promoted,
    )


def append_many(records: list[Mapping[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
    """Append several decided records, in order. Still append-only."""
    return [findings_append(record, **kwargs) for record in records]
