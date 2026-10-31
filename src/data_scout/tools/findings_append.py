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
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from data_scout import workspace
from data_scout.ledger import Finding, Ledger, utcnow
from data_scout.tools._common import measurement


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
