"""``sample`` — the seeded, recorded draw. The anti-theatre instrument.

Every finding line carries ``n`` and ``sample``, and design §10 names
"sampling theatre" as a standing risk: the agent will happily report a sample
number as a corpus number. A seeded, recorded draw is the only defence, so
**the rig is a tool, not a habit** — nothing that draws a sample should be
hand-written in a cell.

The anchor that makes a draw reproducible is three things, not one: the **ids**,
the **seed**, and the **selection rule**. They are written to ``samples/``
because a seed alone reproduces nothing once the population has changed.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from data_scout import workspace
from data_scout.tools._common import measurement, share

_MAX_INLINE_IDS = 200


def _stratum_of(row: Any, stratify_by: Callable[[Any], str] | str | None) -> str:
    if stratify_by is None:
        return "all"
    if isinstance(stratify_by, str):
        if not isinstance(row, dict):
            raise TypeError(
                f"sample: stratify_by={stratify_by!r} needs dict rows, got {type(row).__name__}"
            )
        return str(row.get(stratify_by, "<missing>"))
    return str(stratify_by(row))


def _allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    """Proportional allocation by largest remainder — deterministic, no float ties."""
    population = sum(counts.values())
    if population == 0:
        return {key: 0 for key in counts}
    quotas = {key: total * count / population for key, count in counts.items()}
    allocation = {key: int(quota) for key, quota in quotas.items()}
    remaining = total - sum(allocation.values())
    ordered = sorted(quotas, key=lambda key: (-(quotas[key] - allocation[key]), key))
    for key in ordered[:remaining]:
        allocation[key] += 1
    # A stratum smaller than its allocation cannot give more rows than it has.
    for key, count in counts.items():
        allocation[key] = min(allocation[key], count)
    return allocation


def sample(
    rows: Sequence[Any],
    *,
    n: int,
    seed: int = 0,
    stratify_by: Callable[[Any], str] | str | None = None,
    id_of: Callable[[Any], str] | None = None,
    with_replacement: bool = False,
) -> dict[str, Any]:
    """Draw ``n`` rows, stratified and seeded, and say exactly how.

    ``value`` is the sampling fraction actually achieved (``draw / population``),
    ``n`` is the population size. ``ids`` is the anchor — capped inline, and
    ``anchor=True`` says whether it was trimmed, because a trimmed anchor that
    does not admit it is a lie about reproducibility.
    """
    population = list(rows)
    if n < 0:
        raise ValueError("sample: n must be >= 0")
    if not with_replacement and n > len(population):
        raise ValueError(
            f"sample: refusing to draw {n} from {len(population)} rows without "
            "replacement — a sample larger than the population is a census with a "
            "misleading label"
        )
    if id_of is None:
        id_of = lambda row: str(row.get("id")) if isinstance(row, dict) and "id" in row else str(row)  # noqa: E731

    strata: dict[str, list[Any]] = {}
    for row in population:
        strata.setdefault(_stratum_of(row, stratify_by), []).append(row)
    counts = {key: len(value) for key, value in strata.items()}
    allocation = _allocate(counts, n)

    drawn: list[Any] = []
    drawn_ids: list[str] = []
    per_stratum: dict[str, int] = {}
    for key in sorted(strata):
        members = strata[key]
        take = allocation.get(key, 0)
        if with_replacement:
            chosen = [members[random.Random(f"{seed}:{key}").randrange(len(members))] for _ in range(take)]
        else:
            chosen = random.Random(f"{seed}:{key}").sample(members, take)
        per_stratum[key] = len(chosen)
        drawn.extend(chosen)
        drawn_ids.extend(id_of(row) for row in chosen)

    inline_ids = drawn_ids[:_MAX_INLINE_IDS]
    return measurement(
        value=share(len(drawn), len(population)),
        n=len(population),
        unit_notes="sampling fraction achieved (drawn / population)",
        drawn=len(drawn),
        requested=n,
        population=len(population),
        seed=seed,
        rule=(
            "stratified proportional draw (largest remainder), strata seeded as "
            "'{seed}:{stratum}'"
            if stratify_by is not None
            else "uniform draw from random.Random('{seed}:all')"
        ),
        stratify_by=(
            stratify_by if isinstance(stratify_by, str) or stratify_by is None else "<callable>"
        ),
        with_replacement=with_replacement,
        strata=per_stratum,
        stratum_populations=dict(sorted(counts.items())),
        ids=inline_ids,
        ids_truncated=len(drawn_ids) > _MAX_INLINE_IDS,
        missing_strata=sorted(key for key, count in counts.items() if allocation.get(key, 0) == 0 and count),
    )


def write_sample_anchor(
    result: dict[str, Any],
    *,
    corpus_id: str | None = None,
    session: int | None = None,
    directory: str | os.PathLike[str] | None = None,
) -> Path:
    """Write the draw's anchor into ``samples/`` and return the path.

    The anchor is what makes a draw reproducible after the population changes:
    the ids and the rule travel with the seed. Writing it is a workspace write,
    which is the only kind of write this project does.
    """
    corpus_id = corpus_id or workspace.current_corpus_id()
    session = session or workspace.current_session() or 1
    target_dir = Path(directory) if directory is not None else workspace.samples_dir(corpus_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus_id": corpus_id,
        "session": session,
        "seed": result.get("seed"),
        "rule": result.get("rule"),
        "stratify_by": result.get("stratify_by"),
        "population": result.get("population"),
        "drawn": result.get("drawn"),
        "strata": result.get("strata"),
        "ids": result.get("ids"),
        "ids_truncated": result.get("ids_truncated"),
    }
    path = target_dir / f"sample-s{session}-seed{result.get('seed')}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
