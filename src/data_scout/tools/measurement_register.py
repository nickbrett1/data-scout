"""``measurement_register`` — read the registry, including what is untrusted.

``validated_on`` is the only thing distinguishing a *built* instrument from a
*trusted* one: it lists corpus ids where the number has been read and trusted —
not "works", not "has been checked". At the point this belt is built it is empty
for every tool, and that emptiness is the honest state the pruning rule depends
on knowing.

Walking this alongside ``findings.jsonl`` answers the two questions the belt
cannot otherwise answer: **what has never been run** (no row) and **what has
never been trusted** (no ``validated_on`` entry).
"""

from __future__ import annotations

from typing import Any

from data_scout import registry
from data_scout.tools._common import measurement


def measurement_register(
    name: str | None = None,
    *,
    group: str | None = None,
    validated_only: bool = False,
) -> dict[str, Any]:
    """Return registry entries.

    ``value`` is the number of entries returned, ``n`` is the size of the belt,
    so a filtered read is never mistaken for the whole registry.
    """
    if name is not None:
        entries = [registry.get(name).as_record()]
    else:
        entries = registry.as_records()
        if group is not None:
            entries = [entry for entry in entries if entry.get("group") == group]
        if validated_only:
            entries = [entry for entry in entries if entry.get("validated_on")]

    unvalidated = [entry["name"] for entry in registry.as_records() if not entry.get("validated_on")]
    return measurement(
        value=len(entries),
        n=len(registry.BELT),
        unit_notes="registry entries returned",
        entries=entries,
        belt_size=len(registry.BELT),
        belt_cap=registry.MAX_BELT_SIZE,
        unvalidated=unvalidated,
        registry_problems=registry.problems(),
    )
