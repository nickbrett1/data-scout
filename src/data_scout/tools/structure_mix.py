"""``structure_mix`` — per-document share of each part type. Tier 1, pure.

Pure over a part list, which is the whole point: ``partition_doc`` pays for the
observation (and caches it), this recomputes the number every run. A prose book
and a plate book separate immediately here, which is what makes the structure
lens worth a phase of its own.

The headline is ``narrative_share`` — the share of parts that carry prose —
because that is the number a caller compares across documents. The full mix is
beside it as a first-class field, never as a serialised blob.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from data_scout.tools._common import measurement, share

#: What counts as prose. Section headers are structure, not prose; list items
#: are prose in a list. Both are reported separately so a caller can disagree.
_PROSE = frozenset({"NarrativeText", "ListItem"})


def _part_type(part: Any) -> str:
    if isinstance(part, str):
        return part.strip() or "Other"
    if isinstance(part, Mapping):
        value = part.get("type") or part.get("label") or "Other"
        return str(value)
    raise TypeError(f"structure_mix: expected a part (str or mapping), got {type(part).__name__}")


def _as_parts(parts: Any) -> list[Any]:
    """Accept a partition_doc result, a part list, or a list of labels."""
    if isinstance(parts, Mapping):
        inner = parts.get("parts")
        if inner is None:
            counts = parts.get("part_type_counts")
            if isinstance(counts, Mapping):
                expanded: list[str] = []
                for label, count in counts.items():
                    expanded.extend([str(label)] * int(count))
                return expanded
            raise ValueError("structure_mix: mapping has neither 'parts' nor 'part_type_counts'")
        return list(inner)
    if isinstance(parts, Sequence) and not isinstance(parts, str | bytes):
        return list(parts)
    raise TypeError(
        f"structure_mix: expected parts, a partition_doc result, or a sequence of "
        f"labels; got {type(parts).__name__}"
    )


def structure_mix(parts: Any, *, total_parts: int | None = None) -> dict[str, Any]:
    """Share of each part type over one document's part list.

    ``total_parts`` lets a caller use the **untrimmed** part count from
    ``partition_doc`` when the inline ``parts`` list has been capped. Passing it
    is the difference between an honest mix and one computed over the payload
    that happened to fit.
    """
    labels = [_part_type(part) for part in _as_parts(parts)]
    counted = sum(1 for _ in labels)
    denominator = total_parts if total_parts is not None else counted
    if denominator < counted:
        raise ValueError(
            f"structure_mix: total_parts={denominator} is smaller than the "
            f"{counted} parts supplied; the denominator must cover what was counted"
        )

    counts = Counter(labels)
    mix = {label: share(count, denominator) for label, count in counts.most_common()}
    prose = sum(count for label, count in counts.items() if label in _PROSE)
    dominant = counts.most_common(1)[0][0] if counts else None

    return measurement(
        value=share(prose, denominator),
        n=denominator,
        unit_notes="share of parts carrying prose (NarrativeText or ListItem)",
        narrative_share=share(prose, denominator),
        mix=mix,
        part_type_counts=dict(counts.most_common()),
        dominant_type=dominant,
        picture_share=share(counts.get("Picture", 0), denominator),
        table_share=share(counts.get("Table", 0), denominator),
        title_share=share(counts.get("Title", 0), denominator),
        caption_share=share(counts.get("FigureCaption", 0), denominator),
        section_header_share=share(counts.get("SectionHeader", 0), denominator),
        counted_parts=counted,
        denominator_mismatch=denominator != counted,
    )
