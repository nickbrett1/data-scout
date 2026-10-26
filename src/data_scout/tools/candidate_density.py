"""``candidate_density`` — reader's quote filter, used as a measuring instrument.

**Ported into ``data_scout``, never imported from ``reader``.** The one-way
import rule in ``data-scout-convention`` §5 is the reason: a consumer may import
data-scout's registry; data-scout may not import any consumer. Reader's filter
is the *specification* — the logic is wanted, the dependency is a defect.

Provenance: ported 2026-10-26 from ``reader``'s ``quotes.py`` candidate filter.
The reader source was not present in the data-scout workspace, so the signals
below are reimplemented from the description in ``data-scout-design`` §2.2 and
``data-scout-convention`` §5 rather than transcribed line for line. **Diff this
against ``reader/quotes.py`` deliberately the next time both repos are at
hand** — two copies of one rule drift silently, and the docstring is the only
thing that makes the drift visible.

What the filter encodes, per the design: what a TOC rail, an all-caps heading
and an OCR-noisy sentence look like. The output is a *density*, not a score:
candidates per 1,000 characters, with the reject reasons counted, because "0.4
per 1k because 60% was rejected as TOC" is a different finding from "0.4 per 1k
because the prose is thin".
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any

from data_scout.tools import _text
from data_scout.tools._common import as_texts, capped, measurement, share

_DOTS_RE = re.compile(r"\.{3,}|…{2,}")
_PAGE_REF_RE = re.compile(r"(?:\bpage\b|\bpp?\.)\s*\d+|\s\d{1,4}\s*$", re.IGNORECASE)
_TOC_HEAD_RE = re.compile(
    r"^(contents|table of contents|index|chapter|part|appendix|list of (figures|tables|plates)|"
    r"bibliography|glossary|acknowledge?ments?|preface|foreword|introduction)\b[\s:.\d]*$",
    re.IGNORECASE,
)
_NUMBERED_HEAD_RE = re.compile(r"^(chapter|part|section|figure|plate|table)\s+[ivxlcdm\d]+", re.IGNORECASE)

_MAX_EXAMPLES = 10


def _classify(line: str, *, check_dictionary: bool) -> str | None:
    """Return a reject reason, or ``None`` when the line is a candidate."""
    stripped = line.strip()
    if not stripped:
        return "empty"
    compact = re.sub(r"\s+", " ", stripped)
    letters = [char for char in compact if char.isalpha()]
    digits = [char for char in compact if char.isdigit()]
    length = len(compact)

    if length < 25:
        return "too_short"
    if _TOC_HEAD_RE.match(compact) or _NUMBERED_HEAD_RE.match(compact):
        return "toc_rail" if _TOC_HEAD_RE.match(compact) else "heading"
    if _DOTS_RE.search(compact) or _PAGE_REF_RE.search(compact):
        return "toc_rail"
    if share(len(digits), length) > 0.20:
        return "numeric"

    upper = sum(1 for char in letters if char.isupper())
    if letters and share(upper, len(letters)) > 0.70:
        return "all_caps"

    if _text.non_alpha_ratio(compact) > 0.25:
        return "ocr_noise"
    if len(_text.tokens(compact)) < 5:
        return "too_few_words"
    if check_dictionary and _text.dictionary_hit_rate(_text.tokens(compact)) < 0.5:
        return "low_dictionary_hit"
    return None


def candidate_density(
    source: Any,
    *,
    per_chars: int = 1000,
    min_line_chars: int = 25,
    max_examples: int = _MAX_EXAMPLES,
    check_dictionary: bool = True,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Candidates per ``per_chars`` characters of text.

    ``value`` is the density, ``n`` is the number of characters examined (the
    denominator, so a caller can weight corpora of different sizes).

    ``check_dictionary`` costs a ``wordfreq`` lookup per token. Leave it on for
    samples — the phase-5 rule is *sample, then read* — and turn it off only
    when censusing a corpus exhaustively, recording that in ``params``.
    """
    texts = as_texts(source)
    if max_chars is not None:
        texts = [text[:max_chars] for text in texts]

    candidates: list[str] = []
    rejects: Counter[str] = Counter()
    lines = 0
    chars = 0
    for text in texts:
        chars += len(text)
        for line in text.splitlines():
            if not line.strip():
                continue
            lines += 1
            reason = _classify(line, check_dictionary=check_dictionary)
            if reason is None:
                candidates.append(re.sub(r"\s+", " ", line.strip()))
            else:
                rejects[reason] += 1

    density = (len(candidates) / (chars / per_chars)) if chars else 0.0
    examples, trimmed = capped(candidates, max_examples)
    return measurement(
        value=density,
        n=chars,
        unit_notes=f"candidate lines per {per_chars} characters",
        candidate_count=len(candidates),
        candidate_share=share(len(candidates), lines),
        line_count=lines,
        chars=chars,
        per_chars=per_chars,
        check_dictionary=check_dictionary,
        rejected=dict(rejects.most_common()),
        rejected_share=share(sum(rejects.values()), lines),
        candidates=examples,
        candidates_truncated=trimmed,
        texts=len(texts),
    )


def candidate_density_path(path: str | os.PathLike[str], **kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper: one file, one density."""
    return candidate_density(path, **kwargs)
