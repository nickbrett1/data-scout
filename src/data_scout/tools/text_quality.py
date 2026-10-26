"""``text_quality`` — is this text, and are the characters in words?

Tier 1, pure, free. Five cheap signals, all of them local:

* **dictionary hit rate** (``wordfreq``) — the headline, because "these are
  words" is the question an unknown corpus actually raises;
* **character entropy** — 4.0–4.8 bits/char is prose, 6+ is base64 or a font
  dump, under 3 is a repeated template;
* **non-alpha ratio** — the cheapest OCR-noise signal there is;
* **language ID** — from ``wordfreq`` support, no model;
* **lexical diversity (MTLD) and Flesch reading ease** — computed locally so the
  number is deterministic and needs no model download.

``textdescriptives`` is used **additively** and its absence is recorded rather
than papered over: its extractors need a spaCy pipeline, and a first call will
otherwise try to download one. A measurement that silently reports a default
Flesch score when the model is missing is worse than one that says why it is
absent, so ``textdescriptives_unavailable`` is always present when it is.
"""

from __future__ import annotations

import os
from typing import Any

from data_scout.tools import _text
from data_scout.tools._common import as_texts, measurement, share

_SPACY_MODEL = os.environ.get("DATA_SCOUT_SPACY_MODEL", "en_core_web_sm")


def _textdescriptives(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Optional metrics, or the reason they are absent. Never a silent default."""
    try:
        import spacy.util
    except ImportError:
        return None, "textdescriptives unavailable: spacy is not installed"
    if not spacy.util.is_package(_SPACY_MODEL):
        return (
            None,
            f"textdescriptives unavailable: the spaCy model {_SPACY_MODEL!r} is not "
            f"installed (python -m spacy download {_SPACY_MODEL}); text_quality does "
            "not download models during a measurement",
        )
    try:
        from textdescriptives import extract_metrics

        frame = extract_metrics(text=text[:20000], metrics=["descriptive_stats", "readability", "lexical_diversity"])
        record = frame.iloc[0].to_dict()
    except Exception as exc:  # noqa: BLE001 - optional enrichment, always explained
        return None, f"textdescriptives unavailable: {type(exc).__name__}: {exc}"
    return (
        {
            str(key): (float(value) if isinstance(value, float) else value)
            for key, value in record.items()
            if key != "text"
        },
        None,
    )


def text_quality(
    source: Any,
    *,
    include_textdescriptives: bool = True,
    max_chars: int | None = 200_000,
) -> dict[str, Any]:
    """Quality signals over one text, a path, or a list of either.

    ``value`` is the dictionary hit rate (share of tokens that are ordinary
    words); ``n`` is the character count it was measured over.
    """
    texts = as_texts(source)
    text = "\n".join(texts)
    if max_chars is not None:
        text = text[:max_chars]

    words = _text.tokens(text)
    sentences = _text.sentences(text)
    lowered = [word.lower() for word in words]
    hit_rate = _text.dictionary_hit_rate(words)
    language, confidence = _text.identify_language(words)

    extra: dict[str, Any] = {}
    if include_textdescriptives:
        metrics, reason = _textdescriptives(text)
        extra["textdescriptives"] = metrics
        extra["textdescriptives_unavailable"] = reason

    return measurement(
        value=hit_rate,
        n=len(text),
        unit_notes="share of tokens that are ordinary words (wordfreq zipf >= 3)",
        dictionary_hit_rate=hit_rate,
        entropy_bits_per_char=_text.char_entropy(text),
        non_alpha_ratio=_text.non_alpha_ratio(text),
        language=language,
        language_confidence=confidence,
        token_count=len(words),
        unique_token_count=len(set(lowered)),
        type_token_ratio=share(len(set(lowered)), len(lowered)),
        mtld=_text.mtld(lowered),
        flesch_reading_ease=_text.flesch_reading_ease(words, len(sentences)),
        sentence_count=len(sentences),
        mean_sentence_words=share(len(words), len(sentences)),
        mean_word_length=share(sum(len(word) for word in words), len(words)),
        line_count=len([line for line in text.splitlines() if line.strip()]),
        **extra,
    )
