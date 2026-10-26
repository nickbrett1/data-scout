"""Text primitives shared by the content instruments. No measurement lives here."""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")
_ALPHA_RE = re.compile(r"[A-Za-z]")


def tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.findall(text) if part.strip()]


def char_entropy(text: str) -> float:
    """Shannon entropy in bits per character. ~4.5 is English prose."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def non_alpha_ratio(text: str) -> float:
    """Share of non-whitespace characters that are not letters.

    The cheapest OCR-noise signal there is: a clean page sits near 0.1, a
    garbled scan walks up past 0.25.
    """
    dense = [char for char in text if not char.isspace()]
    if not dense:
        return 0.0
    return 1.0 - (len(_ALPHA_RE.findall(text)) / len(dense))


def dictionary_hit_rate(words: list[str], *, minimum_zipf: float = 3.0) -> float:
    """Share of tokens that are ordinary words in some language.

    ``wordfreq`` is the dictionary. A threshold of zipf 3.0 is roughly "occurs
    at least once per million words", which keeps names and technical terms and
    drops OCR noise.
    """
    if not words:
        return 0.0
    from wordfreq import zipf_frequency

    hits = sum(1 for word in words if max(zipf_frequency(word.lower(), lang) for lang in _LANGS) >= minimum_zipf)
    return hits / len(words)


#: Candidate languages for the cheap identify step. Deliberately short: this is
#: a scouting instrument, not a language detector with a model behind it.
_LANGS = ("en", "de", "fr", "es", "it", "nl", "pt", "sv", "ru", "pl")


def identify_language(words: list[str]) -> tuple[str, float]:
    """Best language and a 0..1 confidence, from ``wordfreq`` support alone.

    ``confidence`` is the lead over the runner-up, normalised: it answers "is
    this decisively one language" rather than "how sure is a model".
    """
    if not words:
        return "und", 0.0
    from wordfreq import zipf_frequency

    sample = [word.lower() for word in words[:4000]]
    scores: dict[str, float] = {}
    for lang in _LANGS:
        hits = sum(1 for word in sample if zipf_frequency(word, lang) >= 3.0)
        scores[lang] = hits / len(sample)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score < 0.2:
        return "und", 0.0
    return best, max(0.0, min(1.0, (best_score - runner_up) / best_score))


def mtld(words: list[str], *, threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity — length-invariant, unlike TTR.

    Standard forward+reverse mean, which is what ``textdescriptives`` reports
    and what a corpus comparison needs: raw type/token ratio falls with length,
    so it cannot compare a 200-word page with a 200,000-word book.
    """
    if len(words) < 10:
        return 0.0

    def _one_pass(sequence: list[str]) -> float:
        factors = 0.0
        types: set[str] = set()
        count = 0
        for word in sequence:
            count += 1
            types.add(word.lower())
            ttr = len(types) / count
            if ttr <= threshold:
                factors += 1
                types.clear()
                count = 0
        if count:
            partial = len(types) / count
            remaining = (1.0 - partial) / (1.0 - threshold) if threshold < 1 else 1.0
            factors += max(0.0, min(1.0, remaining))
        return len(sequence) / factors if factors else float(len(sequence))

    return (_one_pass(words) + _one_pass(list(reversed(words)))) / 2.0


_VOWELS = "aeiouy"


def syllables(word: str) -> int:
    """A vowel-group counter with the usual English corrections.

    Used only for Flesch reading ease, which is a coarse instrument anyway; the
    point is that it is deterministic and needs no model.
    """
    lowered = word.lower().strip("'\"-")
    if not lowered:
        return 0
    count = 0
    previous_was_vowel = False
    for char in lowered:
        is_vowel = char in _VOWELS
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel
    if lowered.endswith("e") and not lowered.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(1, count)


def flesch_reading_ease(words: list[str], sentence_count: int) -> float:
    """206.835 - 1.015 words/sentence - 84.6 syllables/word. 0-100, higher = easier."""
    if not words or sentence_count <= 0:
        return 0.0
    total_syllables = sum(syllables(word) for word in words)
    score = 206.835 - 1.015 * (len(words) / sentence_count) - 84.6 * (total_syllables / len(words))
    return max(0.0, min(121.22, score))
