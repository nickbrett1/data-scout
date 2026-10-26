"""``image_inventory`` — zero-shot tag counts against a short probe list.

Tier 2, network, the only cheap route to "is that a chart or a plate?" (design
§2.2's open gap). Below tier 2 the image lens can say images exist, and cannot
say what they are: ``Picture`` is an opaque bucket covering art, diagram, plate,
chart and screenshot alike.

The method is cosine against a handful of probe phrases in one shared image
embedding space, which is an image inventory for the cost of ~10 embeddings —
far better than trusting a ``kind='rendered'`` column.

Fails loudly without the vision embedder: there is no local fallback, because a
plausible-looking tag histogram from an absent model is exactly the failure
mode tier 2 is most prone to.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from data_scout.cache import cached
from data_scout.tools._common import (
    ToolUnavailable,
    capped,
    iter_files,
    measurement,
    require_import,
    share,
)
from data_scout.tools.embed_sample import _embeddings_endpoint, _gateway, _post

MODEL_ENV = "DATA_SCOUT_VISION_EMBED_MODEL"

#: The probe list from design §2.2, in the order that matters: the common cases
#: first, so a tie resolves towards the boring answer.
DEFAULT_PROBES: tuple[str, ...] = (
    "a photograph of a person",
    "a photograph of a place",
    "a chart or graph",
    "a diagram",
    "a table of data",
    "a page of text",
    "a pixel art sprite",
    "a screenshot of a game",
    "a logo",
)

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff")
_MAX_PER_IMAGE = 3


def _data_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numpy = require_import("image_inventory", "numpy", extra="semantic", packages="numpy")
    a = numpy.asarray(left, dtype=float)
    b = numpy.asarray(right, dtype=float)
    denominator = float(numpy.linalg.norm(a) * numpy.linalg.norm(b))
    return float(a @ b / denominator) if denominator else 0.0


@cached(tool="image_inventory", tool_version="1")
def image_inventory(
    source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    *,
    probes: Sequence[str] = DEFAULT_PROBES,
    model: str | None = None,
    threshold: float = 0.25,
    max_images: int = 200,
) -> dict[str, Any]:
    """Tag a sample of images against ``probes`` and count the tags.

    ``value`` is the share of images that matched any probe above ``threshold``;
    ``n`` is the number of images examined. ``tag_counts`` is how many images
    had each probe as their best match.
    """
    require_import("image_inventory", "PIL", extra="semantic", packages="pillow")
    numpy = require_import("image_inventory", "numpy", extra="semantic", packages="numpy")

    base, key = _gateway()
    model_id = model or os.environ.get(MODEL_ENV, "").strip()
    if not model_id:
        raise ToolUnavailable(
            "image_inventory",
            f"a vision embedding model id (parameter `model` or {MODEL_ENV})",
            "The probe and the image must be embedded by the same multimodal model, "
            "or the cosine is meaningless. Name one deliberately.",
        )

    if isinstance(source, str | os.PathLike):
        target = Path(source)
        paths = list(
            iter_files(target, suffixes=None if target.is_file() else _IMAGE_SUFFIXES, limit=max_images)
        )
    else:
        paths = [Path(item) for item in source][:max_images]
    if not paths:
        return measurement(
            value=0.0,
            n=0,
            unit_notes="share of images matching any probe above threshold",
            tag_counts={},
            probe_count=len(probes),
            model=model_id,
            threshold=threshold,
            per_image=[],
        )

    url = _embeddings_endpoint(base)
    probe_payload = _post(url, key, {"model": model_id, "input": list(probes)})
    probe_vectors = {
        probe: list(item.get("embedding") or [])
        for probe, item in zip(probes, sorted(probe_payload.get("data") or [], key=lambda entry: entry.get("index", 0)), strict=False)
    }
    if len(probe_vectors) != len(probes):
        raise RuntimeError(
            f"image_inventory: gateway returned {len(probe_vectors)} probe vectors for {len(probes)} probes"
        )

    tag_counts: dict[str, int] = {probe: 0 for probe in probes}
    matched: list[str] = []
    per_image: list[dict[str, Any]] = []
    failed: list[str] = []

    for path in paths:
        try:
            payload = _post(url, key, {"model": model_id, "input": [_data_uri(path)]}, timeout=180.0)
        except Exception as exc:  # noqa: BLE001 - one bad image must not lose the census
            failed.append(f"{path.name}: {type(exc).__name__}: {exc}"[:200])
            continue
        data = payload.get("data") or []
        if not data:
            failed.append(f"{path.name}: gateway returned no embedding")
            continue
        vector = list(data[0].get("embedding") or [])
        scores = {probe: _cosine(vector, probe_vector) for probe, probe_vector in probe_vectors.items()}
        ranked = sorted(scores.items(), key=lambda item: -item[1])[:_MAX_PER_IMAGE]
        best_probe, best_score = ranked[0]
        if best_score >= threshold:
            tag_counts[best_probe] = tag_counts.get(best_probe, 0) + 1
            matched.append(str(path))
        per_image.append(
            {
                "path": str(path),
                "top": [{"probe": probe, "score": round(score, 4)} for probe, score in ranked],
                "matched": best_score >= threshold,
            }
        )

    detail, trimmed = capped(per_image, 50)
    examined = len(per_image)
    return measurement(
        value=share(len(matched), examined),
        n=examined,
        unit_notes="share of images matching any probe above threshold",
        matched=len(matched),
        unmatched=examined - len(matched),
        tag_counts=dict(sorted(tag_counts.items(), key=lambda item: -item[1])),
        probes=list(probes),
        probe_count=len(probes),
        model=model_id,
        threshold=threshold,
        dim=len(next(iter(probe_vectors.values()), [])),
        per_image=detail,
        per_image_truncated=trimmed,
        failed=len(failed),
        failed_examples=failed[:10],
        numpy_available=bool(numpy),
    )
