"""``embed_sample`` — stratified sample -> embeddings via the LiteLLM gateway.

Tier 2, network, **paid and therefore an explicit, budgeted turn**. The spend
ceiling per session is zero by default (``data-scout-recipe`` §2), and this is
the instrument that would break it.

It fails loudly, never silently. When the ``[semantic]`` extra is absent or no
gateway key is present, it raises naming what is missing and the install
command. A measurement that reports "no clusters" because a package is missing
is worse than one that refuses, because the report will believe it.

DeepSeek is not an embedder; Cohere is. The model is a parameter with no
default guess for that reason — the wrong model id here is a silent change of
what the number means.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from data_scout.cache import cached
from data_scout.tools._common import ToolUnavailable, measurement, require_import

BASE_URL_ENV = "LITELLM_BASE_URL"
API_KEY_ENV = "LITELLM_API_KEY"
MODEL_ENV = "DATA_SCOUT_EMBED_MODEL"

_MAX_INLINE_VECTORS = 500


def _gateway() -> tuple[str, str]:
    base = (os.environ.get(BASE_URL_ENV) or "").strip()
    key = (os.environ.get(API_KEY_ENV) or "").strip()
    if not base or not key:
        raise ToolUnavailable(
            "embed_sample",
            f"an embedding gateway ({BASE_URL_ENV} and {API_KEY_ENV})",
            "Tier 2 is an explicit, budgeted turn: embedding credentials are not "
            "present in the container until it is taken. Set both variables, and "
            "remember the session's spend ceiling is zero by default.",
        )
    return base.rstrip("/"), key


def _embeddings_endpoint(base: str) -> str:
    return base + "/embeddings" if base.endswith("/v1") else base + "/v1/embeddings"


def _post(url: str, key: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    httpx = require_import("embed_sample", "httpx", extra="dev", packages="httpx")
    try:
        response = httpx.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise ToolUnavailable(
            "embed_sample",
            f"a reachable embedding gateway at {url}",
            f"The request failed ({type(exc).__name__}: {exc}).",
        ) from exc
    if response.status_code >= 400:
        raise RuntimeError(
            f"embed_sample: gateway returned {response.status_code} for model "
            f"{payload.get('model')!r}: {response.text[:400]}"
        )
    return response.json()


@cached(tool="embed_sample", tool_version="1")
def embed_sample(
    texts: Sequence[str],
    *,
    model: str | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Embed ``texts`` and return the vectors.

    ``value`` is the embedding dimension, ``n`` the number of texts. The vectors
    are returned (``cluster_topics`` needs them) but capped inline: the raw
    payload rule applies to a notebook exactly as it applies to a report.
    """
    require_import(
        "embed_sample",
        "numpy",
        extra="semantic",
        packages="numpy, scipy, scikit-learn, umap-learn, hdbscan, bertopic",
    )
    base, key = _gateway()
    model_id = model or os.environ.get(MODEL_ENV, "").strip()
    if not model_id:
        raise ToolUnavailable(
            "embed_sample",
            f"an embedding model id (parameter `model` or {MODEL_ENV})",
            "There is no default: the model id decides what the number means. "
            "Name one deliberately (a real embedder — Cohere, Jina CLIP — not a chat model).",
        )
    if not texts:
        raise ValueError("embed_sample: no texts to embed")

    url = _embeddings_endpoint(base)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        payload = _post(url, key, {"model": model_id, "input": batch})
        data = payload.get("data") or []
        if len(data) != len(batch):
            raise RuntimeError(
                f"embed_sample: gateway returned {len(data)} embeddings for {len(batch)} inputs"
            )
        for item in sorted(data, key=lambda entry: entry.get("index", 0)):
            vectors.append(list(item.get("embedding") or []))

    dimension = len(vectors[0]) if vectors else 0
    if dimension == 0:
        raise RuntimeError("embed_sample: the gateway returned empty embeddings")
    if len({len(vector) for vector in vectors}) != 1:
        raise RuntimeError("embed_sample: inconsistent embedding dimensions across batches")

    return measurement(
        value=dimension,
        n=len(vectors),
        unit_notes="embedding dimension",
        dim=dimension,
        model=model_id,
        vectors=vectors[:_MAX_INLINE_VECTORS],
        vectors_truncated=len(vectors) > _MAX_INLINE_VECTORS,
        ids=list(range(len(vectors))),
    )
