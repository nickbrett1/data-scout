"""The observation cache — a decorator, not a subsystem.

``cache/`` holds **observations**, never measurements:

================  ==================================================  ==========
                  examples                                            policy
================  ==================================================  ==========
observation       Docling part lists, embeddings, image phashes      cached on disk
measurement       structure_mix(parts), candidate_density(text)      recomputed
================  ==================================================  ==========

So a re-executed notebook genuinely recomputes every number in
``findings.jsonl``; only the expensive observation comes from disk. There is no
way for a cached value to masquerade as a fresh measurement.

The cache is what makes the reproducibility guarantee **affordable**: without
it, "re-execute the session notebook and get the same findings" means re-running
Docling over 1,293 PDFs, so it is never actually run.

Two rules on the key, both measurement-drift defences:

1. ``tool_version`` is part of the key, or a stale hit reproduces an old number
   under a new name.
2. For ``purity: network`` tools the **model version goes inside**
   ``tool_version``, because Docling's layout model changes between releases and
   part-type shares move with it.

``rm -rf cache/`` is the entire recovery procedure and must always be safe.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import diskcache

from data_scout import workspace

#: Set to a path to override the per-dataset cache location (tests, scratch runs).
CACHE_DIR_ENV = "DATA_SCOUT_CACHE_DIR"

_CHUNK = 1024 * 1024


class UncacheableInput(TypeError):
    """The decorator was pointed at something it cannot fingerprint."""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(obj: Any) -> str:
    """The ``sha256(input)`` half of the key.

    A path fingerprints its **content**, not its name: the same bytes under a
    different mount are the same observation, and the same name pointing at new
    bytes is a different one.
    """
    if isinstance(obj, Path):
        obj = str(obj)
    if isinstance(obj, str):
        candidate = Path(obj)
        if len(obj) < 4096 and candidate.is_file():
            return "file:" + _hash_file(candidate)
        return "str:" + hashlib.sha256(obj.encode("utf-8")).hexdigest()
    if isinstance(obj, bytes | bytearray):
        return "bytes:" + hashlib.sha256(bytes(obj)).hexdigest()
    if isinstance(obj, int | float | bool) or obj is None:
        return "lit:" + json.dumps(obj, sort_keys=True)
    if isinstance(obj, dict):
        parts = [f"{fingerprint(k)}={fingerprint(v)}" for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))]
        return "map:" + hashlib.sha256("|".join(parts).encode()).hexdigest()
    if isinstance(obj, list | tuple):
        parts = [fingerprint(item) for item in obj]
        return "seq:" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    if isinstance(obj, set | frozenset):
        parts = sorted(fingerprint(item) for item in obj)
        return "set:" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    raise UncacheableInput(
        f"cannot fingerprint {type(obj).__name__}; pass a path, str, bytes, or a "
        "plain container of those"
    )


def cache_directory() -> Path:
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override)
    return workspace.cache_dir()


def cache_key(*, tool: str, tool_version: str, args: tuple, kwargs: dict[str, Any]) -> str:
    """``(sha256(input), kwargs, tool_version)`` — plus the tool name."""
    payload = json.dumps(
        {
            "tool": tool,
            "tool_version": tool_version,
            "args": [fingerprint(arg) for arg in args],
            "kwargs": {str(k): fingerprint(v) for k, v in sorted(kwargs.items())},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cached(
    *,
    tool: str,
    tool_version: str,
    key_fn: Callable[..., Any] | None = None,
    annotate: bool = True,
) -> Callable:
    """Cache an observation, keyed on its inputs.

    Usage::

        @cached(tool="partition_doc", tool_version="1+docling-1.38.1")
        def partition_doc(path, **kwargs): ...

    ``key_fn(args, kwargs)`` may replace the default input fingerprint. When the
    wrapped function returns a ``dict`` and ``annotate`` is true, the result
    gains ``observation_source`` (``"cache"`` or ``"fetched"``) and
    ``cache_key`` — so "same number, same bytes" stays distinguishable from
    "same number, new run", which is the honest half of the two-grade
    reproduction claim.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if key_fn is not None:
                key = str(key_fn(*args, **kwargs))
            else:
                key = cache_key(tool=tool, tool_version=tool_version, args=args, kwargs=kwargs)
            directory = cache_directory()
            directory.mkdir(parents=True, exist_ok=True)
            with diskcache.Cache(str(directory)) as store:
                if key in store:
                    result, source = store[key], "cache"
                else:
                    result, source = func(*args, **kwargs), "fetched"
                    store[key] = result
            if annotate and isinstance(result, dict):
                result = dict(result)
                result.setdefault("observation_source", source)
                result.setdefault("cache_key", key)
            return result

        wrapper.cache_key = lambda *a, **kw: (  # type: ignore[attr-defined]
            str(key_fn(*a, **kw)) if key_fn else cache_key(
                tool=tool, tool_version=tool_version, args=a, kwargs=kw
            )
        )
        wrapper.cache_clear = lambda: clear_cache()  # type: ignore[attr-defined]
        wrapper.tool = tool  # type: ignore[attr-defined]
        wrapper.tool_version = tool_version  # type: ignore[attr-defined]
        return wrapper

    return decorator


def clear_cache(directory: str | Path | None = None) -> None:
    """Delete the cache. The whole recovery procedure, and always safe.

    Nothing a finding needs is stored only here, so this can never lose a
    number — only the work of computing an observation again.
    """
    import shutil

    target = Path(directory) if directory is not None else cache_directory()
    if target.exists():
        shutil.rmtree(target)
