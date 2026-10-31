"""``partition_doc`` — Docling layout-aware parts, as a library client.

Docling is a **network client, not a local model** (design §2.4.1): the call
lives in a notebook cell, the implementation lives in the sidecar, and the
result is content-addressed on disk. That split is why this is promotable at
all — ``purity: network`` is a purity, not a disqualification.

The observation is cached, keyed on the file's bytes **and on the model
version**, because Docling's layout model changes between releases and
part-type shares move with it: a ``structure_mix`` cached from a different model
version is a different measurement.

Two things this deliberately does not do: it does not OCR, and it does not
inline Docling's images. ``image_export_mode=placeholder`` keeps the payload to
text, because a notebook that inherits megabytes per document produces an HTML
file that will not open on a phone.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from data_scout.cache import cached, read_entry
from data_scout.tools._common import (
    ToolUnavailable,
    absent_is_not_zero,
    capped,
    measurement,
)

DOCLING_URL_ENV = "DOCLING_URL"
DOCLING_MODEL_VERSION_ENV = "DOCLING_MODEL_VERSION"
DEFAULT_DOCLING_URL = "http://docling:5001"

_MAX_INLINE_PARTS = 200
_MAX_PART_CHARS = 600

#: docling-serve's asynchronous task API. The submit returns a task id, the
#: caller polls it, and the result is fetched once it is complete — so no
#: deployment-specific sync ceiling silently governs the conversion.
_ASYNC_SUBMIT_PATH = "/v1/convert/file/async"
_POLL_PATH = "/v1/status/poll/{task_id}"
_RESULT_PATH = "/v1/result/{task_id}"

#: How long this client is willing to wait for one task. The job grants each
#: document an hour, and an hour is the honest default for a book.
TASK_TIMEOUT_ENV = "DATA_SCOUT_DOCLING_TASK_TIMEOUT"
POLL_INTERVAL_ENV = "DATA_SCOUT_DOCLING_POLL_INTERVAL"
DEFAULT_TASK_TIMEOUT = 3600.0
DEFAULT_POLL_INTERVAL = 2.0
_MAX_POLL_INTERVAL = 5.0
_READ_TIMEOUT = 600.0

#: Task states that mean the result is fetchable, and states that mean it is not.
_COMPLETED_STATES = frozenset({"success", "partial_success"})
_FAILED_STATES = frozenset({"failure", "skipped"})

#: Part labels, normalised to the five names the design uses.
_LABEL_MAP = {
    "title": "Title",
    "section_header": "SectionHeader",
    "text": "NarrativeText",
    "paragraph": "NarrativeText",
    "list_item": "ListItem",
    "caption": "FigureCaption",
    "page_footer": "PageFooter",
    "page_header": "PageHeader",
    "footnote": "Footnote",
    "reference": "Reference",
    "formula": "Formula",
    "code": "Code",
    "table": "Table",
    "picture": "Picture",
}


def docling_url() -> str:
    value = os.environ.get(DOCLING_URL_ENV, "").strip()
    if not value:
        raise ToolUnavailable(
            "partition_doc",
            f"the Docling service URL ({DOCLING_URL_ENV})",
            "Set DOCLING_URL to the running docling-serve, e.g. "
            "http://docling:5001. partition_doc is a client; it does not ship a model.",
        )
    return value.rstrip("/")


def _model_version() -> str:
    """The model version that belongs inside ``tool_version``.

    Unset is ``unknown`` rather than a guess: an unknown model version invalidates
    nothing, while a wrong one silently reuses a cached measurement.
    """
    return os.environ.get(DOCLING_MODEL_VERSION_ENV, "unknown")


def _normalise_label(label: str | None) -> str:
    return _LABEL_MAP.get((label or "").strip().lower(), "Other")


def _part_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""


def parts_from_docling_json(document: dict[str, Any], *, source: str | None = None) -> list[dict[str, Any]]:
    """Flatten Docling's ``json_content`` into an ordered part list.

    Reading order comes from the body tree when it is there, because that is
    what ``partition_doc`` is *for*; when it is absent, the ``texts`` array in
    document order is the fallback and is labelled as such.
    """
    parts: list[dict[str, Any]] = []

    def add(label: str | None, text: str, ref: str | None, page: int | None) -> None:
        parts.append(
            {
                "type": _normalise_label(label),
                "label": label,
                "text": text[:_MAX_PART_CHARS],
                "text_chars": len(text),
                "self_ref": ref,
                "page": page,
            }
        )

    texts = {
        item.get("self_ref"): item
        for item in document.get("texts") or []
        if isinstance(item, dict)
    }
    tables = {
        item.get("self_ref"): item
        for item in document.get("tables") or []
        if isinstance(item, dict)
    }
    pictures = {
        item.get("self_ref"): item
        for item in document.get("pictures") or []
        if isinstance(item, dict)
    }
    body = (document.get("body") or {}).get("children") or []

    def page_of(item: dict[str, Any]) -> int | None:
        prov = item.get("prov") or []
        if prov and isinstance(prov[0], dict):
            page = prov[0].get("page_no")
            return page if isinstance(page, int) else None
        return None

    if body:
        for child in body:
            ref = child.get("$ref") if isinstance(child, dict) else None
            if ref in texts:
                item = texts[ref]
                add(item.get("label"), _part_text(item.get("text")), ref, page_of(item))
            elif ref in tables:
                item = tables[ref]
                add("table", "", ref, page_of(item))
            elif ref in pictures:
                item = pictures[ref]
                add("picture", "", ref, page_of(item))
    else:
        for ref, item in texts.items():
            add(item.get("label"), _part_text(item.get("text")), ref, page_of(item))
        for ref, item in tables.items():
            add("table", "", ref, page_of(item))
        for ref, item in pictures.items():
            add("picture", "", ref, page_of(item))

    if source:
        for part in parts:
            part["source"] = source
    return parts


def _parts_from_markdown(markdown: str) -> list[dict[str, Any]]:
    """Fallback when the sidecar returns markdown only: headings vs paragraphs."""
    parts: list[dict[str, Any]] = []
    for block in [block.strip() for block in markdown.split("\n\n") if block.strip()]:
        label = "section_header" if block.lstrip().startswith("#") else "text"
        text = block.lstrip("# ").strip()
        parts.append(
            {
                "type": _normalise_label(label),
                "label": label,
                "text": text[:_MAX_PART_CHARS],
                "text_chars": len(text),
                "self_ref": None,
                "page": None,
            }
        )
    return parts


def _client(httpx: Any) -> Any:
    """One HTTP client for a whole conversion: submit, poll and result share it."""
    return httpx.Client(timeout=httpx.Timeout(_READ_TIMEOUT, connect=15.0))


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise RuntimeError(f"partition_doc: {name}={raw!r} is not a number of seconds") from None
    if value < 0:
        raise RuntimeError(f"partition_doc: {name}={raw!r} must not be negative")
    return value


def _check(response: Any, what: str) -> None:
    if response.status_code >= 400:
        raise RuntimeError(
            f"partition_doc: docling-serve returned {response.status_code} while "
            f"{what}: {response.text[:400]}"
        )


def _submit(client: Any, base: str, path: Path) -> str:
    """POST the file to the async endpoint and return the task id."""
    data = {
        "to_formats": "json",
        "do_ocr": "false",
        "image_export_mode": "placeholder",
        "table_mode": "fast",
    }
    with path.open("rb") as handle:
        response = client.post(
            f"{base}{_ASYNC_SUBMIT_PATH}",
            files={"files": (path.name, handle, "application/octet-stream")},
            data=data,
        )
    _check(response, f"submitting {path.name}")
    payload = response.json()
    task_id = payload.get("task_id")
    if not task_id:
        raise RuntimeError(
            f"partition_doc: docling-serve accepted the submission but returned no "
            f"task_id: {str(payload)[:400]}"
        )
    return str(task_id)


def _poll(client: Any, base: str, task_id: str, *, deadline: float, timeout: float) -> str:
    """Wait for the task to reach a terminal state and return its status.

    There is deliberately **no retry loop**: a timed-out conversion keeps running
    on the sidecar, so submitting it again would add a second conversion for the
    same bytes and amplify load. Giving up therefore raises with the task id and
    its result URL — the work is still owned and still retrievable, just not this
    call's to wait for.
    """
    interval = _float_env(POLL_INTERVAL_ENV, DEFAULT_POLL_INTERVAL)
    poll_url = f"{base}{_POLL_PATH.format(task_id=task_id)}"
    while True:
        response = client.get(poll_url)
        if response.status_code == 404:
            raise RuntimeError(
                f"partition_doc: docling-serve no longer knows task {task_id} (404 "
                "from the poll endpoint); the sidecar was probably restarted "
                "mid-conversion. Not retrying: the input bytes are unchanged, and a "
                "retry policy belongs to the caller that can see the queue."
            )
        _check(response, f"polling task {task_id}")
        payload = response.json()
        status = str(payload.get("task_status") or "")
        if status in _COMPLETED_STATES:
            return status
        if status in _FAILED_STATES:
            detail = payload.get("failure") or payload.get("error_message") or payload.get("errors")
            raise RuntimeError(
                f"partition_doc: docling-serve task {task_id} ended {status!r}: {detail}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"partition_doc: task {task_id} is still running after {timeout:g}s and "
                "was NOT cancelled. The conversion continues on the sidecar and is not "
                f"orphaned: poll {poll_url} or fetch the result from "
                f"{base}{_RESULT_PATH.format(task_id=task_id)} once it completes. This "
                "belt does not retry, because a retry would submit a second conversion "
                "for the same bytes."
            )
        time.sleep(min(interval, remaining))
        interval = min(interval * 1.5, _MAX_POLL_INTERVAL)


def _result(client: Any, base: str, task_id: str) -> dict[str, Any]:
    """Fetch the completed task's payload."""
    response = client.get(f"{base}{_RESULT_PATH.format(task_id=task_id)}")
    if response.status_code >= 400:
        raise RuntimeError(
            f"partition_doc: could not fetch the result for task {task_id}: "
            f"HTTP {response.status_code}: {response.text[:400]}"
        )
    payload = response.json()
    if isinstance(payload, dict) and isinstance(payload.get("failure"), dict):
        raise RuntimeError(
            f"partition_doc: task {task_id} has no conversion result: {payload['failure']}"
        )
    return payload


def _convert(path: Path) -> tuple[dict[str, Any], str]:
    """Submit one file asynchronously, wait for the task, and return its result.

    The synchronous endpoint this used to call abandons the request at
    ``DOCLING_SERVE_MAX_SYNC_WAIT`` (120 s by default) while the conversion keeps
    running, so every document over ~229 pages got a clean HTTP 504 *and* left an
    unowned conversion competing with the next one. The async endpoint removes
    that ceiling structurally, with no sidecar restart and no memory edit.
    """
    httpx = _require_httpx()
    base = docling_url()
    timeout = _float_env(TASK_TIMEOUT_ENV, DEFAULT_TASK_TIMEOUT)
    deadline = time.monotonic() + timeout
    try:
        with _client(httpx) as client:
            task_id = _submit(client, base, path)
            status = _poll(client, base, task_id, deadline=deadline, timeout=timeout)
            payload = _result(client, base, task_id)
    except httpx.HTTPError as exc:
        raise ToolUnavailable(
            "partition_doc",
            f"a reachable Docling service at {base}",
            f"The request failed ({type(exc).__name__}: {exc}). Check DOCLING_URL and "
            "that docling-serve is up; do not fall back to a local model.",
        ) from exc
    return payload, status


def _require_httpx():
    try:
        import httpx
    except ImportError:  # pragma: no cover - declared dependency
        raise ToolUnavailable(
            "partition_doc",
            "the `httpx` package",
            "pip install httpx (it is a declared dependency of data-scout).",
        ) from None
    return httpx


@cached(tool="partition_doc", tool_version=f"3+docling-model:{_model_version()}")
def _observation(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The **cached** observation: every part, uncapped.

    This is what ``cache/`` holds. ``partition_doc`` is the capped inline view
    of it, and :func:`cached_parts` is the loud reader. The split matters
    because the old cached value was the inline measurement, so a 2,246-part
    document was cached truncated at 200 and a reader over the cache read a
    truncated observation. The cache version moved to 3 with the shape, because
    a v2 entry is not this object.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"partition_doc: not a file: {target}")

    payload, task_status = _convert(target)
    document = payload.get("document") or {}
    json_content = document.get("json_content")
    if isinstance(json_content, str):
        # Some docling-serve versions return the JSON as a string field.
        import json

        try:
            json_content = json.loads(json_content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"partition_doc: json_content is not JSON: {exc}") from exc

    if isinstance(json_content, dict):
        parts = parts_from_docling_json(json_content, source=target.name)
        origin = "docling-json"
    else:
        markdown = document.get("md_content") or ""
        if not markdown:
            raises = document.get("errors") or payload.get("errors") or "no json_content and no md_content"
            raise RuntimeError(f"partition_doc: unusable response for {target.name}: {raises}")
        parts = _parts_from_markdown(markdown)
        origin = "docling-markdown"

    counts: dict[str, int] = {}
    for part in parts:
        counts[part["type"]] = counts.get(part["type"], 0) + 1

    return measurement(
        value=len(parts),
        n=len(parts),
        unit_notes="parts",
        part_type_counts=counts,
        parts=parts,
        origin=origin,
        docling_task_status=task_status,
        doc_name=document.get("filename") or target.name,
        docling_url=docling_url(),
        model_version=_model_version(),
    )


def partition_doc(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Partition one document into layout-aware parts, capped for inline use.

    ``value`` is the number of parts; ``n`` is the same, because the unit *is*
    the part. ``parts`` is capped at 200: an inline aggregate and a small
    sample, never the whole payload. The **full** part list is the cached
    observation, read by :func:`cached_parts`; ``parts_truncated`` says whether
    this view is the whole thing.
    """
    observation = _observation(path)
    inline, trimmed = capped(observation["parts"], _MAX_INLINE_PARTS)
    return measurement(
        value=observation["value"],
        n=observation["n"],
        unit_notes=observation["unit_notes"],
        part_type_counts=observation["part_type_counts"],
        parts=inline,
        parts_truncated=trimmed,
        origin=observation["origin"],
        docling_task_status=observation["docling_task_status"],
        doc_name=observation["doc_name"],
        docling_url=observation["docling_url"],
        model_version=observation["model_version"],
        observation_source=observation.get("observation_source", "fetched"),
        cache_key=observation.get("cache_key"),
    )


def cached_parts(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read the **full** cached observation for ``path``. Never converts.

    The key comes from ``_observation.cache_key`` — the same decorator that
    writes the entry — so this cannot drift from the writer's version (a
    hardcoded "2" or "3" here would silently stop finding entries the next time
    the observation shape changes). It is a reader, not an instrument: it is
    absent from the registry, and it never re-runs, refreshes or converts. A
    session that needs an entry it does not have must call ``partition_doc``
    itself; that is the session's call, not the reader's.

    Returns the same shape ``partition_doc`` returns, except ``parts`` is the
    full list and there is no ``parts_truncated``: the honest denominator is
    ``n`` and it covers exactly what is in ``parts``.
    """
    target = Path(path)
    key = _observation.cache_key(target)
    observation = read_entry(key)
    if observation is None:
        absent_is_not_zero(
            tool="partition_doc",
            what=f"a cached observation for {target.name!r}",
            hint=(
                "Call partition_doc(...) for this file first. The reader never "
                "converts and never refreshes the cache: a missing entry is the "
                "session's call to re-run, not something to paper over with an "
                "empty part list."
            ),
        )
    result = dict(observation)
    result.setdefault("observation_source", "cache")
    result["cache_key"] = key
    return result
