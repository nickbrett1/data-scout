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
from pathlib import Path
from typing import Any

from data_scout.cache import cached
from data_scout.tools._common import ToolUnavailable, capped, measurement

DOCLING_URL_ENV = "DOCLING_URL"
DOCLING_MODEL_VERSION_ENV = "DOCLING_MODEL_VERSION"
DEFAULT_DOCLING_URL = "http://docling:5001"

_MAX_INLINE_PARTS = 200
_MAX_PART_CHARS = 600

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


def _convert(path: Path) -> dict[str, Any]:
    """POST one file to docling-serve and return the raw response body."""
    httpx = _require_httpx()
    base = docling_url()
    url = f"{base}/v1/convert/file"
    data = {
        "to_formats": "json",
        "do_ocr": "false",
        "image_export_mode": "placeholder",
        "table_mode": "fast",
    }
    try:
        with path.open("rb") as handle:
            response = httpx.post(
                url,
                files={"files": (path.name, handle, "application/octet-stream")},
                data=data,
                timeout=httpx.Timeout(600.0, connect=15.0),
            )
    except httpx.HTTPError as exc:
        raise ToolUnavailable(
            "partition_doc",
            f"a reachable Docling service at {base}",
            f"The request failed ({type(exc).__name__}: {exc}). Check DOCLING_URL and "
            "that docling-serve is up; do not fall back to a local model.",
        ) from exc
    if response.status_code >= 400:
        raise RuntimeError(
            f"partition_doc: docling-serve returned {response.status_code} for "
            f"{path.name}: {response.text[:400]}"
        )
    payload = response.json()
    status = payload.get("status")
    if status and status != "success":
        raise RuntimeError(f"partition_doc: docling-serve reported {status!r}: {payload.get('errors')}")
    return payload


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


@cached(tool="partition_doc", tool_version=f"1+docling-model:{_model_version()}")
def partition_doc(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Partition one document into layout-aware parts.

    ``value`` is the number of parts; ``n`` is the same, because the unit *is*
    the part. ``parts`` is capped: an inline aggregate and a small sample, never
    the raw payload (that is what ``cache/`` is for).
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"partition_doc: not a file: {target}")

    payload = _convert(target)
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
    inline, trimmed = capped(parts, _MAX_INLINE_PARTS)

    return measurement(
        value=len(parts),
        n=len(parts),
        unit_notes="parts",
        part_type_counts=counts,
        parts=inline,
        parts_truncated=trimmed,
        origin=origin,
        doc_name=document.get("filename") or target.name,
        docling_url=docling_url(),
        model_version=_model_version(),
    )
