"""``pdf_forensics`` — ``pdfinfo`` + ``pdffonts`` + ``pdfimages -list`` in one call.

Tier 0, exhaustive, free. **Zero fonts means no text layer**, which is the
single most decisive cheap answer in the whole ladder: a scanned dataset is
disqualified here, before any paid tier is reached.

So the font count is a **first-class field**, not one number in a blob —
``font_count``, ``has_text_layer`` and ``documents_without_text_layer`` are all
directly testable without parsing prose.

The three parsers are separated from the subprocess calls so they can be tested
against captured output on a machine where poppler is not installed. That is
the contract's own consequence (CI's build step installs the Python package but
not the apt belt), and splitting the parse is the honest way to keep coverage.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from data_scout.tools._common import (
    capped,
    iter_files,
    measurement,
    require_binary,
    share,
)

_MAX_DETAIL = 50

_DASHES_RE = re.compile(r"^\s*-{3,}(\s+-{3,})+")
_EMB_RE = re.compile(
    r"^(?P<name>\S+)\s+(?P<type>.+?)\s+(?P<encoding>\S+)\s+"
    r"(?P<emb>yes|no)\s+(?P<sub>yes|no)\s+(?P<uni>yes|no)\b",
    re.IGNORECASE,
)


def _column_spans(dash_line: str) -> list[tuple[int, int]]:
    """Column start/end offsets from a poppler table's dashed rule line."""
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"-+", dash_line):
        start = match.start()
        end = match.end()
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return spans


def _slice(line: str, spans: list[tuple[int, int]]) -> list[str]:
    return [line[start:end].strip() for start, end in spans]


def parse_pdffonts(output: str) -> dict[str, Any]:
    """Parse ``pdffonts`` output into a font list and the counts derived from it."""
    fonts: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] | None = None
    for line in output.splitlines():
        if not line.strip():
            continue
        if _DASHES_RE.match(line):
            spans = _column_spans(line)
            continue
        if line.lstrip().lower().startswith("name"):
            continue
        if spans and len(spans) >= 6:
            cells = _slice(line, spans)
            if len(cells) < 6:
                continue
            fonts.append(
                {
                    "name": cells[0],
                    "type": cells[1],
                    "encoding": cells[2],
                    "embedded": cells[3].lower().startswith("y"),
                    "subset": cells[4].lower().startswith("y"),
                    "unicode": cells[5].lower().startswith("y"),
                }
            )
            continue
        match = _EMB_RE.match(line)
        if match:
            fonts.append(
                {
                    "name": match.group("name"),
                    "type": match.group("type").strip(),
                    "encoding": match.group("encoding"),
                    "embedded": match.group("emb").lower() == "yes",
                    "subset": match.group("sub").lower() == "yes",
                    "unicode": match.group("uni").lower() == "yes",
                }
            )
    types = sorted({font["type"] for font in fonts if font["type"]})
    return {
        "font_count": len(fonts),
        "fonts": fonts,
        "font_types": types,
        "embedded_fonts": sum(1 for font in fonts if font["embedded"]),
        "unicode_fonts": sum(1 for font in fonts if font["unicode"]),
    }


def parse_pdfinfo(output: str) -> dict[str, Any]:
    """Parse ``pdfinfo``'s ``Key:        value`` lines."""
    info: dict[str, Any] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if not key:
            continue
        if key == "pages":
            info["pages"] = int(value) if value.isdigit() else None
        elif key in {"producer", "creator", "title", "author", "subject", "keywords"}:
            info[key] = value or None
        elif key == "pdf version":
            info["pdf_version"] = value or None
        elif key == "encrypted":
            info["encrypted"] = value.lower().startswith("y")
        elif key == "page size":
            info["page_size"] = value or None
        elif key in {"creationdate", "moddate"}:
            info[key] = value or None
    return info


def parse_pdfimages(output: str) -> dict[str, Any]:
    """Parse ``pdfimages -list`` into an image count and a type histogram.

    A page render is an image; a plate is an image; a chart is an image. This
    instrument cannot tell them apart — that is ``image_inventory``'s job — but
    it can say how many there are and in what encodings.
    """
    rows: list[dict[str, Any]] = []
    header: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or _DASHES_RE.match(line):
            continue
        if stripped.lower().startswith("page"):
            header = stripped.split()
            continue
        cells = stripped.split()
        if len(cells) >= 3 and cells[0].isdigit():
            row: dict[str, Any] = {"page": int(cells[0]), "num": cells[1], "type": cells[2]}
            if "width" in header and "height" in header:
                try:
                    row["width"] = int(cells[header.index("width")])
                    row["height"] = int(cells[header.index("height")])
                except (ValueError, IndexError):
                    pass
            if "color" in header:
                try:
                    row["color"] = cells[header.index("color")]
                except IndexError:
                    pass
            if "enc" in header:
                try:
                    row["encoding"] = cells[header.index("enc")]
                except IndexError:
                    pass
            rows.append(row)
    encodings: dict[str, int] = {}
    for row in rows:
        key = str(row.get("encoding") or row.get("type") or "unknown")
        encodings[key] = encodings.get(key, 0) + 1
    return {"image_count": len(rows), "images": rows, "image_encodings": encodings}


def _run(binary: str, args: list[str]) -> str:
    try:
        completed = subprocess.run(  # noqa: S603 - argv list, no shell
            [binary, *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"pdf_forensics: {binary} failed: {exc}") from exc
    # pdfinfo and pdffonts both write their diagnostics to stderr and still exit 0.
    return completed.stdout


def _forensics_one(path: Path) -> dict[str, Any]:
    pdfinfo = require_binary("pdf_forensics", "pdfinfo", package="poppler-utils")
    pdffonts = require_binary("pdf_forensics", "pdffonts", package="poppler-utils")
    pdfimages = require_binary("pdf_forensics", "pdfimages", package="poppler-utils")

    record: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size}
    info = parse_pdfinfo(_run(pdfinfo, [str(path)]))
    record.update({key: value for key, value in info.items()})

    fonts = parse_pdffonts(_run(pdffonts, [str(path)]))
    record.update(
        {
            "font_count": fonts["font_count"],
            "has_text_layer": fonts["font_count"] > 0,
            "embedded_fonts": fonts["embedded_fonts"],
            "font_types": fonts["font_types"],
            "fonts": fonts["fonts"][:10],
        }
    )

    images = parse_pdfimages(_run(pdfimages, ["-list", str(path)]))
    record.update(
        {
            "image_count": images["image_count"],
            "image_encodings": images["image_encodings"],
            "images_per_page": round(
                share(images["image_count"], info.get("pages") or 0), 6
            ),
        }
    )
    # A source that is images and no fonts is a scan: it is identified, and it
    # is not profiled past this instrument (design §11.3).
    record["looks_scanned"] = record["font_count"] == 0 and images["image_count"] > 0
    return record


def pdf_forensics(
    path: str | os.PathLike[str],
    *,
    max_files: int | None = None,
) -> dict[str, Any]:
    """Forensics over one PDF or every PDF under ``path``.

    ``value`` is the count of documents **without a text layer** — the decisive
    number. ``n`` is the number of PDFs examined. ``font_count`` is the total
    across those documents, and each document's own ``font_count`` is in the
    (capped) ``documents`` list.
    """
    suffixes = (".pdf",)
    target = Path(path)
    pdfs = list(iter_files(target, suffixes=None if target.suffix.lower() == ".pdf" else suffixes, limit=max_files))
    if not pdfs:
        return measurement(
            value=0,
            n=0,
            unit_notes="documents without a text layer",
            font_count=0,
            documents=[],
            documents_truncated=False,
            documents_without_text_layer=0,
            documents_looking_scanned=0,
            encrypted=0,
            pages_total=0,
        )

    documents: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for item in pdfs:
        try:
            documents.append(_forensics_one(item))
        except Exception as exc:  # noqa: BLE001 - a broken PDF is a finding, not a crash
            failures.append({"path": str(item), "error": str(exc)[:300]})

    without_text = [doc for doc in documents if doc.get("font_count", 0) == 0]
    detail, trimmed = capped(documents, _MAX_DETAIL)
    return measurement(
        value=len(without_text),
        n=len(pdfs),
        unit_notes="documents without a text layer",
        font_count=sum(doc.get("font_count", 0) for doc in documents),
        documents_without_text_layer=len(without_text),
        documents_looking_scanned=sum(1 for doc in documents if doc.get("looks_scanned")),
        without_text_layer_share=share(len(without_text), len(documents)),
        encrypted=sum(1 for doc in documents if doc.get("encrypted")),
        pages_total=sum(doc.get("pages") or 0 for doc in documents),
        documents=detail,
        documents_truncated=trimmed,
        failures=failures[:10],
    )
