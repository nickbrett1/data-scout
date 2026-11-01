"""``report-render`` — execute, verify, then render. The order is the contract.

This is the piece that makes "no kernel server in the image" a real answer
rather than a downgrade: with ``jupyter-mcp-server`` gone, the agent writes a
notebook with ``nbformat`` and this script is what runs it — returning per-cell
output so failures are visible in the same loop the agent probed in.

The six steps, in order (``data-scout-convention`` §4.3):

1. **Stamp ``metadata.kernelspec``.** ``nbconvert --execute`` does not care,
   which is exactly why its absence went unnoticed — a person does. The fix
   lives here, not only in the writer, because this is the component that runs a
   kernel and knows which one, so it covers hand-edited notebooks too.
2. **Execute the notebook** for the artifact, saving outputs. Its ledger
   appends go to a **staged** copy (``DATA_SCOUT_LEDGER``), never the live
   ``findings.jsonl``: a render writes nothing to the record until both gates
   have passed.
3. **Re-execute into a scratch copy** of the dataset workspace and diff the
   staged ``findings.jsonl`` modulo ``ts``. The verification execution must
   never target the live workspace, or every render appends its own ledger a
   second time, silently, and the append-only ledger stops being a record of
   what the session measured.
4. **Gate the render on that diff, then promote the staged appends.** A failed
   cell or a non-reproducing re-execution leaves ``findings.jsonl`` byte-identical
   and exits non-zero; only a run that passes both gates appends its lines,
   exactly once and byte for byte.
5. **Render** ``session-N.ipynb`` to HTML and ``report-*.md`` to a report
   anchored on the ledger: ``findings.jsonl`` becomes a table of scannable rows
   (number, instrument, value, unit, ``n``, session) and each row opens into the
   commentary the markdown attaches with ``<!-- finding: N -->``. One
   stylesheet, no browser engine, no PDF (a stated non-goal).
6. **Print each cell's output and any error**, so the agent can iterate.

Rendered artifacts are **not** byte-reproducible and must not be tested as if
they were: execution counts, output ids and timestamps vary run to run. The
reproducibility claim is scoped to ``findings.jsonl`` lines, modulo ``ts``.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any

from data_scout import frame as frame_identity
from data_scout import notebook as nbwrite
from data_scout import workspace
from data_scout.ledger import Ledger, diff_lines, normalized_line

_MAX_PRINT_CHARS = 4000


# --------------------------------------------------------------------------
# environment scoping
# --------------------------------------------------------------------------


@contextmanager
def scoped_env(values: dict[str, str]) -> Iterator[None]:
    """Patch the process environment for the duration of a kernel's lifetime.

    The kernel inherits ``os.environ`` when it starts, so this is how the same
    notebook is pointed at the live workspace once and at a scratch copy once.
    """
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def scratch_workspace(source: Path, *, destination: Path) -> Path:
    """Copy a dataset workspace for the verification run, without its cache.

    The cache is *deliberately* not copied: the verification run shares the live
    observation cache, which is what makes a ``purity: network`` tool reproduce
    **exactly from cache** instead of re-fetching. Observations are safe to
    share; measurements are recomputed either way.
    """
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("cache"),
        symlinks=True,
    )
    return destination


# --------------------------------------------------------------------------
# notebook execution
# --------------------------------------------------------------------------


def execute_notebook(
    path: Path,
    *,
    workspace_root: Path,
    dataset_id: str,
    session: int,
    timeout: int = 1800,
) -> Any:
    """Execute one notebook against ``workspace_root``, saving nothing here.

    Errors are allowed to land in the outputs so that step 6 can print them;
    the caller decides what a failed cell means.
    """
    from nbclient import NotebookClient

    notebook = nbwrite.read_notebook(path)
    nbwrite.stamp_kernelspec(notebook)
    kernel_name = notebook.metadata["kernelspec"]["name"]
    env = {
        workspace.WORKSPACE_ENV: str(workspace_root),
        workspace.DATASET_ID_ENV: dataset_id,
        workspace.SESSION_ENV: str(session),
    }
    with scoped_env(env):
        client = NotebookClient(
            copy.deepcopy(notebook),
            timeout=timeout,
            kernel_name=kernel_name,
            resources={"metadata": {"path": str(path.parent)}},
            allow_errors=True,
        )
        executed = client.execute()
    nbwrite.stamp_kernelspec(executed)
    return executed


def cell_errors(notebook: Any) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs") or []:
            if output.get("output_type") == "error":
                found.append(
                    (
                        index,
                        f"{output.get('ename')}: {output.get('evalue')}\n"
                        + "\n".join(output.get("traceback") or []),
                    )
                )
    return found


def print_cell_outputs(notebook: Any, *, stream: Any = None) -> None:
    """Print every code cell's source and output — the agent's inner loop.

    Without this the agent writes notebooks blind and discovers at render time
    that cell 7 raises; with it, a render failure is iterated on exactly like a
    probe.
    """
    out = stream or sys.stdout
    for index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != "code":
            continue
        print(f"\n=== cell {index} ===", file=out)
        print(cell.source, file=out)
        for output in cell.get("outputs") or []:
            kind = output.get("output_type")
            if kind == "stream":
                text = "".join(output.get("text") or "")
                print(_trim(text), file=out)
            elif kind in {"execute_result", "display_data"}:
                data = output.get("data") or {}
                if "text/plain" in data:
                    print(_trim("".join(data["text/plain"])), file=out)
                for mime in data:
                    if mime.startswith("image/"):
                        size = len("".join(data[mime])) if isinstance(data[mime], str) else 0
                        print(f"[{mime} omitted, {size} chars]", file=out)
            elif kind == "error":
                print(f"!! {output.get('ename')}: {output.get('evalue')}", file=out)
                for line in output.get("traceback") or []:
                    print(_trim(_strip_ansi(line)), file=out)


def _trim(text: str) -> str:
    text = text.rstrip("\n")
    if len(text) <= _MAX_PRINT_CHARS:
        return text
    head = text[: _MAX_PRINT_CHARS // 2]
    tail = text[-_MAX_PRINT_CHARS // 2 :]
    return f"{head}\n… [{len(text) - _MAX_PRINT_CHARS} chars omitted] …\n{tail}"


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def stylesheet() -> str:
    """The one CSS file, read from the package so the image and the repo agree."""
    return (resources.files("data_scout") / "styles" / "report.css").read_text(encoding="utf-8")


#: The nav bar is what makes the artifact navigable on a phone: a web app has
#: no browser back button, so the way back has to be on the page. ``__main__``
#: also uses this exact opening tag as the marker for "already has the chrome",
#: which is how a report rendered before it existed still gets the back link.
NAV_MARKER = '<nav class="data-scout-nav">'

#: Marks the stylesheet this app owns inside a document it did not author, so a
#: serve-time splice can *replace* an older copy instead of stacking in front of
#: it: a frozen artifact then follows the current sheet without being rewritten.
STYLE_ATTR = 'data-data-scout=""'
_STYLE_BLOCK = re.compile(r"<style data-data-scout[^>]*>.*?</style>", re.DOTALL)

NAV_HTML = (
    NAV_MARKER
    + '\n<a class="back" href="/"><span aria-hidden="true">\u2190</span> All datasets</a>'
    + '\n<span class="brand">data-scout</span>\n</nav>'
)

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>{title}</title>
<style data-data-scout="">
{css}
</style>
</head>
<body>
{nav}
<div class="data-scout-meta">{meta}</div>
{body}
</body>
</html>
"""


def _plain_text(token: Any) -> str:
    """The visible text of an inline token, for a slug and a contents entry."""
    pieces: list[str] = []
    for child in token.children or []:
        if child.type in {"text", "code_inline"}:
            pieces.append(child.content)
        elif child.type in {"softbreak", "hardbreak"}:
            pieces.append(" ")
    return "".join(pieces).strip()


def _slugify(text: str, seen: dict[str, int]) -> str:
    slug = "".join(char if char.isalnum() else "-" for char in text.lower())
    slug = "-".join(part for part in slug.split("-") if part) or "section"
    count = seen.get(slug, 0)
    seen[slug] = count + 1
    return slug if count == 0 else f"{slug}-{count + 1}"


#: Below this many sections a table of contents is more furniture than aid.
TOC_MIN_SECTIONS = 3


def _structure(tokens: Sequence[Any]) -> tuple[dict[int, str], list[tuple[int, str, str]]]:
    """Heading ids by token index, and the (level, text, id) of each h2.

    Ids are assigned once, in document order, so the anchor a contents entry
    points at is the anchor the heading carries — including the second "Counts"
    section, which becomes ``counts-2`` rather than colliding with the first.
    """
    seen: dict[str, int] = {}
    ids: dict[int, str] = {}
    sections: list[tuple[int, str, str]] = []
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        inline = tokens[index + 1] if index + 1 < len(tokens) else None
        text = _plain_text(inline) if inline is not None else ""
        slug = _slugify(text, seen)
        ids[index] = slug
        if token.tag == "h2" and text:
            sections.append((2, text, slug))
    return ids, sections


def _contents(sections: list[tuple[int, str, str]]) -> str:
    """A contents card for a report long enough to need one."""
    if len(sections) < TOC_MIN_SECTIONS:
        return ""
    items = "".join(
        f'<li><a href="#{_escape(slug)}">{_escape(text)}</a></li>'
        for _level, text, slug in sections
    )
    return (
        '<nav class="data-scout-toc" aria-label="Contents">'
        '<p class="toc-title">Contents</p>'
        f"<ol>{items}</ol></nav>"
    )


#: A report is a table of the ledger with the prose hung off it. The marker
#: below starts the commentary for one finding; the number is a 1-based line in
#: ``findings.jsonl`` for that finding — any of its lines, because a re-append
#: gets the same row as the line it repeats. Everything before the first marker
#: is the report's lede, and stays above the table.
#:
#:     # vg-books-2026-10
#:
#:     Six PDFs. The frame says what the checkout is; this says what is in it.
#:
#:     <!-- finding: 13 -->
#:
#:     Three PDFs are locked, and the same three have text layers that extract
#:     to nothing. That pairing is the finding...
#:
#: A report with no markers still renders — the table comes first and its prose
#: follows — so an older report is not a broken one.
FINDING_MARKER = re.compile(r"^[ \t]*<!--[ \t]*finding:[ \t]*(\d+)[ \t]*-->[ \t]*$", re.MULTILINE)

#: Prose a report needs that belongs to no single finding — the method, the
#: gaps, what would settle them. Without this the last finding would collect
#: everything the author had left to say, which is how a table becomes a wall
#: of text again. Everything after this marker is rendered below the table.
NOTES_MARKER = re.compile(r"^[ \t]*<!--[ \t]*notes[ \t]*-->[ \t]*$", re.MULTILINE)

#: The params worth showing in the row itself, in this order. Everything else
#: is in the row's detail: the row answers "what is this number", the detail
#: answers "how was it measured".
_ROW_PARAMS = ("path", "document", "measure", "attribute", "aggregate", "probe", "suffix")


def split_commentary(markdown: str) -> tuple[str, dict[int, str], str]:
    """The lede, the commentary per finding number, and the trailing notes.

    A block runs from its marker to the next one, so a finding's prose can be
    anything markdown can express — including its own headings and tables —
    without any nesting rule to get wrong. ``<!-- notes -->`` closes the
    per-finding part: it is where the report says things that are true of the
    session rather than of one line.
    """
    markers = [*FINDING_MARKER.finditer(markdown), *NOTES_MARKER.finditer(markdown)]
    markers.sort(key=lambda match: match.start())
    if not markers:
        return markdown, {}, ""
    lede = markdown[: markers[0].start()]
    blocks: dict[int, str] = {}
    notes = ""
    for position, marker in enumerate(markers):
        end = markers[position + 1].start() if position + 1 < len(markers) else len(markdown)
        text = markdown[marker.end() : end].strip()
        if not text:
            continue
        if marker.re is NOTES_MARKER:
            notes = f"{notes}\n\n{text}" if notes else text
            continue
        number = int(marker.group(1))
        blocks[number] = f"{blocks[number]}\n\n{text}" if number in blocks else text
    return lede, blocks, notes


def _format_value(value: Any) -> str:
    if isinstance(value, bool):  # bool is an int; it is never a measurement
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.6g}"
    return str(value)


def _short_value(text: str) -> str:
    """A hash is provenance, not a reading: show its ends, keep it whole in the detail."""
    if len(text) <= 24:
        return text
    return f"{text[:12]}…{text[-4:]}"


def _split_reading(text: str) -> tuple[str, str]:
    """A block's first paragraph is what the row says; the rest is the argument.

    The closed card has to answer "what does this number mean" without being
    opened, so the first paragraph of the commentary is promoted into the row.
    Authoring contract, therefore: lead with the plain reading, then explain.
    """
    paragraph, _, rest = text.partition("\n\n")
    return paragraph.strip(), rest.strip()


def _plain(html: str) -> str:
    """Inline HTML is fine in a row; a block element would break the line."""
    return re.sub(r"</?p>", "", html).strip()


def _shorten(key: str, value: Any) -> str:
    """A row is scanned, not read: a 40-character absolute path is not a label."""
    text = str(value)
    if key == "path":
        parts = [part for part in text.split("/") if part]
        if len(parts) > 2:
            text = "…/" + "/".join(parts[-2:])
    return text if len(text) <= 34 else text[:33] + "…"


def _row_params(params: dict[str, Any]) -> str:
    keys = [key for key in _ROW_PARAMS if key in params]
    keys += [key for key in sorted(params) if key not in keys]
    return " · ".join(f"{key}={_shorten(key, params[key])}" for key in keys[:2])


def _detail_rows(finding: Any, lines: Sequence[int] = ()) -> list[str]:
    """Everything the row does not show: parameters, provenance, the decision."""
    parts = ['<dl class="f-facts">']
    if lines:
        reappends = ", ".join(str(line) for line in lines[1:])
        where = f"line {lines[0]}"
        if reappends:
            where += f" · re-appended at line{', ' if len(lines) > 2 else ''} {reappends}"
        parts.append(f"<dt>ledger</dt><dd>{_escape(where)}</dd>")
    parts.append(
        f'<dt>measured</dt><dd><code>{_escape(finding.tool)}</code> '
        f'v{_escape(finding.tool_version)} · session {finding.session} · '
        f'{_escape(finding.ts.isoformat().replace("+00:00", "Z"))}</dd>'
    )
    if finding.params:
        rendered = ", ".join(
            f"{_escape(str(key))}={_escape(str(value))}" for key, value in sorted(finding.params.items())
        )
        parts.append(f"<dt>params</dt><dd><code>{rendered}</code></dd>")
    if finding.dataset_commit:
        parts.append(
            f'<dt>pin</dt><dd><code>{_escape(finding.dataset_commit[:12])}</code></dd>'
        )
    if finding.promoted:
        parts.append(
            f'<dt>decision</dt><dd>promoted to <code>{_escape(finding.script or "")}</code></dd>'
        )
    else:
        parts.append(
            f'<dt>decision</dt><dd>not promoted — {_escape(finding.retire_reason or "")}</dd>'
        )
    parts.append("</dl>")
    return parts


def _finding_html(
    number: int, finding: Any, lines: Sequence[int], commentary: str, render_block: Any
) -> str:
    """One row of the table: a line you can scan, that opens into the prose."""
    value = _short_value(_format_value(finding.value))
    # Params are how it was measured, not what it means: they belong in the
    # detail. On the card they were the longest, least readable thing on screen.
    meta = [f'<span class="f-n">n={finding.n:,}</span>', f'<span class="f-session">s{finding.session}</span>']
    # Three lines, not six columns: the number and the instrument, the value at
    # a size you can scan, and the provenance as small print. Six columns is
    # fine on a desktop and unreadable at 360px, and this is read on a phone.
    summary = [
        '<span class="f-top">',
        f'<span class="f-no">{number}</span>',
        f'<span class="f-tool"><code>{_escape(finding.tool)}</code></span>',
        '<span class="f-caret" aria-hidden="true"></span>',
        "</span>",
        '<span class="f-figure">',
        f'<span class="f-value{" f-long" if len(value) > 20 else ""}">{_escape(value)}</span>',
        f'<span class="f-unit">{_escape(finding.unit)}</span>',
        "</span>",
        f'<span class="f-meta">{"".join(meta)}</span>',
    ]
    # The reading is the row's reason to exist: a number with no sentence about
    # it is a puzzle, and a report full of puzzles is the wall of text again.
    reading, rest = _split_reading(commentary)
    if reading:
        summary.append(f'<span class="f-reading">{_plain(render_block(reading))}</span>')
    body = _detail_rows(finding, lines)
    if rest:
        body.append(f'<div class="finding-note">{render_block(rest)}</div>')
    elif not reading:
        body.append('<p class="finding-none">No commentary recorded for this line.</p>')
    return (
        f'<details class="finding" id="f{number}">'
        f'<summary>{"".join(summary)}</summary>'
        f'<div class="f-body">{"".join(body)}</div>'
        "</details>"
    )


def _distinct_rows(findings: Sequence[Any]) -> list[tuple[Any, list[int]]]:
    """One row per finding, not per line.

    The ledger is append-only and a sweep can be re-run, so the same
    measurement legitimately appears on several lines — this session has 46
    lines and 21 findings. A reader wants the finding; the lines it came from
    belong in its detail. Row order is first-recorded order, which is also the
    order the session measured things in.
    """
    rows: list[tuple[Any, list[int]]] = []
    seen: dict[str, int] = {}
    for line_number, finding in enumerate(findings, start=1):
        key = normalized_line(finding.to_json())
        if key in seen:
            rows[seen[key]][1].append(line_number)
            continue
        seen[key] = len(rows)
        rows.append((finding, [line_number]))
    return rows


def _findings_html(
    findings: Sequence[Any], commentary: dict[int, str], render_block: Any
) -> tuple[str, set[int]]:
    """The table itself, with a one-line count of what the ledger holds.

    Returns the HTML and the marker numbers it consumed, so the caller can tell
    a stale marker (a line that no longer exists) from a used one.
    """
    rows = _distinct_rows(findings)
    row_of_line = {
        line: row_number
        for row_number, (_finding, lines) in enumerate(rows, start=1)
        for line in lines
    }
    used: set[int] = set()
    blocks: dict[int, str] = {}
    for number, text in commentary.items():
        row_number = row_of_line.get(number, number if 1 <= number <= len(rows) else None)
        if row_number is None:
            continue
        used.add(number)
        blocks[row_number] = f"{blocks[row_number]}\n\n{text}" if row_number in blocks else text

    promoted = sum(1 for finding, _lines in rows if finding.promoted)
    tools = sorted({finding.tool for finding, _lines in rows})
    body = "".join(
        _finding_html(row_number, finding, lines, blocks.get(row_number, ""), render_block)
        for row_number, (finding, lines) in enumerate(rows, start=1)
    )
    lines_total = sum(len(lines) for _finding, lines in rows)
    reappends = (
        f" · {lines_total} ledger lines" if lines_total > len(rows) else ""
    )
    summary = (
        f'<p class="findings-count">{len(rows)} finding'
        f'{"" if len(rows) == 1 else "s"} · {promoted} promoted · '
        f"{len(rows) - promoted} retired{reappends} · {_escape(', '.join(tools))}</p>"
    )
    return (
        '<section class="findings" aria-label="Findings">'
        f"{summary}{body}</section>",
        used,
    )


def _render_parser() -> Any:
    from markdown_it import MarkdownIt

    parser = MarkdownIt("commonmark", {"html": False, "linkify": False})
    parser.enable("table")
    parser.enable("strikethrough")
    return parser


def _after_lede(body: str, insertion: str) -> str:
    """Where the table goes: after the title and its opening paragraph.

    Not immediately after ``</h1>`` — an intro sentence belongs above the
    ledger, or the table reads as the report and the prose as a footnote.
    """
    title_end = body.find("</h1>")
    if title_end == -1:
        return f"{insertion}\n{body}"
    end = title_end + len("</h1>")
    paragraph = body.find("</p>", end)
    if paragraph != -1 and "<h2" not in body[end:paragraph]:
        end = paragraph + len("</p>")
    return body[:end] + "\n" + insertion + body[end:]


def render_markdown(
    markdown: str, *, title: str, meta: str = "", findings: Sequence[Any] = ()
) -> str:
    """``report-*.md`` -> standalone HTML. No browser engine, no PDF.

    The report is anchored on the ledger: ``findings`` — the lines as recorded —
    become a table of rows you can scan, and each row opens into the commentary
    the markdown attaches to it with ``<!-- finding: N -->``. Prose that is not
    attached to a line is still rendered, under a heading that says so, because
    a report must not be able to lose text by being restructured.
    """
    lede, commentary, notes = split_commentary(markdown)

    parser = _render_parser()
    ids, sections = _structure(parser.parse(lede))

    def heading_open(
        _renderer: Any, tokens: Sequence[Any], index: int, *_rest: Any
    ) -> str:
        token = tokens[index]
        return f'<{token.tag} id="{_escape(ids.get(index, "section"))}">'

    parser.add_render_rule("heading_open", heading_open)
    body = parser.render(lede)

    note_parser = _render_parser()

    def render_block(text: str) -> str:
        return note_parser.render(text)

    extras: list[str] = []
    if commentary and not findings:
        # No ledger to anchor to: keep every byte of the commentary, in order.
        extras.extend(
            f'<h3 id="finding-{number}">Finding {number}</h3>{render_block(text)}'
            for number, text in sorted(commentary.items())
        )
    elif findings:
        table, used = _findings_html(findings, commentary, render_block)
        extras.append(table)
        orphans = {number: text for number, text in commentary.items() if number not in used}
        if orphans:
            extras.append('<h2 id="unattached">Commentary with no ledger line</h2>')
            extras.append(
                '<p class="muted">These blocks name a finding number the ledger '
                "does not have. Nothing is dropped; the number is wrong.</p>"
            )
            extras.extend(
                f'<h3 id="unattached-{number}">Finding {number}</h3>{render_block(text)}'
                for number, text in sorted(orphans.items())
            )
    if notes:
        extras.append(f'<section class="report-notes" id="notes">{render_block(notes)}</section>')

    toc = _contents(sections)
    if toc:
        extras.insert(0, toc)
    if extras:
        body = _after_lede(body, "".join(extras))

    return _HTML_TEMPLATE.format(
        title=_escape(title),
        css=stylesheet(),
        meta=_escape(meta),
        nav=NAV_HTML,
        body=body,
    )


_BODY_OPEN = re.compile(r"<body[^>]*>", re.IGNORECASE)


def with_chrome(document: str, *, meta: str = "") -> str:
    """Splice the reading chrome into a whole HTML document somebody else wrote.

    ``nbconvert`` hands back a complete document — doctype, ``<head>``, its own
    stylesheets, ``<body class="jp-Notebook">`` — so wrapping it in
    :data:`_HTML_TEMPLATE` nests one document inside another: two doctypes, and
    the nav bar stranded above the inner ``<html>``. This edits the document in
    place instead: the nav bar and meta line go first in ``<body>``, and our
    sheet goes **last** in ``<head>`` so it wins over nbconvert's while leaving
    nbconvert's rules there for the things this sheet does not style (syntax
    colours, ``--jp-*`` variables, output rendering).

    Idempotent, and deliberately idle on a fragment: a document already carrying
    the nav is returned untouched, and so is a file with no ``<head>`` and no
    ``<body>`` — that is not a document, and guessing at it would corrupt it.
    Used for the notebook rendition and, at serve time, for a report rendered
    before the chrome existed (``__main__.artifact``).
    """
    opened = _BODY_OPEN.search(document)
    if "</head>" not in document and opened is None:
        return document

    # The sheet is replaced, not merely inserted. An artifact carries a copy of
    # the stylesheet it was rendered with, so inserting ours a second time would
    # leave the old rules in front of the new ones — and a frozen report would
    # never pick up a restyle. Marked, therefore, and swapped.
    sheet = f"<style {STYLE_ATTR}>\n{stylesheet()}\n</style>"
    if _STYLE_BLOCK.search(document):
        document = _STYLE_BLOCK.sub(lambda _match: sheet, document, count=1)
    elif "</head>" in document:
        document = document.replace("</head>", f"{sheet}\n</head>", 1)
    elif opened is not None:
        document = document[: opened.start()] + sheet + "\n" + document[opened.start() :]
    else:
        document = sheet + document

    if NAV_MARKER in document:
        return document

    opened = _BODY_OPEN.search(document)
    chrome = NAV_HTML
    if meta:
        chrome += f'\n<div class="data-scout-meta">{_escape(meta)}</div>'
    if opened is None:
        return chrome + document
    return document[: opened.end()] + "\n" + chrome + document[opened.end() :]


def render_notebook(notebook: Any, *, title: str, meta: str = "") -> str:
    """An executed ``.ipynb`` -> standalone HTML, code blocks scrolling not clipping."""
    from nbconvert import HTMLExporter

    exporter = HTMLExporter(template_name="lab")
    body, _resources = exporter.from_notebook_node(notebook, resources={"metadata": {"name": title}})
    # The exporter's own document, dressed — never wrapped. One stylesheet for
    # both renditions: ours lands last, so the report and the appendix read like
    # one document family without losing nbconvert's output rendering.
    return with_chrome(body, meta=meta)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def pick_notebook(dataset_id: str, session: int | None) -> tuple[Path, int]:
    directory = workspace.notebook_dir(dataset_id)
    if session is not None:
        path = directory / f"session-{session}.ipynb"
        if not path.is_file():
            raise SystemExit(f"report-render: no notebook at {path}")
        return path, session
    candidates: list[tuple[int, Path]] = []
    for path in sorted(directory.glob("session-*.ipynb")):
        stem = path.stem.removeprefix("session-")
        if stem.isdigit():
            candidates.append((int(stem), path))
    if not candidates:
        raise SystemExit(f"report-render: no session-*.ipynb under {directory}")
    number, path = max(candidates)
    return path, number


def pick_report(dataset_id: str, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"report-render: no report at {path}")
        return path
    candidates = sorted(workspace.workspace_dir(dataset_id).glob("report-*.md"))
    return candidates[-1] if candidates else None


def _meta_line(dataset_id: str, session: int, source: str) -> str:
    from datetime import UTC, datetime

    stamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return f"data-scout · dataset {dataset_id} · session {session} · {source} · rendered {stamp}"


def write_report(dataset_id: str, session: int, report_path: Path, ledger: Ledger) -> Path:
    """Render ``report-*.md`` against the ledger as it stands.

    A report is the deliverable and lives at the dataset root, even if
    ``--report`` named it from somewhere else on disk. The ledger is its spine:
    every row is a line as recorded, and the markdown hangs commentary off it.
    """
    report_html = workspace.artifact_path(dataset_id, report_path.with_suffix(".html").name)
    report_html.write_text(
        render_markdown(
            report_path.read_text(encoding="utf-8"),
            title=report_path.stem,
            meta=_meta_line(dataset_id, session, report_path.name),
            findings=ledger.findings(),
        ),
        encoding="utf-8",
    )
    return report_html


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="report-render",
        description=(
            "Execute a session notebook, verify that a re-execution reproduces "
            "its findings.jsonl lines, then render report and notebook to HTML."
        ),
    )
    parser.add_argument("dataset_id", help="the dataset id, as it appears under the datasets root")
    parser.add_argument("--session", type=int, default=None, help="session number (default: the highest present)")
    parser.add_argument("--report", default=None, help="path to report-*.md (default: the latest in the workspace)")
    parser.add_argument("--workspace", default=None, help="override DATA_SCOUT_WORKSPACE")
    parser.add_argument("--timeout", type=int, default=1800, help="per-cell execution timeout, seconds")
    parser.add_argument("--verify-only", action="store_true", help="execute and verify, but render nothing")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="re-render report-*.md against the ledger as it stands, executing nothing",
    )
    parser.add_argument("--quiet", action="store_true", help="do not print per-cell output")
    args = parser.parse_args(argv)

    dataset_id = workspace.validate_dataset_id(args.dataset_id)
    if args.workspace:
        os.environ[workspace.WORKSPACE_ENV] = args.workspace
    root = workspace.workspace_root()
    dataset_ws = workspace.workspace_dir(dataset_id)
    if not dataset_ws.is_dir():
        print(f"report-render: no workspace at {dataset_ws}", file=sys.stderr)
        return 2
    workspace.ensure_workspace(dataset_id)

    notebook_path, session = pick_notebook(dataset_id, args.session)
    ledger = Ledger(workspace.ledger_path(dataset_id))
    print(f"report-render: dataset {dataset_id}, session {session}")
    print(f"report-render: notebook {notebook_path}")
    print(f"report-render: ledger {ledger.path} ({len(ledger.lines())} lines)")

    if args.report_only:
        # Prose is not a measurement: re-rendering a report against the ledger
        # as it already stands must not execute anything, or every wording fix
        # would append the session's lines a second time and the gate would
        # have to be rolled back by hand.
        report_path = pick_report(dataset_id, args.report)
        if report_path is None:
            print("report-render: no report-*.md in the workspace; nothing to render", file=sys.stderr)
            return 2
        print(f"report-render: wrote {write_report(dataset_id, session, report_path, ledger)}")
        return 0

    # --- 3a. snapshot the live workspace BEFORE anything executes -----------
    scratch_parent = Path(tempfile.mkdtemp(prefix="data-scout-verify-"))
    scratch_root = scratch_parent
    try:
        scratch_workspace(dataset_ws, destination=scratch_root / dataset_id)

        # Sampled BEFORE the artifact execution: the lines this render appends
        # are the difference between here and the staged ledger. Sampling it
        # after execution made ``added`` arithmetically always 0.
        before = ledger.lines()

        # The artifact execution appends to a *staged* copy of the ledger rather
        # than the live file. Cell 4 of a notebook appends unconditionally, so
        # without this a run that later fails a gate has already written a line
        # — twice now, and both times it had to be rolled back from a snapshot.
        # Seeding the stage with the live lines means a notebook that *reads*
        # the ledger sees exactly the history it would see live.
        staged_path = scratch_parent / "staged" / workspace.LEDGER_FILENAME
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path.write_text(
            "".join(line + "\n" for line in before), encoding="utf-8"
        )
        staged = Ledger(staged_path)

        # --- 1+2. execute for the artifact ---------------------------------
        print("report-render: executing for the artifact")
        with scoped_env({workspace.LEDGER_OVERRIDE_ENV: str(staged_path)}):
            executed = execute_notebook(
                notebook_path,
                workspace_root=root,
                dataset_id=dataset_id,
                session=session,
                timeout=args.timeout,
            )
        if not args.quiet:
            print_cell_outputs(executed)
        errors = cell_errors(executed)
        if errors:
            print(f"report-render: {len(errors)} cell error(s); not rendering", file=sys.stderr)
            for index, detail in errors:
                print(f"  cell {index}: {detail.splitlines()[0]}", file=sys.stderr)
            return 1

        from nbformat import write as write_nbformat

        write_nbformat(executed, str(notebook_path))

        # --- 3b. re-execute into the scratch copy and diff ------------------
        print("report-render: re-executing into a scratch workspace for the diff")
        verification = execute_notebook(
            notebook_path,
            workspace_root=scratch_root,
            dataset_id=dataset_id,
            session=session,
            timeout=args.timeout,
        )
        verification_errors = cell_errors(verification)
        staged_after = staged.lines()
        scratch_ledger = Ledger(scratch_root / dataset_id / workspace.LEDGER_FILENAME)

        problems = diff_lines(staged_after, scratch_ledger.lines())
        ledger_problems = ledger.verify()
        # A frame line is a ledger line, so the diff would otherwise read a v1
        # vs v2 difference as a byte delta and call it drift. It is not drift:
        # the two numbers were measured under different frame rules.
        frame_problems = frame_identity.comparability_problems(
            staged_after, scratch_ledger.lines()
        )

        # --- 4. the gate ----------------------------------------------------
        if verification_errors or problems or ledger_problems or frame_problems:
            print("report-render: VERIFICATION FAILED — refusing to render", file=sys.stderr)
            for index, detail in verification_errors:
                print(f"  verification cell {index}: {detail.splitlines()[0]}", file=sys.stderr)
            for problem in problems[:40]:
                print(f"  {problem}", file=sys.stderr)
            for problem in ledger_problems[:20]:
                print(f"  ledger {problem}", file=sys.stderr)
            for problem in frame_problems:
                print(f"  {problem}", file=sys.stderr)
            print(
                "  The rendered HTML outlives a correction, so a known-bad "
                "rendition is worse than no rendition. Nothing was rendered.",
                file=sys.stderr,
            )
            return 1

        # Both gates passed: only now does anything reach the live ledger. The
        # staged lines are appended byte for byte, so the record is exactly what
        # the notebook would have written had it executed against the live
        # workspace, and exactly once.
        appended = staged_after[len(before):]
        # Append-once for frame lines: an unchanged notebook re-rendered must
        # not grow a duplicate frame statement. The key includes the pin
        # (frame_dedup_key), so a re-pin still appends. Every other tool keeps
        # ordinary append-only semantics.
        to_append, duplicates = frame_identity.promote_once(appended, before)
        for line in to_append:
            ledger.append_line(line)
        added = len(to_append)
        if duplicates:
            print(
                f"report-render: skipped {len(duplicates)} frame line(s) already "
                "in the live ledger (append-once, frame only)"
            )
        print(
            f"report-render: verification passed "
            f"({len(before) + added} lines, {added} appended by this render)"
        )
        if args.verify_only:
            print("report-render: --verify-only, nothing rendered")
            return 0

        # --- 5. render ------------------------------------------------------
        rendered: list[Path] = []
        # One location rule, shared with the `/report/` reader. Both artifacts
        # live at the dataset root beside frame.md; writing this one beside the
        # notebook by `with_suffix` was the defect that hid every correct render
        # from the index.
        session_html = workspace.artifact_path(
            dataset_id, notebook_path.with_suffix(".html").name
        )
        session_html.write_text(
            render_notebook(
                executed,
                title=f"{dataset_id} — session {session}",
                meta=_meta_line(dataset_id, session, notebook_path.name),
            ),
            encoding="utf-8",
        )
        rendered.append(session_html)

        report_path = pick_report(dataset_id, args.report)
        if report_path is None:
            print("report-render: no report-*.md in the workspace; nothing to render for the deliverable")
        else:
            rendered.append(write_report(dataset_id, session, report_path, ledger))

        # --- 6 is above, deliberately: the agent reads cells, not a summary.
        for path in rendered:
            print(f"report-render: wrote {path}")
        return 0
    finally:
        shutil.rmtree(scratch_parent, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover - console script is the entry point
    raise SystemExit(main())
