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
2. **Execute the notebook** for the artifact, saving outputs.
3. **Re-execute into a scratch copy** of the corpus workspace and diff
   ``findings.jsonl`` modulo ``ts``. The verification execution must never
   target the live workspace, or every render appends its own ledger a second
   time, silently, and the append-only ledger stops being a record of what the
   session measured.
4. **Gate the render on that diff.** The HTML outlives a correction, so a
   known-bad rendition is worse than no rendition.
5. **Render** ``report-*.md`` and ``session-N.ipynb`` to HTML: one stylesheet,
   no browser engine, no PDF (a stated non-goal).
6. **Print each cell's output and any error**, so the agent can iterate.

Rendered artifacts are **not** byte-reproducible and must not be tested as if
they were: execution counts, output ids and timestamps vary run to run. The
reproducibility claim is scoped to ``findings.jsonl`` lines, modulo ``ts``.
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any

from data_scout import notebook as nbwrite
from data_scout import workspace
from data_scout.ledger import Ledger, diff_lines

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
    """Copy a corpus workspace for the verification run, without its cache.

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
    corpus_id: str,
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
        workspace.CORPUS_ID_ENV: corpus_id,
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


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
<div class="data-scout-meta">{meta}</div>
{body}
</body>
</html>
"""


def render_markdown(markdown: str, *, title: str, meta: str = "") -> str:
    """``report-*.md`` -> standalone HTML. No browser engine, no PDF."""
    from markdown_it import MarkdownIt

    parser = MarkdownIt("commonmark", {"html": False, "linkify": False})
    parser.enable("table")
    parser.enable("strikethrough")
    body = parser.render(markdown)
    return _HTML_TEMPLATE.format(title=_escape(title), css=stylesheet(), meta=_escape(meta), body=body)


def render_notebook(notebook: Any, *, title: str, meta: str = "") -> str:
    """An executed ``.ipynb`` -> standalone HTML, code blocks scrolling not clipping."""
    from nbconvert import HTMLExporter

    exporter = HTMLExporter(template_name="lab")
    body, _resources = exporter.from_notebook_node(notebook, resources={"metadata": {"name": title}})
    # One stylesheet for both renditions: replace nbconvert's bundle with ours so
    # the report and the appendix look like one document family.
    start = body.find("<style")
    end = body.find("</style>", start)
    injected = f"<style>\n{stylesheet()}\n</style>"
    if start != -1 and end != -1:
        body = body[:start] + injected + body[end + len("</style>") :]
    else:
        body = injected + body
    return _HTML_TEMPLATE.format(title=_escape(title), css=stylesheet(), meta=_escape(meta), body=body)


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


def pick_notebook(corpus_id: str, session: int | None) -> tuple[Path, int]:
    directory = workspace.notebook_dir(corpus_id)
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


def pick_report(corpus_id: str, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"report-render: no report at {path}")
        return path
    candidates = sorted(workspace.workspace_dir(corpus_id).glob("report-*.md"))
    return candidates[-1] if candidates else None


def _meta_line(corpus_id: str, session: int, source: str) -> str:
    from datetime import UTC, datetime

    stamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return f"data-scout · corpus {corpus_id} · session {session} · {source} · rendered {stamp}"


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
    parser.add_argument("corpus_id", help="the corpus id, as under the corpora root")
    parser.add_argument("--session", type=int, default=None, help="session number (default: the highest present)")
    parser.add_argument("--report", default=None, help="path to report-*.md (default: the latest in the workspace)")
    parser.add_argument("--workspace", default=None, help="override DATA_SCOUT_WORKSPACE")
    parser.add_argument("--timeout", type=int, default=1800, help="per-cell execution timeout, seconds")
    parser.add_argument("--verify-only", action="store_true", help="execute and verify, but render nothing")
    parser.add_argument("--quiet", action="store_true", help="do not print per-cell output")
    args = parser.parse_args(argv)

    corpus_id = workspace.validate_corpus_id(args.corpus_id)
    if args.workspace:
        os.environ[workspace.WORKSPACE_ENV] = args.workspace
    root = workspace.workspace_root()
    corpus_ws = workspace.workspace_dir(corpus_id)
    if not corpus_ws.is_dir():
        print(f"report-render: no workspace at {corpus_ws}", file=sys.stderr)
        return 2
    workspace.ensure_workspace(corpus_id)

    notebook_path, session = pick_notebook(corpus_id, args.session)
    ledger = Ledger(workspace.ledger_path(corpus_id))
    print(f"report-render: corpus {corpus_id}, session {session}")
    print(f"report-render: notebook {notebook_path}")
    print(f"report-render: ledger {ledger.path} ({len(ledger.lines())} lines)")

    # --- 3a. snapshot the live workspace BEFORE anything executes -----------
    scratch_parent = Path(tempfile.mkdtemp(prefix="data-scout-verify-"))
    scratch_root = scratch_parent
    try:
        scratch_workspace(corpus_ws, destination=scratch_root / corpus_id)

        # --- 1+2. execute for the artifact ---------------------------------
        print("report-render: executing for the artifact")
        executed = execute_notebook(
            notebook_path,
            workspace_root=root,
            corpus_id=corpus_id,
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
        before = ledger.lines()
        verification = execute_notebook(
            notebook_path,
            workspace_root=scratch_root,
            corpus_id=corpus_id,
            session=session,
            timeout=args.timeout,
        )
        verification_errors = cell_errors(verification)
        live_after = ledger.lines()
        scratch_ledger = Ledger(scratch_root / corpus_id / workspace.LEDGER_FILENAME)

        problems = diff_lines(live_after, scratch_ledger.lines())
        ledger_problems = ledger.verify()

        # --- 4. the gate ----------------------------------------------------
        if verification_errors or problems or ledger_problems:
            print("report-render: VERIFICATION FAILED — refusing to render", file=sys.stderr)
            for index, detail in verification_errors:
                print(f"  verification cell {index}: {detail.splitlines()[0]}", file=sys.stderr)
            for problem in problems[:40]:
                print(f"  {problem}", file=sys.stderr)
            for problem in ledger_problems[:20]:
                print(f"  ledger {problem}", file=sys.stderr)
            print(
                "  The rendered HTML outlives a correction, so a known-bad "
                "rendition is worse than no rendition. Nothing was rendered.",
                file=sys.stderr,
            )
            return 1

        added = len(live_after) - len(before)
        print(f"report-render: verification passed ({len(live_after)} lines, {added} appended by this render)")
        if args.verify_only:
            print("report-render: --verify-only, nothing rendered")
            return 0

        # --- 5. render ------------------------------------------------------
        rendered: list[Path] = []
        session_html = notebook_path.with_suffix(".html")
        session_html.write_text(
            render_notebook(
                executed,
                title=f"{corpus_id} — session {session}",
                meta=_meta_line(corpus_id, session, notebook_path.name),
            ),
            encoding="utf-8",
        )
        rendered.append(session_html)

        report_path = pick_report(corpus_id, args.report)
        if report_path is None:
            print("report-render: no report-*.md in the workspace; nothing to render for the deliverable")
        else:
            report_html = report_path.with_suffix(".html")
            report_html.write_text(
                render_markdown(
                    report_path.read_text(encoding="utf-8"),
                    title=report_path.stem,
                    meta=_meta_line(corpus_id, session, report_path.name),
                ),
                encoding="utf-8",
            )
            rendered.append(report_html)

        # --- 6 is above, deliberately: the agent reads cells, not a summary.
        for path in rendered:
            print(f"report-render: wrote {path}")
        return 0
    finally:
        shutil.rmtree(scratch_parent, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover - console script is the entry point
    raise SystemExit(main())
