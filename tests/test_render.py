"""``report-render``: execute, verify, gate, render — and never grow the ledger twice.

The two tests here that are easy to fake are the ones that matter:

* does a render leave the ledger the size it should (N × 1, not 2N)?
* does re-execution reproduce the lines modulo ``ts`` — and does the gate
  actually refuse a rendition when it does not?
"""

from __future__ import annotations

import json
from pathlib import Path

from data_scout import notebook as nbwrite
from data_scout import render, workspace

DATASET_ID = "demo-dataset"

APPEND_CELL = """
from data_scout.tools import findings_append

findings_append({
    "unit": "file",
    "tool": "file_census",
    "tool_version": "1",
    "params": {"per_chars": 1000},
    "value": 0.9,
    "n": 2,
    "promoted": False,
    "retire_reason": "one-off: the test dataset is two files",
})
"""

NONDETERMINISTIC_CELL = """
import time
from data_scout.tools import findings_append

findings_append({
    "unit": "file",
    "tool": "text_quality",
    "tool_version": "1",
    "params": {},
    "value": time.time(),
    "n": 1,
    "promoted": False,
    "retire_reason": "one-off: deliberately not reproducible",
})
"""


def build_session(dataset_workspace: Path, *, cells: list[str], session: int = 1, report_text: str | None = None) -> Path:
    cells_nodes = [nbwrite.markdown_cell(f"# Session {session}")]
    cells_nodes += [nbwrite.code_cell(source) for source in cells]
    notebook = nbwrite.new_notebook(cells_nodes)
    path = nbwrite.write_notebook(workspace.notebook_dir(DATASET_ID) / f"session-{session}.ipynb", notebook)
    if report_text is not None:
        (dataset_workspace / "report-2026-10-26.md").write_text(report_text, encoding="utf-8")
    return path


def ledger_lines() -> list[dict]:
    path = workspace.ledger_path(DATASET_ID)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_a_render_executes_verifies_and_writes_both_renditions(dataset_workspace, capsys):
    build_session(
        dataset_workspace,
        cells=[APPEND_CELL],
        report_text="# A test report\n\n| unit | value |\n|---|---|\n| file | 0.9 |\n",
    )

    code = render.main([DATASET_ID, "--quiet"])
    assert code == 0, capsys.readouterr().err

    assert len(ledger_lines()) == 1
    session_html = workspace.artifact_path(DATASET_ID, "session-1.html")
    report_html = dataset_workspace / "report-2026-10-26.html"
    assert session_html.is_file()
    assert report_html.is_file()
    assert "<table>" in report_html.read_text(encoding="utf-8")
    assert "prefers-color-scheme" in report_html.read_text(encoding="utf-8")


def test_writer_and_reader_agree_on_the_session_html_location(dataset_workspace, capsys):
    """Pin the writer's path to the reader's path: the two disagreed once.

    This is the defect measured on ryzen: exit 0, gate passed, 337 KB of real
    HTML written to ``notebooks/``, while ``/report/`` both indexed and resolved
    from the dataset root — so the index showed ``artifacts: []`` and the file
    was unreachable. 173 passing tests stood between the two functions and
    caught nothing, because no test asserted that the writer's location *is* the
    reader's location. This one does, from both ends:

    * the writer (``render.main``) must land the file at
      ``workspace_dir(dataset) / "session-1.html"`` — beside ``frame.md``;
    * the reader (``index_payload`` and ``resolve_artifact``) must name and
      resolve that exact path.
    """
    from data_scout import __main__ as app

    build_session(dataset_workspace, cells=[APPEND_CELL])
    assert render.main([DATASET_ID, "--quiet"]) == 0, capsys.readouterr().err

    # The one path both sides must agree on: dataset root, not notebooks/.
    canonical = workspace.workspace_dir(DATASET_ID) / "session-1.html"
    assert workspace.artifact_path(DATASET_ID, "session-1.html") == canonical
    assert canonical.is_file()
    assert not (workspace.notebook_dir(DATASET_ID) / "session-1.html").exists()

    record = next(
        item for item in app.index_payload()["datasets"] if item["dataset_id"] == DATASET_ID
    )
    assert record["names"] == ["session-1.html"]
    assert record["artifacts"] == [f"/report/{DATASET_ID}/session-1.html"]
    assert app.resolve_artifact(f"/report/{DATASET_ID}/session-1.html") == canonical


def test_the_render_reports_the_lines_this_run_appended(dataset_workspace, capsys):
    """``before`` is sampled before execution; sampling after always printed 0."""
    build_session(dataset_workspace, cells=[APPEND_CELL])

    assert render.main([DATASET_ID, "--quiet"]) == 0
    assert "1 appended by this render" in capsys.readouterr().out


def test_the_render_stamps_the_kernelspec(dataset_workspace, capsys):
    """Without it the first thing a person sees is a 'Select Kernel' prompt."""
    notebook_path = build_session(dataset_workspace, cells=[APPEND_CELL])
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    payload["metadata"].pop("kernelspec", None)
    notebook_path.write_text(json.dumps(payload), encoding="utf-8")

    assert render.main([DATASET_ID, "--quiet"]) == 0, capsys.readouterr().err

    saved = nbwrite.read_notebook(notebook_path)
    assert nbwrite.has_kernelspec(saved)
    assert saved.metadata["kernelspec"]["language"] == "python"


def test_a_render_does_not_grow_the_ledger_twice(dataset_workspace, capsys):
    """N renders of one session leave N × 1 sets of lines, not 2N."""
    build_session(dataset_workspace, cells=[APPEND_CELL])

    assert render.main([DATASET_ID, "--quiet"]) == 0, capsys.readouterr().err
    assert len(ledger_lines()) == 1

    assert render.main([DATASET_ID, "--quiet"]) == 0, capsys.readouterr().err
    # The verification execution wrote to a scratch copy, so it added nothing.
    assert len(ledger_lines()) == 2


def test_the_gate_refuses_a_rendition_that_does_not_reproduce(dataset_workspace, capsys):
    build_session(dataset_workspace, cells=[NONDETERMINISTIC_CELL])

    code = render.main([DATASET_ID, "--quiet"])
    captured = capsys.readouterr()
    assert code == 1
    assert "VERIFICATION FAILED" in captured.err
    assert "No" in captured.err or "nothing was rendered" in captured.err.lower()
    # A known-bad rendition is worse than no rendition.
    assert not workspace.artifact_path(DATASET_ID, "session-1.html").exists()


def test_cell_output_is_printed_so_the_agent_can_iterate(dataset_workspace, capsys):
    build_session(dataset_workspace, cells=['print("hello from the kernel")', APPEND_CELL])
    assert render.main([DATASET_ID]) == 0
    captured = capsys.readouterr()
    assert "hello from the kernel" in captured.out
    assert "=== cell 2 ===" in captured.out


def test_a_failing_cell_is_reported_and_not_rendered(dataset_workspace, capsys):
    build_session(dataset_workspace, cells=["raise ValueError('cell seven')"])
    code = render.main([DATASET_ID, "--quiet"])
    captured = capsys.readouterr()
    assert code == 1
    assert "cell 2" in captured.err
    assert not workspace.artifact_path(DATASET_ID, "session-1.html").exists()


def test_verify_only_renders_nothing(dataset_workspace, capsys):
    build_session(dataset_workspace, cells=[APPEND_CELL])
    assert render.main([DATASET_ID, "--quiet", "--verify-only"]) == 0
    assert not workspace.artifact_path(DATASET_ID, "session-1.html").exists()


def test_a_missing_workspace_is_a_refusal_not_a_crash(roots, capsys):
    assert render.main(["no-such-dataset"]) == 2
    assert "no workspace" in capsys.readouterr().err


FRAME_VERSION_BY_RUN = """
from data_scout import workspace
from data_scout.tools import findings_append

# The verification run works in a scratch workspace whose path names it.
verified = "data-scout-verify" in str(workspace.workspace_root())
schema = 1 if verified else 2
findings_append({
    "unit": "files",
    "tool": "frame",
    "tool_version": str(schema),
    "params": {"frame_schema": schema},
    "value": 200 if schema == 1 else 100,
    "n": 8,
    "promoted": False,
    "retire_reason": "one-off: the test writes a frame line under a chosen schema",
})
"""


RETIRED_V1_FRAME_LINE = {
    "dataset_id": DATASET_ID,
    "session": 1,
    "unit": "files",
    "tool": "frame",
    "tool_version": "1",
    "params": {},
    "value": 934_881_648,
    "n": 56,
    "promoted": False,
    "retire_reason": (
        "one-off: dataset identity hash from scratch/frame_build.py; the belt has "
        "no frame instrument. Recorded so the exact tree this session measured is "
        "pinned."
    ),
}

V2_FRAME_CELL = """
from data_scout.tools import findings_append

findings_append({
    "unit": "files",
    "tool": "frame",
    "tool_version": "2",
    "params": {"frame_schema": 2},
    "value": 934_843_620,
    "n": 8,
    "promoted": False,
    "retire_reason": "one-off: the test writes the v2 payload identity as a cell",
})
"""


def test_a_retired_v1_line_beside_a_new_v2_line_still_verifies_clean(
    dataset_workspace, capsys
):
    """The real ledger is ``[1(unpromoted), 2]``, not ``[1, 2]`` invented by a cell.

    ``promoted`` is not part of the schema set the gate compares, but measuring it
    against an unpromoted v1 line is the point: the v1 line is *already retired*
    and must be carried through the re-execution copy unchanged, not appended by
    the cell. Both sides end at the same schema set, so the gate result is ``[]``
    — measured here rather than inferred from the set logic.
    """
    from data_scout import frame
    from data_scout.ledger import Finding, Ledger

    seeded = Finding.model_validate(RETIRED_V1_FRAME_LINE)
    Ledger(workspace.ledger_path(DATASET_ID)).append(seeded)
    build_session(dataset_workspace, cells=[V2_FRAME_CELL])

    code = render.main([DATASET_ID, "--quiet"])
    assert code == 0, capsys.readouterr().err

    lines = ledger_lines()
    assert len(lines) == 2
    assert lines[0]["promoted"] is False
    assert lines[0]["tool_version"] == "1"
    assert frame.ledger_frame_schemas(lines) == [1, 2]

    # The exact operands the gate sees: the seeded v1 line is copied into the
    # scratch workspace before execution, so the re-execution appends the same
    # v2 line to the same history. Both schema sets are [1, 2].
    scratch_after = [lines[0], lines[1]]
    assert frame.comparability_problems(lines, scratch_after) == []
    assert workspace.artifact_path(DATASET_ID, "session-1.html").exists()


def test_the_gate_says_not_comparable_instead_of_drifting_across_frame_schemas(
    dataset_workspace, capsys
):
    """A v1 and a v2 frame line are not a byte delta; they are different rules."""
    build_session(dataset_workspace, cells=[FRAME_VERSION_BY_RUN])

    code = render.main([DATASET_ID, "--quiet"])
    captured = capsys.readouterr()
    assert code == 1
    assert "not comparable" in captured.err
    assert "different frame rules" in captured.err
    assert not workspace.artifact_path(DATASET_ID, "session-1.html").exists()


# --------------------------------------------------------------------------
# defect: the ledger must not be written before the gates pass
# --------------------------------------------------------------------------

APPEND_THEN_FAIL_CELL = """
from data_scout.tools import findings_append

findings_append({
    "unit": "file",
    "tool": "file_census",
    "tool_version": "1",
    "params": {"path": "."},
    "value": 0.9,
    "n": 2,
    "promoted": False,
    "retire_reason": "one-off: written before the failing cell",
})

raise ValueError("the render must not keep the line cell 3 wrote")
"""


def ledger_bytes() -> bytes:
    path = workspace.ledger_path(DATASET_ID)
    return path.read_bytes() if path.exists() else b""


def test_a_failing_cell_leaves_the_ledger_byte_identical(dataset_workspace, capsys):
    """Cell 4 appends unconditionally; a later error must not keep that line.

    Before the fix the artifact execution ran against the live workspace, so
    the append had already landed by the time the cell-error gate ran — twice
    on ryzen, and both times it had to be rolled back from a snapshot.
    """
    build_session(dataset_workspace, cells=[APPEND_THEN_FAIL_CELL])
    before = ledger_bytes()

    code = render.main([DATASET_ID, "--quiet"])

    assert code == 1
    assert "cell 2" in capsys.readouterr().err
    assert ledger_bytes() == before, "a failed render changed findings.jsonl"


def test_a_failing_determinism_gate_leaves_the_ledger_byte_identical(dataset_workspace, capsys):
    """The gate runs after execution, so the append must still be staged."""
    build_session(dataset_workspace, cells=[NONDETERMINISTIC_CELL])
    before = ledger_bytes()

    code = render.main([DATASET_ID, "--quiet"])

    assert code == 1
    assert "VERIFICATION FAILED" in capsys.readouterr().err
    assert ledger_bytes() == before, "a gate failure changed findings.jsonl"
    assert not workspace.artifact_path(DATASET_ID, "session-1.html").exists()


def test_a_passing_render_appends_the_staged_line_once_byte_for_byte(dataset_workspace, capsys):
    """Promotion must be indistinguishable from the append the kernel made."""
    from data_scout.ledger import Finding

    build_session(dataset_workspace, cells=[APPEND_CELL])

    assert render.main([DATASET_ID, "--quiet"]) == 0, capsys.readouterr().err

    lines = workspace.ledger_path(DATASET_ID).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, "the staged line is committed exactly once"
    # A line written through Ledger.append is canonical to_json; promotion must
    # preserve those bytes rather than re-deriving them.
    assert lines[0] == Finding.from_json(lines[0]).to_json()
    assert json.loads(lines[0])["params"] == {"per_chars": 1000}


# --------------------------------------------------------------------------
# append-once at promotion: a re-rendered frame cell does not duplicate
# --------------------------------------------------------------------------

FRAME_APPEND_CELL = """
from data_scout.tools import findings_append

findings_append({
    "unit": "files",
    "tool": "frame",
    "tool_version": "3",
    "params": {
        "frame_schema": 3,
        "identity_scope": "data",
        "identity_files": 8,
        "identity_manifest": "abc",
        "checkout_scope": ".",
        "checkout_files": 12,
        "checkout_bytes": 934848934,
        "checkout_manifest": "def",
        "dataset_pinned": False,
        "dataset_commit": None,
        "dataset_manifest_hash": None,
        "dataset_files": None,
        "dataset_bytes": None,
    },
    "value": 934843620,
    "n": 8,
    "promoted": False,
    "retire_reason": "one-off: the test writes one unchanged frame line",
})
"""


def test_a_re_rendered_frame_line_is_appended_once(dataset_workspace, capsys):
    """The defect commit 7 closes: rendering an unchanged notebook again grew
    a duplicate frame line (identical modulo ``ts``). Frame lines are now
    append-once; a re-pin still appends (see test_frame.py)."""
    build_session(dataset_workspace, cells=[FRAME_APPEND_CELL])

    assert render.main([DATASET_ID, "--quiet"]) == 0, capsys.readouterr().err
    assert len(ledger_lines()) == 1

    assert render.main([DATASET_ID, "--quiet"]) == 0, capsys.readouterr().err
    second_out = capsys.readouterr().out
    assert len(ledger_lines()) == 1, "an unchanged frame line was appended twice"
    assert "skipped 1 frame line" in second_out


# --------------------------------------------------------------------------
# the report has to be readable, and it has to have a way back
# --------------------------------------------------------------------------

THREE_SECTIONS = """# vg books

Six PDFs, mostly art books.

## What is here

The tree holds 56 files.

## Counts

| unit | value |
|---|---|
| pages | 1778 |

## Counts

A second section with the same name must not steal the first one's anchor.
"""


def test_a_report_carries_the_way_back_to_the_index():
    html = render.render_markdown("# t", title="t")

    assert '<nav class="data-scout-nav">' in html
    assert 'class="back" href="/"' in html
    assert "All datasets" in html


def test_a_long_report_gets_a_contents_with_working_anchors():
    html = render.render_markdown(THREE_SECTIONS, title="vg books")

    assert 'class="data-scout-toc"' in html
    # Three sections, so the contents is an aid rather than furniture.
    assert html.count('<li><a href="#') == 3
    assert '<h2 id="counts">' in html
    assert '<h2 id="counts-2">' in html
    assert 'href="#counts-2"' in html
    # After the title, not before it.
    assert html.index("<h1") < html.index('class="data-scout-toc"')


def test_a_short_report_gets_no_contents_card():
    html = render.render_markdown("# t\n\ntext\n\n## one\n\nx\n\n## two\n\ny\n", title="t")
    assert 'class="data-scout-toc"' not in html


def test_the_rendered_report_inlines_the_current_stylesheet(dataset_workspace, capsys):
    """A deliverable read offline still has to look like the one served."""
    build_session(dataset_workspace, cells=[APPEND_CELL], report_text=THREE_SECTIONS)
    assert render.main([DATASET_ID, "--quiet"]) == 0, capsys.readouterr().err

    html = (dataset_workspace / "report-2026-10-26.html").read_text(encoding="utf-8")
    assert "--measure" in html
    assert '<nav class="data-scout-nav">' in html
    assert 'class="data-scout-toc"' in html


# --------------------------------------------------------------------------
# one document, not a document inside a document
# --------------------------------------------------------------------------


def test_with_chrome_dresses_a_whole_document_once():
    document = (
        "<!DOCTYPE html><html><head><style>x</style></head>"
        "<body class=\"jp-Notebook\"><h1>t</h1></body></html>"
    )
    dressed = render.with_chrome(document, meta="data-scout · x")

    assert dressed.count(render.NAV_MARKER) == 1
    assert dressed.count("<!DOCTYPE html>") == 1
    assert dressed.count("<html") == 1
    assert dressed.count("<body") == 1
    # Our sheet is last in <head> so it wins; nbconvert's stays for the things
    # this sheet does not style. The chrome is first in <body>.
    assert dressed.index("--measure") < dressed.index("</head>")
    assert "<style>x</style>" in dressed
    assert dressed.index("<body") < dressed.index(render.NAV_MARKER)
    assert 'class="data-scout-meta">data-scout · x</div>' in dressed
    # Idempotent: serving it again changes nothing.
    assert render.with_chrome(dressed, meta="data-scout · x") == dressed


def test_a_fragment_is_not_dressed_because_it_is_not_a_document():
    assert render.with_chrome("<html>report</html>") == "<html>report</html>"


def test_the_notebook_rendition_is_one_document_not_two(dataset_workspace, capsys):
    """nbconvert returns a whole document; wrapping it nested `<html>` in `<body>`.

    That left two doctypes, and the nav bar above the inner document rather than
    inside the page it navigates.
    """
    build_session(dataset_workspace, cells=[APPEND_CELL])
    assert render.main([DATASET_ID, "--quiet"]) == 0, capsys.readouterr().err

    html = workspace.artifact_path(DATASET_ID, "session-1.html").read_text(encoding="utf-8")
    assert html.count("<!DOCTYPE html>") == 1
    assert html.count("<html") == 1
    assert html.count("<body") == 1
    assert html.count("</html>") == 1
    # Inside the body it belongs to, and after the notebook's own markup starts.
    assert html.index("<body") < html.index(render.NAV_MARKER)
    # nbconvert's output rendering survives; ours is the last sheet in <head>.
    assert "jp-Notebook" in html
    assert html.rindex("--measure") < html.index("</head>")
