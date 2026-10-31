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
    notebook_path = build_session(
        dataset_workspace,
        cells=[APPEND_CELL],
        report_text="# A test report\n\n| unit | value |\n|---|---|\n| file | 0.9 |\n",
    )

    code = render.main([DATASET_ID, "--quiet"])
    assert code == 0, capsys.readouterr().err

    assert len(ledger_lines()) == 1
    session_html = notebook_path.with_suffix(".html")
    report_html = dataset_workspace / "report-2026-10-26.html"
    assert session_html.is_file()
    assert report_html.is_file()
    assert "<table>" in report_html.read_text(encoding="utf-8")
    assert "prefers-color-scheme" in report_html.read_text(encoding="utf-8")


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
    notebook_path = build_session(dataset_workspace, cells=[NONDETERMINISTIC_CELL])

    code = render.main([DATASET_ID, "--quiet"])
    captured = capsys.readouterr()
    assert code == 1
    assert "VERIFICATION FAILED" in captured.err
    assert "No" in captured.err or "nothing was rendered" in captured.err.lower()
    # A known-bad rendition is worse than no rendition.
    assert not notebook_path.with_suffix(".html").exists()


def test_cell_output_is_printed_so_the_agent_can_iterate(dataset_workspace, capsys):
    build_session(dataset_workspace, cells=['print("hello from the kernel")', APPEND_CELL])
    assert render.main([DATASET_ID]) == 0
    captured = capsys.readouterr()
    assert "hello from the kernel" in captured.out
    assert "=== cell 2 ===" in captured.out


def test_a_failing_cell_is_reported_and_not_rendered(dataset_workspace, capsys):
    notebook_path = build_session(dataset_workspace, cells=["raise ValueError('cell seven')"])
    code = render.main([DATASET_ID, "--quiet"])
    captured = capsys.readouterr()
    assert code == 1
    assert "cell 2" in captured.err
    assert not notebook_path.with_suffix(".html").exists()


def test_verify_only_renders_nothing(dataset_workspace, capsys):
    notebook_path = build_session(dataset_workspace, cells=[APPEND_CELL])
    assert render.main([DATASET_ID, "--quiet", "--verify-only"]) == 0
    assert not notebook_path.with_suffix(".html").exists()


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
    notebook_path = build_session(dataset_workspace, cells=[V2_FRAME_CELL])

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
    assert notebook_path.with_suffix(".html").exists()


def test_the_gate_says_not_comparable_instead_of_drifting_across_frame_schemas(
    dataset_workspace, capsys
):
    """A v1 and a v2 frame line are not a byte delta; they are different rules."""
    notebook_path = build_session(dataset_workspace, cells=[FRAME_VERSION_BY_RUN])

    code = render.main([DATASET_ID, "--quiet"])
    captured = capsys.readouterr()
    assert code == 1
    assert "not comparable" in captured.err
    assert "different frame rules" in captured.err
    assert not notebook_path.with_suffix(".html").exists()
