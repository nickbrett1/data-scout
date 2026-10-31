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
