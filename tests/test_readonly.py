"""A data-scout run cannot modify the corpus. Asserted two ways.

The mount gives the guarantee in production; these tests state it as the
convention does — as the **absence** of a write handle, plus a live attempt that
must fail — so a future tool that forgets is caught before a session, not after.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from data_scout.tools import file_census, text_quality

TOOLS_DIR = Path(__file__).resolve().parents[1] / "src" / "data_scout" / "tools"

#: The only modules allowed to open anything for writing, and why:
#: ``findings_append`` appends to the workspace ledger, ``sample`` writes the
#: reproducibility anchor into ``samples/``. Both write inside the workspace
#: only; neither can reach the corpus.
_WRITE_ALLOWED = {"findings_append.py", "sample.py"}


def _write_modes(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
        if name not in {"open", "write_text", "write_bytes"}:
            continue
        for arg in [*node.args, *(keyword.value for keyword in node.keywords)]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if name == "open" and any(flag in arg.value for flag in ("w", "a", "x", "+")):
                    found.append(f"open({arg.value!r})")
                elif name in {"write_text", "write_bytes"}:
                    found.append(name)
    return found


def test_no_tool_opens_the_corpus_for_writing():
    offenders: dict[str, list[str]] = {}
    for path in sorted(TOOLS_DIR.glob("*.py")):
        if path.name in _WRITE_ALLOWED:
            continue
        modes = _write_modes(path)
        if modes:
            offenders[path.name] = modes
    assert offenders == {}, f"tools must not open anything for writing: {offenders}"


def test_the_two_writers_write_only_inside_the_workspace(corpus_workspace):
    """``findings_append`` and the sample anchor both resolve under the workspace."""
    from data_scout import workspace

    ledger = workspace.ledger_path("demo-corpus")
    samples = workspace.samples_dir("demo-corpus")
    root = workspace.workspace_root()
    assert ledger.is_relative_to(root)
    assert samples.is_relative_to(root)
    assert not str(ledger).startswith(str(workspace.corpora_root()))


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores mode bits; the mount is the real guarantee")
def test_a_write_to_a_read_only_corpus_fails_with_a_permission_error(corpus):
    corpus.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            (corpus / "should-not-exist.txt").write_text("nope", encoding="utf-8")
        # And the instruments still work against it.
        result = file_census(corpus)
        assert result["n"] >= 2
    finally:
        corpus.chmod(0o755)


def test_measuring_a_read_only_file_needs_no_write_handle(corpus):
    target = corpus / "one.txt"
    target.chmod(0o444)
    try:
        result = text_quality(target, include_textdescriptives=False)
        assert result["n"] > 0
    finally:
        target.chmod(0o644)
