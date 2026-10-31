"""A data-scout run cannot modify the dataset. Asserted two ways.

The mount gives the guarantee in production; these tests state it as the
convention does — as the **absence** of a write handle, plus a live attempt that
must fail — so a future tool that forgets is caught before a session, not after.
"""

from __future__ import annotations

import ast
import os
import shutil
from pathlib import Path

import pytest

from data_scout import inventory, workspace
from data_scout.tools import file_census, text_quality

DATASET_ID = "demo-dataset"

TOOLS_DIR = Path(__file__).resolve().parents[1] / "src" / "data_scout" / "tools"

#: The only modules allowed to open anything for writing, and why:
#: ``findings_append`` appends to the workspace ledger, ``sample`` writes the
#: reproducibility anchor into ``samples/``. Both write inside the workspace
#: only; neither can reach the dataset.
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


def test_no_tool_opens_the_dataset_for_writing():
    offenders: dict[str, list[str]] = {}
    for path in sorted(TOOLS_DIR.glob("*.py")):
        if path.name in _WRITE_ALLOWED:
            continue
        modes = _write_modes(path)
        if modes:
            offenders[path.name] = modes
    assert offenders == {}, f"tools must not open anything for writing: {offenders}"


def test_the_two_writers_write_only_inside_the_workspace(dataset_workspace):
    """``findings_append`` and the sample anchor both resolve under the workspace."""
    from data_scout import workspace

    ledger = workspace.ledger_path("demo-dataset")
    samples = workspace.samples_dir("demo-dataset")
    root = workspace.workspace_root()
    assert ledger.is_relative_to(root)
    assert samples.is_relative_to(root)
    assert not str(ledger).startswith(str(workspace.datasets_root()))


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores mode bits; the mount is the real guarantee")
def test_a_write_to_a_read_only_dataset_fails_with_a_permission_error(dataset):
    dataset.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            (dataset / "should-not-exist.txt").write_text("nope", encoding="utf-8")
        # And the instruments still read it. The measurement half is skipped where
        # `file` is absent (CI's build step has no apt belt), but the permission
        # assertion above is the point of the test and always runs.
        if shutil.which("file") is None:
            pytest.skip("`file` is not installed here, so the read half cannot be demonstrated")
        result = file_census(dataset)
        assert result["n"] >= 2
    finally:
        dataset.chmod(0o755)


def test_measuring_a_read_only_file_needs_no_write_handle(dataset):
    target = dataset / "one.txt"
    target.chmod(0o444)
    try:
        result = text_quality(target, include_textdescriptives=False)
        assert result["n"] > 0
    finally:
        target.chmod(0o644)


# ---- item 1: the frame is state, and state lives in the workspace ----------


def test_read_frame_reads_the_frame_from_the_workspace_under_a_read_only_mount(
    dataset, dataset_workspace
):
    """The dataset is mounted ``:ro`` by design, so the frame cannot live in it.

    This is the acceptance test for the defect: on the pre-fix code
    ``read_frame`` looks for ``frame.md`` inside the dataset, so it raises
    ``DatasetUnprofiled`` here even though the frame exists — and this test
    fails. It must not pass before the fix.
    """
    dataset.chmod(0o555)
    try:
        assert not (dataset / workspace.FRAME_FILENAME).exists(), (
            "the fixture must put the frame in the workspace, not the dataset"
        )
        frame = workspace.read_frame(DATASET_ID)
        assert frame.dataset_id == DATASET_ID
        assert frame.raw.startswith("---")
        # The frame's home is the workspace, not the read-only dataset.
        assert frame.path == dataset_workspace
        assert not frame.path.is_relative_to(workspace.datasets_root())
    finally:
        dataset.chmod(0o755)


def test_read_frame_raises_unprofiled_when_the_workspace_has_no_frame(dataset):
    """A genuinely absent frame still fails loudly, with the convention's error."""
    with pytest.raises(workspace.DatasetUnprofiled) as exc:
        workspace.read_frame(DATASET_ID)
    assert "frame.md" in str(exc.value)
    assert "unprofiled" in str(exc.value).lower()


def test_dataset_path_resolves_a_dataset_that_has_no_frame(roots):
    """``dataset_path`` answers "does this dataset exist", and nothing else.

    It used to smuggle the frame-location rule in, which made the two modules
    that ask about frames answer with two different rules.
    """
    (roots.datasets / DATASET_ID).mkdir()
    assert workspace.dataset_path(DATASET_ID) == roots.datasets / DATASET_ID


def test_inventory_and_read_frame_agree_about_where_the_frame_is(dataset, dataset_workspace):
    """One rule, two callers: the flag and the reader cannot drift again."""
    summary = inventory.summarize(DATASET_ID, dataset)
    assert summary.frame is True
    assert summary.title == DATASET_ID
    assert workspace.read_frame(DATASET_ID).dataset_id == DATASET_ID


def test_inventory_says_no_frame_when_only_the_dataset_has_one(dataset):
    """A frame inside the dataset is not a frame: that location cannot even exist."""
    (dataset / workspace.FRAME_FILENAME).write_text(f"# {DATASET_ID}\n", encoding="utf-8")
    summary = inventory.summarize(DATASET_ID, dataset)
    assert summary.frame is False
