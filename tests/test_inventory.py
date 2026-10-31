"""The inventory is cheap triage, and it must stay bounded and honest."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from data_scout import inventory


def make_dataset(root: Path, dataset_id: str = "demo-dataset") -> Path:
    directory = root / dataset_id
    (directory / "docs").mkdir(parents=True)
    (directory / "a.txt").write_text("hello", encoding="utf-8")
    (directory / "b.txt").write_text("world!", encoding="utf-8")
    (directory / "docs" / "c.pdf").write_bytes(b"x" * 20)
    return directory


def test_a_summary_counts_files_directories_bytes_and_extensions(roots):
    make_dataset(roots.datasets)
    summary = inventory.summarize("demo-dataset", roots.dataset())

    assert summary.files == 3
    assert summary.directories == 1
    assert summary.bytes == 31
    assert dict(summary.extensions) == {".txt": 2, ".pdf": 1}
    assert summary.frame is False
    assert summary.truncated is False
    assert summary.error is None
    assert summary.newest is not None


def test_the_frame_heading_becomes_the_display_name(roots):
    directory = make_dataset(roots.datasets)
    (directory / "frame.md").write_text("# Harbour Archive\n\nnotes\n", encoding="utf-8")

    summary = inventory.summarize("demo-dataset", directory)

    assert summary.frame is True
    assert summary.title == "Harbour Archive"
    assert summary.name == "Harbour Archive"
    assert summary.as_record()["dataset_id"] == "demo-dataset"


def test_discover_sorts_and_skips_hidden_directories_and_loose_files(roots):
    make_dataset(roots.datasets, "zebra")
    make_dataset(roots.datasets, "alpha")
    (roots.datasets / ".hidden").mkdir()
    (roots.datasets / "stray.txt").write_text("no", encoding="utf-8")

    datasets, error = inventory.discover(roots.datasets)

    assert error is None
    assert [item.dataset_id for item in datasets] == ["alpha", "zebra"]


def test_a_missing_root_is_an_error_not_an_empty_inventory(roots):
    datasets, error = inventory.discover(roots.tmp / "not-there")

    assert datasets == []
    assert error is not None
    assert "not readable" in error


def test_the_walk_is_bounded_and_says_so(roots):
    directory = make_dataset(roots.datasets)

    summary = inventory.summarize("demo-dataset", directory, max_entries=2)

    assert summary.truncated is True
    assert summary.files + summary.directories == 2


def test_a_workspace_only_entry_has_no_root_facts():
    record = inventory.WorkspaceOnlyDataset("demo-dataset").as_record()

    assert record["in_root"] is False
    assert record["files"] is None
    assert record["bytes"] is None
    assert record["dataset_id"] == "demo-dataset"


def test_symlinks_are_not_followed(roots, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("x" * 500, encoding="utf-8")
    directory = make_dataset(roots.datasets)
    try:
        os.symlink(outside, directory / "linked")
    except OSError:
        pytest.skip("symlinks are not available here")

    summary = inventory.summarize("demo-dataset", directory)

    assert summary.files == 3
    assert summary.bytes == 31


def test_format_bytes_is_short_and_relative():
    assert inventory.format_bytes(None) == "—"
    assert inventory.format_bytes(512) == "512 B"
    assert inventory.format_bytes(2048) == "2.0 KB"
    assert inventory.format_bytes(5 * 1024**3) == "5.0 GB"
