"""Phase-1 validation: fail loudly, never infer."""

from __future__ import annotations

import pytest

from data_scout import workspace

DATASET_ID = "demo-dataset"


def test_a_missing_dataset_fails_naming_the_resolved_root(roots):
    with pytest.raises(workspace.DatasetNotFound) as exc:
        workspace.dataset_path("no-such-dataset")
    assert str(roots.datasets) in str(exc.value)
    assert workspace.DATASETS_ROOT_ENV in str(exc.value)


def test_a_dataset_without_a_frame_is_unprofiled_not_inferred(roots):
    (roots.datasets / DATASET_ID).mkdir()
    with pytest.raises(workspace.DatasetUnprofiled) as exc:
        workspace.read_frame(DATASET_ID)
    assert "frame.md" in str(exc.value)


def test_the_frame_must_declare_the_id_it_is_addressed_by(dataset_workspace):
    (dataset_workspace / "frame.md").write_text(
        "---\ndataset_id: something-else\n---\n", encoding="utf-8"
    )
    with pytest.raises(workspace.WorkspaceError, match="two ids"):
        workspace.read_frame(DATASET_ID)


def test_the_frame_is_read_as_written(dataset_workspace):
    frame = workspace.read_frame(DATASET_ID)
    assert frame.dataset_id == DATASET_ID
    assert frame.unit == "file"
    assert frame.lenses == ("text", "structure")
    assert frame.tier_ceiling == 1


def test_a_dataset_id_cannot_be_a_path(roots):
    for bad in ("../etc", "a/b", "", "..", "with space"):
        with pytest.raises(workspace.WorkspaceError):
            workspace.validate_dataset_id(bad)


def test_ensure_workspace_is_idempotent(dataset_workspace):
    again = workspace.ensure_workspace(DATASET_ID)
    assert again == dataset_workspace
    for name in workspace.WORKSPACE_SUBDIRS:
        assert (dataset_workspace / name).is_dir()


def test_the_ledger_and_cache_live_in_the_workspace_not_the_dataset(roots, dataset):
    ledger = workspace.ledger_path(DATASET_ID)
    assert ledger.parent == roots.workspace / DATASET_ID
    assert not str(ledger).startswith(str(roots.datasets))
    assert workspace.cache_dir(DATASET_ID).parent == roots.workspace / DATASET_ID


def test_an_artifact_has_one_location_and_it_follows_the_convention(roots):
    """Report at the root; session HTML with its notebook under ``notebooks/``.

    This is the writer/reader contract as a test: ``report-render`` and the
    ``/report/`` index both resolve through this function.
    """
    report = workspace.artifact_path(DATASET_ID, "report-2026-10-31.html")
    session = workspace.artifact_path(DATASET_ID, "session-1.html")

    assert report.parent == roots.workspace / DATASET_ID
    assert session.parent == roots.workspace / DATASET_ID / "notebooks"
    assert report.name == "report-2026-10-31.html"
    assert session.name == "session-1.html"
    assert workspace.artifact_kind("report-2026-10-31.html") == "report"
    assert workspace.artifact_kind("session-1.html") == "session"
    assert workspace.artifact_kind("findings.jsonl") is None
    with pytest.raises(workspace.WorkspaceError):
        workspace.artifact_path(DATASET_ID, "findings.jsonl")


def test_artifact_paths_finds_both_locations_and_ignores_everything_else(dataset_workspace):
    (dataset_workspace / "report-2026-10-31.html").write_text("r", encoding="utf-8")
    (dataset_workspace / "frame.md").write_text("not an artifact", encoding="utf-8")
    notebooks = dataset_workspace / "notebooks"
    (notebooks / "session-1.html").write_text("s", encoding="utf-8")
    (notebooks / "session-1.ipynb").write_text("{}", encoding="utf-8")

    names = [path.name for path in workspace.artifact_paths(DATASET_ID)]
    assert names == ["report-2026-10-31.html", "session-1.html"]


def test_an_unset_dataset_id_is_refused_rather_than_guessed(monkeypatch):
    monkeypatch.delenv(workspace.DATASET_ID_ENV)
    with pytest.raises(workspace.NotScoped):
        workspace.current_dataset_id()
