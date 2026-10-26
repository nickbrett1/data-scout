"""Phase-1 validation: fail loudly, never infer."""

from __future__ import annotations

import pytest

from data_scout import workspace

CORPUS_ID = "demo-corpus"


def test_a_missing_corpus_fails_naming_the_resolved_root(roots):
    with pytest.raises(workspace.CorpusNotFound) as exc:
        workspace.corpus_path("no-such-corpus")
    assert str(roots.corpora) in str(exc.value)
    assert workspace.CORPORA_ROOT_ENV in str(exc.value)


def test_a_corpus_without_a_frame_is_unprofiled_not_inferred(roots):
    (roots.corpora / CORPUS_ID).mkdir()
    with pytest.raises(workspace.CorpusUnprofiled) as exc:
        workspace.corpus_path(CORPUS_ID)
    assert "frame.md" in str(exc.value)


def test_the_frame_must_declare_the_id_it_is_addressed_by(corpus):
    (corpus / "frame.md").write_text("---\ncorpus_id: something-else\n---\n", encoding="utf-8")
    with pytest.raises(workspace.WorkspaceError, match="two ids"):
        workspace.read_frame(CORPUS_ID)


def test_the_frame_is_read_as_written(corpus):
    frame = workspace.read_frame(CORPUS_ID)
    assert frame.corpus_id == CORPUS_ID
    assert frame.unit == "file"
    assert frame.lenses == ("text", "structure")
    assert frame.tier_ceiling == 1


def test_a_corpus_id_cannot_be_a_path(roots):
    for bad in ("../etc", "a/b", "", "..", "with space"):
        with pytest.raises(workspace.WorkspaceError):
            workspace.validate_corpus_id(bad)


def test_ensure_workspace_is_idempotent(corpus_workspace):
    again = workspace.ensure_workspace(CORPUS_ID)
    assert again == corpus_workspace
    for name in workspace.WORKSPACE_SUBDIRS:
        assert (corpus_workspace / name).is_dir()


def test_the_ledger_and_cache_live_in_the_workspace_not_the_corpus(roots, corpus):
    ledger = workspace.ledger_path(CORPUS_ID)
    assert ledger.parent == roots.workspace / CORPUS_ID
    assert not str(ledger).startswith(str(roots.corpora))
    assert workspace.cache_dir(CORPUS_ID).parent == roots.workspace / CORPUS_ID


def test_an_unset_corpus_id_is_refused_rather_than_guessed(monkeypatch):
    monkeypatch.delenv(workspace.CORPUS_ID_ENV)
    with pytest.raises(workspace.NotScoped):
        workspace.current_corpus_id()
