"""Shared fixtures: two isolated roots, and one small corpus to measure.

Every test runs against a throwaway corpora root and workspace, so no test can
touch a real corpus and no test depends on one being mounted. The environment is
patched rather than the code being told, because the code reads the environment —
that is the contract (``data-scout-convention`` §8).
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from data_scout import workspace

PROSE = (
    "The harbour was quiet at that hour, and the fishing boats lay still against "
    "the wall while the tide went out beneath them. Nobody had told the harbour "
    "master that the ferry would be late, and he took the news badly, walking the "
    "length of the pier twice before he would speak to anyone at all."
)

CORPUS_ID = "demo-corpus"


@dataclass
class Roots:
    tmp: Path
    corpora: Path
    workspace: Path

    def corpus(self, corpus_id: str = CORPUS_ID) -> Path:
        return self.corpora / corpus_id


@pytest.fixture(autouse=True)
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Roots:
    """Isolate both roots, and make the optional services absent by default."""
    corpora = tmp_path / "corpora"
    ws = tmp_path / "workspace"
    corpora.mkdir()
    ws.mkdir()
    monkeypatch.setenv(workspace.CORPORA_ROOT_ENV, str(corpora))
    monkeypatch.setenv(workspace.WORKSPACE_ENV, str(ws))
    monkeypatch.setenv(workspace.CORPUS_ID_ENV, CORPUS_ID)
    monkeypatch.setenv(workspace.SESSION_ENV, "1")
    monkeypatch.setenv("DATA_SCOUT_CACHE_DIR", str(ws / CORPUS_ID / "cache"))
    # Tier-2 services are absent unless a test says otherwise: a suite that
    # silently spends money because a developer happened to export a key is a
    # suite nobody runs twice.
    for name in ("DOCLING_URL", "LITELLM_BASE_URL", "LITELLM_API_KEY", "DATA_SCOUT_EMBED_MODEL", "DATA_SCOUT_VISION_EMBED_MODEL"):
        monkeypatch.delenv(name, raising=False)
    return Roots(tmp=tmp_path, corpora=corpora, workspace=ws)


@pytest.fixture
def corpus(roots: Roots) -> Path:
    """A profiled corpus: a directory with a ``frame.md`` that names itself."""
    directory = roots.corpus()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / workspace.FRAME_FILENAME).write_text(
        f"""---
corpus_id: {CORPUS_ID}
unit: file
lenses:
  - text
  - structure
tier_ceiling: 1
created: 2026-10-26
---

# {CORPUS_ID}

A two-file test corpus.
""",
        encoding="utf-8",
    )
    (directory / "one.txt").write_text(PROSE, encoding="utf-8")
    (directory / "two.md").write_text("# Chapter One\n\n" + PROSE, encoding="utf-8")
    return directory


@pytest.fixture
def corpus_workspace(roots: Roots) -> Path:
    return workspace.ensure_workspace(CORPUS_ID)


def not_installed(module: str) -> bool:
    """True when ``module`` is absent — used to test the loud-failure path."""
    return importlib.util.find_spec(module) is None


def which_missing(*binaries: str) -> str | None:
    """The first binary not on PATH, or ``None``."""
    import shutil

    for binary in binaries:
        if shutil.which(binary) is None:
            return binary
    return None


def can_write(directory: Path) -> bool:
    """Whether this process may create a file in ``directory``.

    Used to skip the read-only assertion where the mount genuinely is not
    read-only (a host checkout), rather than asserting something the environment
    cannot prove.
    """
    return os.access(directory, os.W_OK)
