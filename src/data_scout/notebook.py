"""Notebook helpers — the agent writes cells, ``report-render`` executes them.

There is no interactive kernel in the image (design §2.4), so the agent writes
``notebooks/session-N.ipynb`` with ``nbformat`` and asks ``report-render`` to
execute it. That is also why the kernelspec stamp lives in ``render`` *and*
here: ``nbconvert --execute`` does not care whether a notebook declares a
kernel, and a person opening one in JupyterLab does — the first thing they see
without it is a "Select Kernel" prompt.

**Do not generate a notebook from a call log.** A trace is not an analysis; the
notebook is the record of reasoning and ``findings.jsonl`` is the record of
measurements.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import nbformat
from nbformat import NotebookNode

DEFAULT_KERNEL = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}


def stamp_kernelspec(
    notebook: NotebookNode,
    *,
    name: str = DEFAULT_KERNEL["name"],
    display_name: str = DEFAULT_KERNEL["display_name"],
    language: str = DEFAULT_KERNEL["language"],
) -> NotebookNode:
    """Give the notebook a kernelspec so opening it does not prompt.

    Idempotent, and it overwrites an existing spec: ``report-render`` is the
    component that actually runs a kernel and knows which one, so its answer is
    the one that should survive.
    """
    notebook.metadata.setdefault("kernelspec", {})
    notebook.metadata["kernelspec"].update({"name": name, "display_name": display_name, "language": language})
    language_info = notebook.metadata.setdefault("language_info", {})
    language_info.setdefault("name", language)
    return notebook


def markdown_cell(source: str) -> NotebookNode:
    return nbformat.v4.new_markdown_cell(source)


def code_cell(source: str, *, tags: Iterable[str] = ()) -> NotebookNode:
    cell = nbformat.v4.new_code_cell(source)
    tags = list(tags)
    if tags:
        cell.metadata.setdefault("tags", []).extend(tags)
    return cell


def new_notebook(cells: Iterable[NotebookNode] = (), **kernel: str) -> NotebookNode:
    notebook = nbformat.v4.new_notebook(cells=list(cells))
    stamp_kernelspec(notebook, **kernel)
    return notebook


def write_notebook(path: str | os.PathLike[str], notebook: NotebookNode) -> Path:
    """Write the notebook, stamped, creating its directory. A workspace write."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp_kernelspec(notebook)
    nbformat.write(notebook, str(target))
    return target


def read_notebook(path: str | os.PathLike[str]) -> NotebookNode:
    return nbformat.read(str(path), as_version=4)


def has_kernelspec(notebook: NotebookNode) -> bool:
    spec: dict[str, Any] = notebook.metadata.get("kernelspec") or {}
    return bool(spec.get("name") and spec.get("display_name") and spec.get("language"))
