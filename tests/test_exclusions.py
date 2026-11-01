"""Version control is not the dataset, and every walk must agree on that.

The defect this pins: a dataset sits in a checkout, so walking the whole tree
measured git's index and DVC's lockfiles alongside the books. The numbers were
true and useless — the most common file type was git's text hooks, and a
tree-wide hash moved every time ``.dvc/tmp/lock`` ticked. There is one rule
(:mod:`data_scout.exclusions`) and three consumers: the frame, the toolbelt's
walks, and the inventory.
"""

from __future__ import annotations

from pathlib import Path

from data_scout import exclusions, inventory, render
from data_scout.ledger import Finding
from data_scout.tools import _common


def make_checkout(root: Path) -> Path:
    """A tiny dataset as it actually arrives: a few books wrapped in VCS state."""
    directory = root / "demo-dataset"
    (directory / "data").mkdir(parents=True)
    (directory / "data" / "one.pdf").write_bytes(b"x" * 10)
    (directory / "data" / "two.epub").write_bytes(b"y" * 20)
    # git's bookkeeping: large, texty, and not a book
    (directory / ".git" / "objects" / "ab").mkdir(parents=True)
    (directory / ".git" / "index").write_text("DIRC" * 1000, encoding="utf-8")
    (directory / ".git" / "objects" / "ab" / "cdef").write_bytes(b"z" * 500)
    # DVC's bookkeeping: rewrites itself while the payload sits still
    (directory / ".dvc" / "tmp").mkdir(parents=True)
    (directory / ".dvc" / "tmp" / "lock").write_text("8", encoding="utf-8")
    # the host's claim about the payload, written when a pin is recorded
    (directory / "pin.json").write_text('{"files": 2}', encoding="utf-8")
    return directory


def test_the_rule_names_the_vcs_state_and_the_pin():
    assert exclusions.EXCLUDED_DIRS == frozenset({".git", ".dvc"})
    assert exclusions.EXCLUDED_FILES == frozenset({"pin.json"})
    assert exclusions.EXCLUSION_NAMES == ("git-internals", "dvc-internals", "dataset-pin")


def test_a_walk_never_descends_into_version_control(roots):
    directory = make_checkout(roots.datasets)

    walked = {path.relative_to(directory).as_posix() for path in _common.iter_files(directory)}

    assert walked == {"data/one.pdf", "data/two.epub"}


def test_the_index_counts_the_dataset_not_the_checkout(roots):
    directory = make_checkout(roots.datasets)

    summary = inventory.summarize("demo-dataset", directory)

    assert summary.files == 2
    assert summary.bytes == 30
    assert dict(summary.extensions) == {".pdf": 1, ".epub": 1}


def _line(value: object) -> Finding:
    return Finding(
        dataset_id="demo-dataset",
        session=1,
        unit="files",
        tool="file_census",
        tool_version="1",
        value=value,
        n=1,
        promoted=False,
        retire_reason="measured git's index, not the corpus",
    )


def test_a_marker_hides_the_lines_about_the_checkout_and_says_so(roots):
    findings = [_line("text/plain"), _line(2)]
    markdown = "lede\n\n<!-- exclude: 1 -->\n\n<!-- finding: 2 -->\n\nTwo books.\n"

    html = render.render_markdown(markdown, title="t", findings=findings)

    assert "text/plain" not in html
    assert "Not shown: 1 finding measured version-control internals" in html
