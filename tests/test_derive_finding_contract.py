"""``derive_finding`` must accept the param shapes the live ledger already has.

This is the test the ryzen deploy needed and did not have. The notebook that
produced ``findings.jsonl`` lives in ryzen's workspace, not in this repo, so no
test bound its call sites to ``derive_finding``'s contract. The merged
``scope``-mandatory check passed locally — where every test wrote
``params["scope"]`` — and then rejected all twelve of the notebook's shapes,
because **no existing ledger line carries a ``scope`` key**. Scope is expressed
in the vocabulary the measurement actually used: ``path`` (an absolute path),
``document``, ``documents``, ``file``, ``pages``.

The twelve rows below are transcribed literally from the live ledger (line,
tool, ``n``, params). The assertion is deliberately the stronger one available
without ryzen: each shape is accepted, the twelve canonical keys are distinct,
and the ``params`` in the derived line are **exactly** the shape supplied — an
absent ``scope`` stays absent, because a default written into ``params`` would
change the digest of every line that predates it. (Hard-coding the on-disk
digests would also catch a silent key change, but it needs the ``value``,
``unit`` and promotion fields of each line, which the brief did not supply.)
"""

from __future__ import annotations

import json

import pytest

from data_scout.ledger import normalized_line
from data_scout.tools import derive_finding

#: (ledger line, tool, n, params) — verbatim from the live ledger.
LIVE_LEDGER_SHAPES = [
    (2, "file_census", 56, {"follow_symlinks": False, "max_files": None, "path": "/datasets/vg-books-2026-10"}),
    (3, "file_census", 8, {"follow_symlinks": False, "max_files": None, "path": "/datasets/vg-books-2026-10/data"}),
    (4, "file_census", 56, {"measure": "extensionless_files", "path": "/datasets/vg-books-2026-10"}),
    (5, "file_census", 56, {"measure": "zero_byte_files", "path": "/datasets/vg-books-2026-10"}),
    (6, "pdf_forensics", 6, {"path": "/datasets/vg-books-2026-10/data", "suffix": ".pdf"}),
    (7, "pdf_forensics", 6, {"aggregate": "font_count summed across PDFs", "path": "/datasets/vg-books-2026-10/data"}),
    (8, "pdf_forensics", 6, {"attribute": "pdfinfo Encrypted: yes", "path": "/datasets/vg-books-2026-10/data"}),
    (9, "pdf_forensics", 6, {"attribute": "pdfimages -list image_count summed across PDFs", "path": "/datasets/vg-books-2026-10/data"}),
    (10, "pdf_forensics", 97, {"document": "data/BG3_Artbook_High_Res.pdf", "pages": 97, "path": "/datasets/vg-books-2026-10/data"}),
    (11, "pdf_forensics", 264, {"document": "data/GREMLIN_DLC_2024_PDF.pdf", "pages": 264, "path": "/datasets/vg-books-2026-10/data"}),
    (12, "exif_census", 0, {"path": "/datasets/vg-books-2026-10"}),
    (13, "pdf_forensics", 3, {"pages": 1778, "path": "/datasets/vg-books-2026-10/data", "probe": "pdftotext first+last page", "variants": ["no-upw", "-upw \"\""]}),
]


def _derive(tool: str, n: int, params: dict) -> dict:
    """Derive one line from a live-ledger param shape.

    ``tool`` is passed explicitly because this module is not the instrument
    belt: the frame that ``derive_finding`` reads for a tool identity must not
    be satisfied by a test helper.
    """
    return derive_finding(
        tool=tool,
        value=1,
        n=n,
        params=params,
        promoted=False,
        retire_reason="contract: a live-ledger param shape must validate",
    )


def test_the_twelve_live_ledger_shapes_are_accepted():
    """One test per shape, so a rejection names the line it killed."""
    for line, tool, n, params in LIVE_LEDGER_SHAPES:
        derived = _derive(tool, n, params)
        assert derived["tool"] == tool, f"line {line}"
        assert derived["n"] == n, f"line {line}"


def test_the_twelve_live_ledger_shapes_are_distinct_keys():
    """Twelve measurements, twelve canonical keys — scope-mandatory gave zero.

    This is the property that would have caught the ryzen failure: with
    ``params["scope"]`` required, every one of these raised before producing any
    key at all.
    """
    keys = []
    for _line, tool, n, params in LIVE_LEDGER_SHAPES:
        derived = _derive(tool, n, params)
        keys.append(normalized_line(derived))

    assert len(keys) == 12
    assert len(set(keys)) == 12, "two live-ledger shapes collapsed to one key"


def test_an_absent_scope_stays_absent_and_a_supplied_one_is_preserved():
    """No default is invented: the historical params must hash identically."""
    for line, tool, n, params in LIVE_LEDGER_SHAPES:
        key = json.loads(normalized_line(_derive(tool, n, params)))
        assert key["params"] == params, f"line {line} params changed on the way in"
        assert "scope" not in key["params"], f"line {line} gained an invented scope"

    # A literal ``scope`` is still accepted: dropping the *requirement* is not
    # dropping the key for callers that already use it.
    with_scope = _derive("file_census", 56, {"scope": ".", "max_files": None})
    assert json.loads(normalized_line(with_scope))["params"]["scope"] == "."


def test_scope_alone_is_not_a_substitute_for_params():
    """``params`` is still required; only the ``scope`` key stopped being so."""
    with pytest.raises(ValueError, match=r"`params` is required"):
        derive_finding(tool="file_census", value=1, n=1, promoted=False, retire_reason="x")


def test_n_is_required_by_the_signature_not_only_by_the_docstring():
    """The prose said "no default" while the signature said ``= None``.

    ``inspect.signature`` is the contract a caller can see; the docstring is
    what was asserted from and got wrong. They must agree.
    """
    import inspect

    parameter = inspect.signature(derive_finding).parameters["n"]
    assert parameter.default is inspect.Parameter.empty, "n must have no default"
