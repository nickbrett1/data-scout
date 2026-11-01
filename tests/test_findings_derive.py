"""``derive_finding`` derives a line; it does not copy one that already exists.

The defect this file is written against: a ``reproduce()`` helper that matched a
measurement against lines already in ``findings.jsonl`` and appended a **copy**
of the matched record. From an empty ledger it raised on the first call, and
every line it appended duplicated one that was already there.

So this test starts where that code could not: a workspace path where
``findings.jsonl`` does not exist at all. The expectation table below is written
out by hand from the twelve measurements — tool, unit, params, value, n and the
promotion decision — and never from anything ``derive_finding`` returns, so a
drift in what a derived line *says* fails here instead of quietly becoming the
new expectation.

Canonicalisation is ``data_scout.ledger.normalized_line``, the same function the
render gate uses: there is deliberately no second implementation of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_scout import workspace
from data_scout.ledger import Ledger, normalized_line
from data_scout.tools import derive_finding

DATASET_ID = "demo-dataset"
SESSION = 1
LEDGER_FILENAME = "findings.jsonl"

# --------------------------------------------------------------------------
# the measurements, as measured
# --------------------------------------------------------------------------

ROOT_SHARE_UNIT = "share of files in the largest mime bucket"
DATA_SHARE_UNIT = "share of files under data/ in the largest mime bucket"
EXTENSIONLESS_UNIT = "files with no extension"
ZERO_BYTE_UNIT = "zero-byte files"
NO_TEXT_UNIT = "documents without a text layer"
FONTS_UNIT = "fonts"
ENCRYPTED_UNIT = "documents with Encrypted: yes"
IMAGES_UNIT = "images across the PDFs"
IMAGES_PER_PAGE_UNIT = "images per page"
IMAGES_PER_PAGE_ALT_UNIT = "images per page (alternate denominator)"
EXIF_SHARE_UNIT = "share of images carrying camera metadata"
#: ``pdf_forensics``' own unit, from ``data_scout.registry``: the twelfth line
#: omits ``unit=`` so the registry default is the thing under test.
PDF_UNIT = "documents (PDFs)"

#: Manifests of the pinned file sets, as ``frame.MANIFEST_RULE`` computes them
#: (sha256 over "relpath<TAB>size<LF>"). Literal here so a line that loses its
#: manifest is a visible diff rather than a silent omission.
MANIFEST_ROOT = "4813494d137e1631bba301d5acab6e7bb7aa74ce1185d456565ef51d737677b2"
MANIFEST_DATA = "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7"
MANIFEST_PDFS = "b0ce975f9314b741951b9ad9b586f3784428146ed765a041696d0e98bd56ce5d"
MANIFEST_IMAGES = "21b2eed1e328a2c62fe4c17d51188bdea73450f29956dc5c8c95429313ddd72c"

PARAMS_ROOT = {"scope": ".", "max_files": None, "manifest": MANIFEST_ROOT}
PARAMS_DATA = {"scope": "data/", "max_files": None, "manifest": MANIFEST_DATA}
PARAMS_PDFS = {"scope": "data/", "max_files": None, "manifest": MANIFEST_PDFS}
PARAMS_IMAGES = {"scope": "data/", "max_files": None, "manifest": MANIFEST_IMAGES}
#: The twelfth measurement pins its page count beside the manifest.
PARAMS_PDFS_PAGES = {**PARAMS_PDFS, "pages": 1778}

ROOT_SHARE_REASON = (
    "unit error: 0.321429 is a share of files, and the line was filed under a "
    "count unit; it is re-derived rather than promoted"
)
EXIF_ZERO_REASON = (
    "real zero: the census ran over the images that exist, found none carrying "
    "camera metadata, and 0.0 of 0 is the measurement rather than a failed "
    "instrument"
)
ENCRYPTION_REASON = (
    "encryption caveat: an encrypted PDF reports no fonts and no images, so 0 "
    "documents without a text layer cannot separate 'no text' from 'not "
    "readable'; the page count in params is what the caveat was measured against"
)


# --------------------------------------------------------------------------
# the calling instruments
# --------------------------------------------------------------------------
#
# ``derive_finding`` reads the tool off the calling frame (``__name__``), so
# these helpers are named exactly after the instruments and stand in for the
# notebook's measurement functions. Nothing below passes ``tool=``: if the
# derivation stopped working, every line would raise instead of silently
# acquiring a new tool name.


def file_census(**kwargs: object) -> dict[str, object]:
    return derive_finding(**kwargs)


def pdf_forensics(**kwargs: object) -> dict[str, object]:
    return derive_finding(**kwargs)


def exif_census(**kwargs: object) -> dict[str, object]:
    return derive_finding(**kwargs)


def _derive_twelve(ledger: Ledger) -> None:
    """Derive the twelve measurements, in order, appending as we go."""
    assert not ledger.path.exists(), "the fixture must start with no ledger on disk"

    # 1. the root mime share, retired for a unit error
    first = file_census(
        unit=ROOT_SHARE_UNIT,
        value=0.321429,
        n=56,
        params=PARAMS_ROOT,
        promoted=False,
        retire_reason=ROOT_SHARE_REASON,
    )
    assert not ledger.path.exists(), "derive_finding must not create the ledger"
    ledger.append(first)
    assert ledger.path.exists(), "the first append is what creates the ledger"

    # 2. the mime share under data/
    ledger.append(
        file_census(
            unit=DATA_SHARE_UNIT,
            value=0.75,
            n=8,
            params=PARAMS_DATA,
            promoted=True,
        )
    )

    # 3. extensionless files
    ledger.append(
        file_census(
            unit=EXTENSIONLESS_UNIT,
            value=30,
            n=56,
            params=PARAMS_ROOT,
            promoted=True,
        )
    )

    # 4. zero-byte files
    ledger.append(
        file_census(
            unit=ZERO_BYTE_UNIT,
            value=1,
            n=56,
            params=PARAMS_ROOT,
            promoted=True,
        )
    )

    # 5. PDFs without a text layer
    ledger.append(
        pdf_forensics(
            unit=NO_TEXT_UNIT,
            value=0,
            n=6,
            params=PARAMS_PDFS,
            promoted=True,
        )
    )

    # 6. total fonts
    ledger.append(
        pdf_forensics(
            unit=FONTS_UNIT,
            value=201,
            n=6,
            params=PARAMS_PDFS,
            promoted=True,
        )
    )

    # 7. documents reporting Encrypted: yes
    ledger.append(
        pdf_forensics(
            unit=ENCRYPTED_UNIT,
            value=3,
            n=6,
            params=PARAMS_PDFS,
            promoted=True,
        )
    )

    # 8. images across the PDFs
    ledger.append(
        pdf_forensics(
            unit=IMAGES_UNIT,
            value=8643,
            n=6,
            params=PARAMS_PDFS,
            promoted=True,
        )
    )

    # 9. images per page
    ledger.append(
        pdf_forensics(
            unit=IMAGES_PER_PAGE_UNIT,
            value=40.845361,
            n=97,
            params=PARAMS_PDFS,
            promoted=True,
        )
    )

    # 10. images per page, the alternate denominator
    ledger.append(
        pdf_forensics(
            unit=IMAGES_PER_PAGE_ALT_UNIT,
            value=0.375,
            n=264,
            params=PARAMS_PDFS,
            promoted=True,
        )
    )

    # 11. images with EXIF — a real zero, not a failed instrument
    ledger.append(
        exif_census(
            unit=EXIF_SHARE_UNIT,
            value=0.0,
            n=0,
            params=PARAMS_IMAGES,
            promoted=False,
            retire_reason=EXIF_ZERO_REASON,
        )
    )

    # 12. documents (PDFs) — unit omitted, so the registry default applies
    ledger.append(
        pdf_forensics(
            value=0,
            n=3,
            params=PARAMS_PDFS_PAGES,
            promoted=False,
            retire_reason=ENCRYPTION_REASON,
        )
    )


# --------------------------------------------------------------------------
# what the twelve lines must say
# --------------------------------------------------------------------------

#: Invariant frame stamps: the dataset and session the run is scoped to, and the
#: absence of a sample on every one of these twelve lines.
_BASE = {"dataset_id": DATASET_ID, "session": SESSION, "sample": None}

#: The twelve expected canonical lines, declared field by field. ``promoted``
#: lines name the module the registry publishes for that instrument; the three
#: unpromoted ones carry no ``script`` and do carry their reason.
EXPECTED_LINES: list[dict[str, object]] = [
    _BASE
    | {
        "tool": "file_census",
        "tool_version": "1",
        "script": None,
        "unit": ROOT_SHARE_UNIT,
        "params": PARAMS_ROOT,
        "value": 0.321429,
        "n": 56,
        "promoted": False,
        "retire_reason": ROOT_SHARE_REASON,
    },
    _BASE
    | {
        "tool": "file_census",
        "tool_version": "1",
        "script": "data_scout.tools.file_census",
        "unit": DATA_SHARE_UNIT,
        "params": PARAMS_DATA,
        "value": 0.75,
        "n": 8,
        "promoted": True,
        "retire_reason": None,
    },
    _BASE
    | {
        "tool": "file_census",
        "tool_version": "1",
        "script": "data_scout.tools.file_census",
        "unit": EXTENSIONLESS_UNIT,
        "params": PARAMS_ROOT,
        "value": 30,
        "n": 56,
        "promoted": True,
        "retire_reason": None,
    },
    _BASE
    | {
        "tool": "file_census",
        "tool_version": "1",
        "script": "data_scout.tools.file_census",
        "unit": ZERO_BYTE_UNIT,
        "params": PARAMS_ROOT,
        "value": 1,
        "n": 56,
        "promoted": True,
        "retire_reason": None,
    },
    _BASE
    | {
        "tool": "pdf_forensics",
        "tool_version": "1",
        "script": "data_scout.tools.pdf_forensics",
        "unit": NO_TEXT_UNIT,
        "params": PARAMS_PDFS,
        "value": 0,
        "n": 6,
        "promoted": True,
        "retire_reason": None,
    },
    _BASE
    | {
        "tool": "pdf_forensics",
        "tool_version": "1",
        "script": "data_scout.tools.pdf_forensics",
        "unit": FONTS_UNIT,
        "params": PARAMS_PDFS,
        "value": 201,
        "n": 6,
        "promoted": True,
        "retire_reason": None,
    },
    _BASE
    | {
        "tool": "pdf_forensics",
        "tool_version": "1",
        "script": "data_scout.tools.pdf_forensics",
        "unit": ENCRYPTED_UNIT,
        "params": PARAMS_PDFS,
        "value": 3,
        "n": 6,
        "promoted": True,
        "retire_reason": None,
    },
    _BASE
    | {
        "tool": "pdf_forensics",
        "tool_version": "1",
        "script": "data_scout.tools.pdf_forensics",
        "unit": IMAGES_UNIT,
        "params": PARAMS_PDFS,
        "value": 8643,
        "n": 6,
        "promoted": True,
        "retire_reason": None,
    },
    _BASE
    | {
        "tool": "pdf_forensics",
        "tool_version": "1",
        "script": "data_scout.tools.pdf_forensics",
        "unit": IMAGES_PER_PAGE_UNIT,
        "params": PARAMS_PDFS,
        "value": 40.845361,
        "n": 97,
        "promoted": True,
        "retire_reason": None,
    },
    _BASE
    | {
        "tool": "pdf_forensics",
        "tool_version": "1",
        "script": "data_scout.tools.pdf_forensics",
        "unit": IMAGES_PER_PAGE_ALT_UNIT,
        "params": PARAMS_PDFS,
        "value": 0.375,
        "n": 264,
        "promoted": True,
        "retire_reason": None,
    },
    _BASE
    | {
        "tool": "exif_census",
        "tool_version": "1",
        "script": None,
        "unit": EXIF_SHARE_UNIT,
        "params": PARAMS_IMAGES,
        "value": 0.0,
        "n": 0,
        "promoted": False,
        "retire_reason": EXIF_ZERO_REASON,
    },
    _BASE
    | {
        "tool": "pdf_forensics",
        "tool_version": "1",
        "script": None,
        "unit": PDF_UNIT,
        "params": PARAMS_PDFS_PAGES,
        "value": 0,
        "n": 3,
        "promoted": False,
        "retire_reason": ENCRYPTION_REASON,
    },
]


@pytest.fixture
def absent_ledger(roots, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A workspace directory that does not exist, so no ledger exists anywhere.

    ``findings.jsonl`` is absent *and* the tree it would live in is absent: a
    constructor that fished for the ledger could not find one by accident.
    """
    empty = roots.tmp / "workspace-that-does-not-exist"
    monkeypatch.setenv(workspace.WORKSPACE_ENV, str(empty))
    path = empty / DATASET_ID / LEDGER_FILENAME
    assert not path.exists()
    return path


def test_twelve_derived_lines_from_a_ledger_that_does_not_exist(absent_ledger):
    assert not absent_ledger.exists(), "there must be no ledger before the first append"

    ledger = Ledger(absent_ledger)
    _derive_twelve(ledger)

    lines = ledger.lines()
    assert len(lines) == 12, "twelve measurements, twelve lines, no copies"

    keys = [normalized_line(line) for line in lines]
    assert len(set(keys)) == 12, "every derived line is its own canonical key"

    declared = {normalized_line(row) for row in EXPECTED_LINES}
    assert set(keys) == declared, "a derived line drifted from what was measured"

    in_order = [normalized_line(row) for row in EXPECTED_LINES]
    assert keys == in_order, "the order of the measurements is part of the record"

    # The identity above was read off the calling frame, not defaulted. A caller
    # that is not a belt instrument is refused by name instead of inventing one.
    with pytest.raises(ValueError, match="is not in the belt"):
        derive_finding(unit="files", value=1, n=1, params={"scope": "."}, promoted=True)


def test_the_three_unpromoted_lines_stay_unpromoted(absent_ledger):
    ledger = Ledger(absent_ledger)
    _derive_twelve(ledger)
    findings = ledger.findings()

    unpromoted = [finding for finding in findings if not finding.promoted]
    assert len(unpromoted) == 3, "nine measurements are promoted; three are not"

    # Identified by their units, in ledger order, so a quietly promoted line
    # moves out of this list and fails rather than blending in.
    assert [finding.unit for finding in unpromoted] == [
        ROOT_SHARE_UNIT,
        EXIF_SHARE_UNIT,
        PDF_UNIT,
    ]
    assert [finding.value for finding in unpromoted] == [0.321429, 0.0, 0]
    assert "unit error" in unpromoted[0].retire_reason
    assert "real zero" in unpromoted[1].retire_reason
    assert "encryption caveat" in unpromoted[2].retire_reason
    assert all(finding.script is None for finding in unpromoted)

    # The twelfth line records what the caveat was measured against.
    assert unpromoted[2].params["pages"] == 1778

    promoted = [finding for finding in findings if finding.promoted]
    assert len(promoted) == 9
    assert all(finding.script == f"data_scout.tools.{finding.tool}" for finding in promoted)
    assert all(finding.retire_reason is None for finding in promoted)
