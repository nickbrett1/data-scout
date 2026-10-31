"""Forensics: the parsers are tested without poppler, the calls skip without it.

CI's ``build`` step installs the Python package but **not** the apt belt, so a
test that shells to ``pdffonts`` must skip there rather than fail on a machine
that is behaving correctly. Splitting the parse out is what keeps the interesting
half of this file running everywhere.
"""

from __future__ import annotations

import pytest

from data_scout.tools import exif_census, file_census, pdf_forensics
from data_scout.tools.pdf_forensics import (
    parse_pdffonts,
    parse_pdfimages,
    parse_pdfinfo,
)


def which_missing(*binaries: str) -> str | None:
    """The first binary not on PATH, or ``None``."""
    import shutil

    for binary in binaries:
        if shutil.which(binary) is None:
            return binary
    return None

PDFFONTS = """name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
ABCDEF+Arial                         TrueType          WinAnsi          yes no  no      12  0
Times-Roman                          Type 1            Standard         no  no  no       5  0
"""

PDFFONTS_EMPTY = """name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
"""

PDFINFO = """Title:           A Test Document
Producer:        pdfTeX-1.40.21
Creator:         TeX
CreationDate:    Mon Oct 26 00:00:00 2026 UTC
Pages:           3
Encrypted:       no
Page size:       612 x 792 pts (letter)
PDF version:     1.7
"""

PDFIMAGES = """page   num  type   width height color comp bpc  enc interp  object ID x-ppi y-ppi size ratio
--------------------------------------------------------------------------------------------
   1     0 image    1200  1600  rgb     3   8  jpeg   no        12  0   150   150  123K 1.1%
   2     1 image     800   600  gray    1   8  png    no        13  0    96    96   50K 2.0%
"""


# ---- parsers ---------------------------------------------------------------


def test_pdffonts_parses_the_font_list_as_first_class_fields():
    parsed = parse_pdffonts(PDFFONTS)
    assert parsed["font_count"] == 2
    assert parsed["embedded_fonts"] == 1
    assert [font["name"] for font in parsed["fonts"]] == ["ABCDEF+Arial", "Times-Roman"]
    assert parsed["fonts"][1]["type"] == "Type 1"
    assert parsed["fonts"][1]["embedded"] is False


def test_zero_fonts_is_visible_as_zero_not_as_a_missing_field():
    parsed = parse_pdffonts(PDFFONTS_EMPTY)
    assert parsed["font_count"] == 0
    assert parsed["fonts"] == []


def test_pdfinfo_parses_the_decisive_fields():
    parsed = parse_pdfinfo(PDFINFO)
    assert parsed["pages"] == 3
    assert parsed["encrypted"] is False
    assert parsed["pdf_version"] == "1.7"
    assert parsed["producer"] == "pdfTeX-1.40.21"


def test_pdfimages_parses_counts_and_encodings():
    parsed = parse_pdfimages(PDFIMAGES)
    assert parsed["image_count"] == 2
    assert parsed["image_encodings"] == {"jpeg": 1, "png": 1}
    assert parsed["images"][0]["width"] == 1200


# ---- the instrument --------------------------------------------------------


def test_pdf_forensics_skips_when_poppler_is_absent(roots):
    """The contract's own consequence: absent binaries skip, they do not fail."""
    missing = which_missing("pdffonts", "pdfinfo", "pdfimages")
    pdf = roots.dataset("pdfs")
    pdf.mkdir()
    (pdf / "scan.pdf").write_bytes(b"%PDF-1.4\n% not a real pdf\n")
    if missing is not None:
        pytest.skip(f"{missing} is not installed here; CI's build step has no apt belt either")
    result = pdf_forensics(pdf / "scan.pdf")
    assert result["n"] == 1
    assert result["font_count"] == 0


def test_exif_census_skips_when_exiftool_is_absent(roots):
    missing = which_missing("exiftool")
    directory = roots.dataset("photos")
    directory.mkdir()
    (directory / "a.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    if missing is not None:
        pytest.skip("exiftool is not installed here; CI's build step has no apt belt either")
    result = exif_census(directory)
    assert result["n"] >= 1


def test_exif_census_on_an_empty_directory_is_zero_not_an_error(roots):
    empty = roots.dataset("empty")
    empty.mkdir()
    result = exif_census(empty)
    assert result["n"] == 0
    assert result["value"] == 0.0


# ---- file_census -----------------------------------------------------------


def test_file_census_counts_the_tree_and_names_the_dominant_bucket(dataset):
    missing = which_missing("file")
    if missing is not None:
        pytest.skip("file/libmagic1 is not installed here; CI's build step has no apt belt either")
    result = file_census(dataset)
    assert result["n"] >= 2
    assert result["dominant_mime"] == "text/plain"
    assert result["value"] > 0.5
    assert result["bytes_total"] > 0
    assert result["by_extension"].get(".txt") == 1
    assert result["zero_byte"] == 0


def test_file_census_reports_a_truncated_walk_as_truncated(dataset):
    missing = which_missing("file")
    if missing is not None:
        pytest.skip("file/libmagic1 is not installed here; CI's build step has no apt belt either")
    result = file_census(dataset, max_files=1)
    assert result["n"] == 1
    assert result["truncated_walk"] is True


def test_file_census_on_a_missing_path_is_an_empty_census(roots):
    result = file_census(roots.tmp / "nope")
    assert result["n"] == 0


def test_file_census_without_the_file_binary_refuses_rather_than_guessing(dataset, monkeypatch):
    """The absent-binary path is measured in CI, where the belt really is absent.

    "no files" and "no ``file``" are the same number and opposite facts, so the
    instrument must refuse and name the package rather than return a zero.
    """
    import shutil

    from data_scout.tools._common import ToolUnavailable

    real_which = shutil.which
    monkeypatch.setattr(
        shutil, "which", lambda name, *a, **k: None if name == "file" else real_which(name, *a, **k)
    )
    with pytest.raises(ToolUnavailable) as exc:
        file_census(dataset)
    assert "libmagic1" in str(exc.value)
    assert "`file` binary" in str(exc.value)
