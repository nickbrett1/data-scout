"""The brief: from readings to a recommendation, and what would overturn it.

The report answers "what did the instruments say". This answers the only question
a reader actually arrived with: *what should I run on this corpus?* The tests
below pin the four decisions and the honesty rule — a fact the ledger does not
carry is reported as open, never as a zero.
"""

from __future__ import annotations

from data_scout import brief
from data_scout.ledger import Finding


def line(
    *,
    tool: str = "pdf_forensics",
    value: object = 0,
    unit: str = "documents",
    n: int | None = 1,
    params: dict[str, object] | None = None,
) -> Finding:
    return Finding(
        dataset_id="demo-dataset",
        session=1,
        unit=unit,
        tool=tool,
        tool_version="1",
        value=value,
        n=n,
        params=params or {},
        promoted=False,
        retire_reason="test fixture: not a real measurement",
    )


def document(path: str, *, pages: int, words: int) -> Finding:
    return line(
        tool="pdf_text_compare",
        value=0.9,
        unit="ratio",
        n=pages,
        params={"document": path, "pages": pages, "docling_words": words},
    )


def images(path: str, *, pages: int, per_page: float) -> Finding:
    return line(
        unit="images per page",
        n=pages,
        value=per_page,
        params={"document": path, "pages": pages},
    )


def tables(*, empty_parts: int, with_text: int) -> list[Finding]:
    return [
        line(unit="table parts with non-empty text", value=with_text, n=empty_parts),
    ]


PROSE_BOOK = document("data/novel.pdf", pages=500, words=170_000)
PLATE_BOOK = images("data/artbook.pdf", pages=97, per_page=40.8)


def test_a_prose_corpus_is_a_document_intelligence_job():
    corpus = brief.read([PROSE_BOOK])

    verdict = brief.verdict(corpus)

    assert verdict.system.startswith("Document intelligence")
    assert "chunk by section" in verdict.thesis
    assert verdict.not_to_do  # there is always something not to build


def test_a_picture_book_is_a_vision_job_not_a_text_job():
    corpus = brief.read([PLATE_BOOK])

    verdict = brief.verdict(corpus)

    assert verdict.system.startswith("Vision over page images")
    assert "caption" in verdict.thesis
    assert any("text pipeline" in item for item in verdict.not_to_do)


def test_a_table_shaped_corpus_is_the_one_case_for_sql():
    corpus = brief.read(tables(empty_parts=8, with_text=7))

    verdict = brief.verdict(corpus)

    assert verdict.system.startswith("Relational store")
    assert "SQL" in verdict.system


def test_two_populations_in_one_directory_are_routed_not_averaged():
    """The defect this prevents: one pipeline over a novel and an artbook."""
    corpus = brief.read([PROSE_BOOK, PLATE_BOOK] + tables(empty_parts=8, with_text=0))

    verdict = brief.verdict(corpus)

    assert verdict.system == "Document intelligence, routed per document"
    assert "two corpora in one directory" in verdict.thesis
    assert any("whole directory" in item for item in verdict.not_to_do)


def test_the_routing_table_reads_words_and_pictures_a_page():
    corpus = brief.read([PROSE_BOOK, PLATE_BOOK])

    kinds = {document.document: document.kind for document in corpus.documents}

    assert kinds == {"data/novel.pdf": "prose", "data/artbook.pdf": "plate"}
    assert corpus.documents[0].words_per_page == 340.0


def test_nothing_measured_is_an_honest_undecided_not_a_zero():
    corpus = brief.read([])

    verdict = brief.verdict(corpus)

    assert verdict.system.startswith("Undecided")
    assert "guess" in verdict.thesis
    # and the axes say which question to answer first
    shape, modality, _fidelity = brief.axes(corpus)
    assert shape.open and modality.open
    assert "measure" in shape.implies


def test_a_re_appended_fact_is_cited_once_at_its_newest_line():
    """The ledger is append-only; a re-run must not look like two answers."""
    corpus = brief.read([PROSE_BOOK, PROSE_BOOK])

    cited = [fact.line for fact in corpus.facts if fact.name.startswith("words_per_page")]

    assert cited == [2]


def test_parser_loss_is_its_own_axis_because_it_silently_drops_pages():
    lossy = line(unit="pages (of 896 measured)", value=20, n=896)

    _shape, _modality, fidelity = brief.axes(brief.read([PROSE_BOOK, lossy]))

    assert not fidelity.open
    assert "20 of 896" in fidelity.reading
    assert "fallback" in fidelity.implies


def test_the_brief_is_one_styled_document_with_the_way_back():
    html = brief.render_brief(brief.read([PROSE_BOOK, PLATE_BOOK]), dataset_id="demo-dataset")

    assert html.count("<!DOCTYPE html>") == 1
    assert html.count("data-data-scout") == 1
    assert '<nav class="data-scout-nav">' in html
    assert "What would change this" in html
