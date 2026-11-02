"""The brief: from readings to a recommendation, and what would overturn it.

The report answers "what did the instruments say". This answers the only question
a reader actually arrived with: *what should I run on this corpus?* The brief no
longer holds the judgement itself — it reads facts and asks the catalogue — so
these tests pin the reading (facts and their citations) and the rendering (all
profiles shown, disqualified ones closed, the verdict naming the winner).
"""

from __future__ import annotations

from data_scout import brief, profiles
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
        tool="partition_doc",
        value=0.9,
        unit="ratio",
        n=pages,
        params={"document": path, "pages": pages, "docling_words": words},
    )


def images(path: str, *, pages: int, per_page: float) -> Finding:
    return line(unit="images per page", n=pages, value=per_page, params={"document": path, "pages": pages})


def tables(*, parts: int, with_text: int) -> Finding:
    return line(unit="table parts with non-empty text", value=with_text, n=parts)


def text_layer(*, without: int, total: int) -> Finding:
    return line(value=without, n=total, unit="documents (PDFs)")


PROSE_BOOK = document("data/novel.pdf", pages=500, words=170_000)
PLATE_BOOK = images("data/artbook.pdf", pages=97, per_page=40.8)
HAS_TEXT = text_layer(without=0, total=6)


def winner_for(findings: list[Finding]) -> profiles.Assessment | None:
    corpus = brief.read(findings)
    return profiles.recommendation(profiles.evaluate(corpus))


def test_a_prose_corpus_recommends_literature_mining():
    winner = winner_for([PROSE_BOOK, HAS_TEXT, tables(parts=8, with_text=0)])

    assert winner is not None
    assert winner.profile.id == "literature-mining"
    assert winner.headline == "fits 3/3"


def test_a_picture_book_recommends_vision_and_not_a_text_pipeline():
    winner = winner_for([PLATE_BOOK])

    assert winner is not None
    assert winner.profile.id == "visual-search"


def test_a_table_shaped_corpus_is_the_one_case_for_sql():
    corpus = brief.read([tables(parts=10, with_text=9), HAS_TEXT])
    scored = {assessment.profile.id: assessment for assessment in profiles.evaluate(corpus)}

    assert scored["warehouse"].outcome == "fits"
    assert scored["literature-mining"].outcome == "disqualified"


def test_the_artbook_beside_a_novel_is_routed_not_averaged_away():
    findings = [PROSE_BOOK, PLATE_BOOK, HAS_TEXT, tables(parts=8, with_text=0)]
    winner = winner_for(findings)
    html = brief.render_brief(brief.read(findings))

    assert winner is not None and winner.profile.id == "literature-mining"
    # the note is read from the same routing the table shows, so the two agree
    assert "1 of 2 documents is outside this profile" in html
    assert "artbook.pdf → visual search" in html
    # and the vision profile is not silently dropped: it is partial, with the document named
    assert "partial — 1 document" in html


def test_nothing_measured_is_undecided_and_says_so():
    corpus = brief.read([])

    assert profiles.recommendation(profiles.evaluate(corpus)) is None
    html = brief.render_brief(corpus)
    assert "Undecided — measure before building" in html


def test_facts_carry_the_lines_they_came_from():
    corpus = brief.read([PROSE_BOOK, HAS_TEXT, tables(parts=8, with_text=0)])

    assert corpus.fact("words_per_page_median").lines == (1,)
    assert corpus.fact("text_layer_share").lines == (2,)
    assert corpus.fact("table_share").lines == (3,)


def test_a_re_appended_fact_is_cited_once_at_its_newest_line():
    """The ledger is append-only; a re-run must not look like two answers."""
    corpus = brief.read([PROSE_BOOK, PROSE_BOOK])

    assert len([fact for fact in corpus.facts if fact.name == "words_per_page_median"]) == 1
    assert corpus.fact("words_per_page_median").lines == (2,)


def test_the_text_layer_question_is_not_confused_with_encryption():
    """All three share a unit; only one of them answers 'is there text?'."""
    encrypted = line(value=3, n=6, unit="documents (PDFs)", params={"attribute": "pdfinfo Encrypted: yes"})
    blocked = line(
        value=0,
        n=3,
        unit="documents (PDFs)",
        params={"probe": "pdftotext first+last page", "variants": ["no -upw"]},
    )

    corpus = brief.read([encrypted, blocked])

    assert corpus.fact("text_layer_share") is None


def test_the_brief_is_one_styled_document_showing_every_profile():
    # table parts measured and carrying no text: a warehouse is ruled *out*, not unmeasured
    corpus = brief.read([PROSE_BOOK, PLATE_BOOK, tables(parts=8, with_text=0)])
    html = brief.render_brief(corpus, dataset_id="demo-dataset")

    assert html.count("<!DOCTYPE html>") == 1
    assert '<nav class="data-scout-nav">' in html
    for profile in profiles.CATALOGUE:
        assert f'id="p-{profile.id}"' in html
    # a ruled-out profile is one line of information, not a wall
    assert 'class="prof o-disqualified" id="p-warehouse"' in html
    assert 'class="prof o-disqualified" id="p-warehouse" open' not in html


def test_the_per_document_table_says_how_much_of_the_corpus_it_covers():
    """Four rows over a corpus of eight is a gap, not a corpus of four."""
    frame = line(tool="frame", value=1000, unit="files", params={"identity_files": 8})
    corpus = brief.read([frame, PROSE_BOOK, PLATE_BOOK, HAS_TEXT])

    html = brief.render_brief(corpus)

    assert "Per document — 2 of 8 profiled" in html
    assert "Only 2 of 8 documents have a per-document reading" in html
    assert "a per-document reading for the 6 documents not yet profiled" in html


def test_the_brief_stops_at_the_decision():
    """Advice and provenance live on /profiles and in the report, not here."""
    html = brief.render_brief(brief.read([PROSE_BOOK, HAS_TEXT, tables(parts=8, with_text=0)]))

    for heading in ("First moves", "What would change this", "Where this comes from"):
        assert heading not in html


def test_the_document_table_says_where_each_document_goes():
    """A row a reader can act on: this book is not this corpus's job."""
    corpus = brief.read([PROSE_BOOK, PLATE_BOOK, HAS_TEXT, tables(parts=8, with_text=0)])

    html = brief.render_brief(corpus)

    assert "goes to" in html
    assert "literature-mining" in html
    assert "Not worth splitting" in html or "Worth splitting" in html


def test_the_document_table_cannot_push_a_phone_page_wider():
    """A six-column table of long file names is wider than a phone.

    Two guarantees, because the second is the one a reader feels: the table sits
    in a scroll container so it can never widen the page, and every reading
    carries a label so the row can stack as a card below 40rem.
    """
    findings = [
        line(tool="frame", value=1000, unit="files", params={"identity_files": 1}),
        PROSE_BOOK,
        HAS_TEXT,
    ]
    html = brief.render_brief(brief.read(findings))

    assert '<div class="table-scroll"><table class="brief-table">' in html
    for label in ("pages", "words / page", "images / page", "reads as", "goes to"):
        assert f'data-label="{label}"' in html
