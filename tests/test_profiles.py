"""The interpretation layer: a closed catalogue scored against named facts.

These tests pin the four rules that make a brief trustworthy rather than merely
confident:

1. tri-state — a *failed* gate disqualifies, an *unknown* gate never does;
2. citation — nothing is pass or fail without a ledger line behind it;
3. reproducibility — the same corpus and catalogue version give the same matrix;
4. ordering — all profiles are shown, and only a profile that fits can win.
"""

from __future__ import annotations

import pytest

from data_scout import profiles
from data_scout.brief import Corpus, DocProfile, Fact


def fact(name: str, value: float, lines: tuple[int, ...] = ()) -> Fact:
    return Fact(name=name, value=value, observed=f"{name} = {value}", lines=lines)


def corpus(*findings: Fact, documents: tuple[DocProfile, ...] = ()) -> Corpus:
    return Corpus(facts=tuple(findings), documents=documents)


PROSE = DocProfile(document="data/novel.pdf", pages=500, words=170_000, lines=(1,))
PLATE = DocProfile(document="data/artbook.pdf", pages=97, images_per_page=40.8, lines=(2,))

#: Everything a prose corpus needs to be decided, each reading citing its line.
PROSE_CORPUS = corpus(
    fact("documents", 1, (1,)),
    fact("words_per_page_median", 340, (1,)),
    fact("text_layer_share", 1.0, (3,)),
    fact("table_share", 0.0, (4,)),
    fact("corpus_words", 170_000, (1,)),
    documents=(PROSE,),
)


def scored(corpus_: Corpus) -> dict[str, profiles.Assessment]:
    return {assessment.profile.id: assessment for assessment in profiles.evaluate(corpus_)}


# --- rule 1: tri-state -----------------------------------------------------


def test_an_unmeasured_gate_is_a_task_and_never_a_disqualification():
    """'Nobody measured it' is not 'measured and wrong'."""
    assessment = scored(corpus())[
        "literature-mining"
    ]

    assert assessment.outcome == "unknown"
    assert assessment.score is None
    assert all(cell.state == "unknown" for cell in assessment.gates)
    assert assessment.headline == "unknown — 3 to measure"


def test_a_failed_gate_disqualifies_even_when_everything_else_is_unknown():
    disqualified = scored(corpus(fact("table_share", 0.9, (7,))))["literature-mining"]

    assert disqualified.outcome == "disqualified"
    assert [cell.state for cell in disqualified.gates] == ["unknown", "unknown", "fail"]


def test_a_disqualified_profile_becomes_partial_when_one_document_still_fits():
    """The artbook beside a novel is a second job, not a rounding error."""
    mixed = corpus(
        fact("table_share", 0.9, (7,)),
        fact("text_layer_share", 0.2, (8,)),
        fact("words_per_page_median", 340, (1,)),
        documents=(PROSE, PLATE),
    )

    assessment = scored(mixed)["literature-mining"]

    assert assessment.outcome == "partial"
    assert assessment.partial_documents == ("data/novel.pdf",)
    assert assessment.headline == "partial — 1 document"


def test_no_documents_at_all_means_unknown_rather_than_partial():
    """Without per-document readings there is nothing to be partial about."""
    assessment = scored(corpus(fact("table_share", 0.9, (7,))))["visual-search"]

    assert assessment.outcome == "unknown"
    assert assessment.partial_documents == ()


# --- rule 2: citation ------------------------------------------------------


@pytest.mark.parametrize("corpus_", [PROSE_CORPUS, corpus(), corpus(fact("table_share", 0.9, (7,)))])
def test_nothing_is_passed_or_failed_without_a_ledger_line(corpus_: Corpus):
    for assessment in profiles.evaluate(corpus_):
        for cell in (*assessment.gates, *assessment.favours):
            if cell.state != "unknown":
                assert cell.lines, f"{assessment.profile.id} decided {cell.criterion.fact} with no line"


def test_a_reading_without_a_line_cannot_decide_anything():
    """An uncited number is a claim about nobody's data: wait for a citation."""
    unlined = corpus(fact("documents", 1))

    cell = scored(unlined)["searchable-archive"].gates[0]

    assert cell.state == "unknown"
    assert cell.lines == ()


# --- rule 3: reproducibility ----------------------------------------------


def test_the_same_corpus_and_version_give_the_same_matrix():
    first = profiles.evaluate(PROSE_CORPUS)
    second = profiles.evaluate(PROSE_CORPUS)

    assert first == second
    assert {assessment.catalog_version for assessment in first} == {profiles.CATALOGUE_VERSION}


def test_the_matrix_is_in_catalogue_order_so_two_datasets_line_up():
    ids = [assessment.profile.id for assessment in profiles.evaluate(PROSE_CORPUS)]

    assert ids == [profile.id for profile in profiles.CATALOGUE]


def test_a_subset_catalogue_is_honoured():
    only = profiles.evaluate(PROSE_CORPUS, catalogue=profiles.CATALOGUE[:2])

    assert [assessment.profile.id for assessment in only] == ["literature-mining", "corpus-qa"]


# --- rule 4: ordering and recommendation ----------------------------------


def test_only_a_profile_that_fits_can_be_recommended():
    recommendation = profiles.recommendation(profiles.evaluate(corpus()))

    assert recommendation is None


def test_the_recommendation_prefers_gates_then_weighted_favours():
    recommendation = profiles.recommendation(profiles.evaluate(PROSE_CORPUS))

    assert recommendation is not None
    assert recommendation.outcome == "fits"
    assert recommendation.profile.id == "literature-mining"
    assert recommendation.headline == "fits 3/3"


def test_a_partial_profile_never_outranks_a_fitting_one():
    """literature-mining fits; visual-search is only partial on the artbook."""
    mixed = corpus(
        fact("documents", 2, (1,)),
        fact("words_per_page_median", 340, (1,)),
        fact("text_layer_share", 1.0, (3,)),
        fact("table_share", 0.0, (4,)),
        fact("corpus_words", 170_000, (1,)),
        fact("plate_share", 0.5, (2,)),
        documents=(PROSE, PLATE),
    )
    assessments = scored(mixed)

    assert assessments["literature-mining"].outcome == "fits"
    assert assessments["visual-search"].outcome == "partial"
    assert profiles.recommendation(profiles.evaluate(mixed)).profile.id == "literature-mining"


# --- the catalogue itself -------------------------------------------------


def test_every_profile_is_a_use_case_with_a_consumer_and_a_tool_per_test():
    for profile in profiles.CATALOGUE:
        assert profile.gates, f"{profile.id} gates nothing"
        assert profile.summary, f"{profile.id} does not say what it is for"
        for criterion in (*profile.gates, *profile.favours, *profile.partial):
            criterion.parsed()  # raises on an unreadable test
            assert criterion.tool, f"{profile.id} asks {criterion.fact} with no instrument named"


def test_profile_ids_are_unique():
    ids = [profile.id for profile in profiles.CATALOGUE]

    assert len(ids) == len(set(ids))


def test_a_criterion_is_a_declarative_test_printable_as_a_sentence():
    assert profiles.Criterion("corpus_words", ">= 50000").holds(50_000)
    assert not profiles.Criterion("corpus_words", ">= 50000").holds(49_999)
    assert profiles.Criterion("dup_share", "< 0.5").holds(0.49)
    assert profiles.Criterion("dup_share", "!= 0.5").holds(0.0)
    assert profiles.Criterion("corpus_words", "<= 5000000").describe() == "corpus_words <= 5,000,000"
    assert profiles.Criterion("dup_share", "< 0.5").describe() == "dup_share < 0.5"


def test_an_unreadable_test_fails_loudly_rather_than_scoring_zero():
    with pytest.raises(ValueError):
        profiles.Criterion("table_share", "more or less").parsed()


# --- the page that shows the ruler ----------------------------------------


def test_the_catalogue_page_shows_every_profile_and_its_rule():
    html = profiles.render_catalogue()

    assert html.count("<!DOCTYPE html>") == 1
    for profile in profiles.CATALOGUE:
        assert f'id="profile-{profile.id}"' in html
        assert profile.title.split(" — ")[0] in html
    assert "never disqualifies" in html
    assert f"catalog v{profiles.CATALOGUE_VERSION}" in html
