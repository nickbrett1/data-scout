"""The pure content instruments: one implementation each, measured not asserted."""

from __future__ import annotations

import json

import pytest

from data_scout.tools import (
    candidate_density,
    near_dup_docs,
    near_dup_images,
    sample,
    structure_mix,
    text_quality,
    write_sample_anchor,
)

PROSE = (
    "The harbour was quiet at that hour, and the fishing boats lay still against "
    "the wall while the tide went out beneath them. Nobody had told the harbour "
    "master that the ferry would be late, and he took the news badly, walking the "
    "length of the pier twice before he would speak to anyone at all."
)

TOC = """CONTENTS
The Harbour at Night .......... 12
The Long Pier ................ 34
A Note on the Tide ........... 61"""


def test_candidate_density_counts_prose_and_rejects_the_toc_rail():
    prose = candidate_density(PROSE)
    toc_and_prose = candidate_density(PROSE + "\n" + TOC)
    assert prose["candidate_count"] >= 1
    assert toc_and_prose["rejected"].get("toc_rail", 0) >= 1
    assert toc_and_prose["candidate_count"] == prose["candidate_count"]
    assert prose["unit_notes"] == "candidate lines per 1000 characters"
    assert prose["value"] > 0


def test_candidate_density_rejects_an_all_caps_heading():
    result = candidate_density("THE HARBOUR AND ITS PEOPLE THROUGH THE LONG CENTURY")
    assert result["candidate_count"] == 0
    assert "all_caps" in result["rejected"]


def test_candidate_density_reports_its_denominator():
    result = candidate_density(PROSE * 3)
    assert result["n"] == len(PROSE * 3)
    assert result["chars"] == result["n"]


def test_text_quality_separates_prose_from_noise():
    prose = text_quality(PROSE, include_textdescriptives=False)
    noise = text_quality("qqz xvbn mmmzzz tttt rrrr kkkk wwwq zzzx", include_textdescriptives=False)
    assert prose["dictionary_hit_rate"] > 0.7
    assert noise["dictionary_hit_rate"] < 0.3
    assert prose["language"] == "en"
    assert prose["value"] == prose["dictionary_hit_rate"]
    assert prose["entropy_bits_per_char"] > 3.0
    assert prose["mtld"] > 10


def test_text_quality_says_why_textdescriptives_is_absent_rather_than_defaulting(corpus_workspace):
    """No spaCy model here, so the optional block must explain itself."""
    result = text_quality(PROSE)
    if result.get("textdescriptives") is None:
        assert result["textdescriptives_unavailable"]
        assert "spacy" in result["textdescriptives_unavailable"].lower()


def test_structure_mix_separates_a_prose_book_from_a_plate_book():
    prose = structure_mix([{"type": "NarrativeText"}] * 18 + [{"type": "Title"}, {"type": "Picture"}])
    plates = structure_mix([{"type": "Picture"}] * 16 + [{"type": "FigureCaption"}] * 4)
    assert prose["narrative_share"] > 0.85
    assert plates["picture_share"] == 0.8
    assert plates["narrative_share"] == 0.0
    assert plates["dominant_type"] == "Picture"


def test_structure_mix_uses_the_untrimmed_denominator_when_told(roots):
    parts = [{"type": "NarrativeText"}] * 5
    result = structure_mix(parts, total_parts=100)
    assert result["n"] == 100
    assert result["narrative_share"] == 0.05
    assert result["denominator_mismatch"] is True


def test_structure_mix_accepts_a_partition_result():
    result = structure_mix({"parts": [{"type": "Table"}], "part_type_counts": {"Table": 1}})
    assert result["table_share"] == 1.0


def test_sample_is_seeded_and_recorded():
    rows = [{"id": f"row-{index}", "stratum": "a" if index < 60 else "b"} for index in range(100)]
    first = sample(rows, n=20, seed=7, stratify_by="stratum")
    second = sample(rows, n=20, seed=7, stratify_by="stratum")
    other = sample(rows, n=20, seed=8, stratify_by="stratum")
    assert first["ids"] == second["ids"]
    assert first["ids"] != other["ids"]
    assert first["seed"] == 7
    assert first["drawn"] == 20
    assert first["n"] == 100
    assert first["strata"] == {"a": 12, "b": 8}


def test_sample_refuses_to_draw_more_than_the_population():
    with pytest.raises(ValueError, match="census"):
        sample(["a", "b"], n=3)


def test_sample_writes_the_anchor_it_needs_to_be_reproducible(corpus_workspace):
    result = sample([{"id": f"r{index}"} for index in range(10)], n=3, seed=5)
    path = write_sample_anchor(result)
    anchor = json.loads(path.read_text(encoding="utf-8"))
    assert anchor["seed"] == 5
    assert anchor["rule"]
    assert anchor["ids"]
    assert anchor.pop("ids") is not None
    assert path.parent.name == "samples"


def test_near_dup_docs_finds_one_work_in_three_manifestations():
    body = PROSE * 2
    documents = [body, body, "An entirely different essay about boats and weather and tide."]
    result = near_dup_docs(documents, threshold=0.8)
    assert result["duplicate_count"] == 1
    assert result["cluster_count"] == 1
    assert result["value"] == pytest.approx(1 / 3)


def test_near_dup_images_clusters_a_resaved_image(tmp_path):
    from PIL import Image

    base = Image.new("RGB", (64, 64), (10, 80, 160))
    for index, shade in enumerate(((10, 80, 160), (12, 82, 158), (240, 240, 10))):
        image = Image.new("RGB", (64, 64), shade)
        image.paste(base.crop((0, 0, 32, 32)), (0, 0))
        image.save(tmp_path / f"image-{index}.png")
    result = near_dup_images(tmp_path)
    assert result["n"] == 3
    assert result["decoded"] == 3
    assert result["duplicate_count"] >= 1
    assert result["cluster_count"] >= 1
