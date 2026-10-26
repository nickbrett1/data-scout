"""The registry is the source of truth, so the belt's shape is a test."""

from __future__ import annotations

import importlib

import pytest

from data_scout import registry
from data_scout.tools import measurement_register

EXPECTED = {
    "file_census",
    "pdf_forensics",
    "exif_census",
    "partition_doc",
    "structure_mix",
    "candidate_density",
    "text_quality",
    "embed_sample",
    "cluster_topics",
    "image_inventory",
    "near_dup_docs",
    "near_dup_images",
    "sample",
    "findings_append",
    "measurement_register",
}


def test_belt_is_the_fifteen_in_the_design():
    assert set(registry.names()) == EXPECTED
    assert len(registry.BELT) == registry.MAX_BELT_SIZE == 15


def test_registry_has_no_problems():
    assert registry.problems() == []


def test_every_tool_module_resolves_and_exports_the_function():
    """``script`` is a module path; it must resolve as a module *and* a function."""
    for spec in registry.all_tools():
        module = importlib.import_module(spec.module)
        assert callable(getattr(module, spec.name)), spec.name
        package = importlib.import_module("data_scout.tools")
        assert callable(getattr(package, spec.name)), spec.name


def test_nothing_is_validated_yet_and_that_is_the_honest_state():
    """Built, nothing validated. The pruning rule depends on knowing this."""
    for spec in registry.all_tools():
        assert spec.validated_on == (), spec.name


def test_tier_zero_is_free_by_construction():
    for spec in registry.all_tools():
        if spec.tier == 0:
            assert spec.purity in {"pure", "read-only-sql"}, spec.name


def test_unknown_tool_names_the_belt_rather_than_inventing_one():
    with pytest.raises(KeyError) as exc:
        registry.get("chart_or_art")
    assert "sixteenth" in str(exc.value)


def test_measurement_register_reports_the_whole_belt_and_the_unvalidated_set():
    result = measurement_register()
    assert result["value"] == 15
    assert result["n"] == 15
    assert len(result["unvalidated"]) == 15
    assert result["registry_problems"] == []


def test_measurement_register_can_be_asked_for_one_tool():
    result = measurement_register("pdf_forensics")
    assert result["value"] == 1
    assert result["entries"][0]["tier"] == 0
    assert result["entries"][0]["resolvable"] is True
