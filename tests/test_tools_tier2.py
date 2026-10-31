"""Tier 2 must fail loudly — never return zero, empty, or a plausible default.

"A measurement that silently reports 'no clusters' when BERTopic is not
installed is worse than one that refuses, because the report will believe it."
"""

from __future__ import annotations

import importlib.util

import pytest

from data_scout.tools import (
    cluster_topics,
    embed_sample,
    image_inventory,
    partition_doc,
)
from data_scout.tools._common import ToolUnavailable


def installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def test_partition_doc_without_a_docling_url_says_which_variable(roots, dataset):
    with pytest.raises(ToolUnavailable) as exc:
        partition_doc(dataset / "one.txt")
    assert "DOCLING_URL" in str(exc.value)
    assert "sidecar" in str(exc.value) or "does not ship a model" in str(exc.value)


def test_partition_doc_without_the_extra_semantic_never_returns_a_zero(roots, dataset):
    """No URL, no httpx call, no measurement: it refuses."""
    with pytest.raises(ToolUnavailable):
        partition_doc(dataset / "one.txt")


def test_embed_sample_without_a_gateway_names_both_variables():
    with pytest.raises(ToolUnavailable) as exc:
        embed_sample(["some text"])
    message = str(exc.value)
    assert "LITELLM_BASE_URL" in message
    assert "LITELLM_API_KEY" in message or "embedding gateway" in message
    assert "budgeted" in message or "spend" in message


def test_embed_sample_without_a_model_id_refuses_to_guess_the_meaning(monkeypatch):
    if not installed("numpy"):
        pytest.skip("numpy is part of the [semantic] extra; the loud-failure path is covered above")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://gateway.invalid:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    with pytest.raises(ToolUnavailable) as exc:
        embed_sample(["some text"])
    assert "model" in str(exc.value).lower()


@pytest.mark.skipif(installed("bertopic"), reason="bertopic present: the absent-extra path cannot be exercised")
def test_cluster_topics_without_the_semantic_extra_names_the_install_command():
    with pytest.raises(ToolUnavailable) as exc:
        cluster_topics([[0.1, 0.2]] * 40)
    message = str(exc.value)
    assert "semantic" in message
    assert 'pip install -e ".[semantic]"' in message


def test_image_inventory_without_a_vision_model_refuses(roots, dataset, monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://gateway.invalid:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    with pytest.raises(ToolUnavailable) as exc:
        image_inventory(dataset / "one.txt")
    assert "vision embedding model" in str(exc.value)


def test_image_inventory_without_a_gateway_refuses(roots, dataset):
    with pytest.raises(ToolUnavailable) as exc:
        image_inventory(dataset / "one.txt")
    assert "gateway" in str(exc.value)


def test_a_tool_unavailable_error_names_the_tool_the_missing_thing_and_a_hint():
    error = ToolUnavailable("some_tool", "the `thing` binary", "apt-get install thing")
    message = str(error)
    assert "some_tool" in message
    assert "thing" in message
    assert "apt-get install" in message
