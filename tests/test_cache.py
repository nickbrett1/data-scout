"""The observation cache: keys on inputs, never hides a measurement, safe to delete."""

from __future__ import annotations

from pathlib import Path

from data_scout import cache


def test_a_second_call_is_served_from_cache(tmp_path):
    calls = {"n": 0}

    @cache.cached(tool="demo", tool_version="1")
    def observe(path):
        calls["n"] += 1
        return {"value": Path(path).stat().st_size}

    target = tmp_path / "input.txt"
    target.write_text("hello", encoding="utf-8")
    first = observe(str(target))
    second = observe(str(target))
    assert calls["n"] == 1
    assert first["observation_source"] == "fetched"
    assert second["observation_source"] == "cache"
    assert first["value"] == second["value"]


def test_tool_version_is_part_of_the_key(tmp_path):
    calls = {"n": 0}

    def make(version):
        @cache.cached(tool="demo", tool_version=version)
        def observe(text):
            calls["n"] += 1
            return {"value": len(text)}

        return observe

    make("1")("hello")
    make("2")("hello")
    assert calls["n"] == 2


def test_kwargs_are_part_of_the_key(tmp_path):
    calls = {"n": 0}

    @cache.cached(tool="demo", tool_version="1")
    def observe(text, per_chars=1000):
        calls["n"] += 1
        return {"value": len(text) / per_chars}

    observe("hello")
    observe("hello", per_chars=500)
    observe("hello", per_chars=500)
    assert calls["n"] == 2


def test_a_path_fingerprints_its_content_not_its_name(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("same bytes", encoding="utf-8")
    second.write_text("same bytes", encoding="utf-8")
    assert cache.fingerprint(str(first)) == cache.fingerprint(str(second))
    second.write_text("different bytes", encoding="utf-8")
    assert cache.fingerprint(str(first)) != cache.fingerprint(str(second))


def test_removing_the_cache_is_safe_and_complete(tmp_path):
    calls = {"n": 0}

    @cache.cached(tool="demo", tool_version="1")
    def observe(text):
        calls["n"] += 1
        return {"value": len(text)}

    assert observe("hello")["value"] == 5
    cache.clear_cache()
    assert observe("hello")["value"] == 5  # recomputed, same number
    assert calls["n"] == 2


def test_the_key_is_stable_across_processes():
    """A key that depends on ``hash()`` is a cache that misses every run."""
    import subprocess
    import sys

    script = (
        "from data_scout.cache import cache_key;"
        "print(cache_key(tool='demo', tool_version='1', args=('abc',), kwargs={'n': 3}))"
    )
    first = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    second = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert first.stdout == second.stdout


def test_no_finding_is_stored_only_in_the_cache(tmp_path):
    """``rm -rf cache/`` must not lose a number, only the work of computing it."""
    from data_scout.tools import text_quality

    text = "The harbour was quiet at that hour, and the boats lay still."
    before = text_quality(text, include_textdescriptives=False)
    cache.clear_cache()
    after = text_quality(text, include_textdescriptives=False)
    assert before["value"] == after["value"]
    assert before["n"] == after["n"]
