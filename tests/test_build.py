"""The build stamp: the deployment's env first, a checkout's git second, never raise.

The order matters. Inside the image there is no ``.git`` and no git binary, so
the env var baked from the build arg is the only possible answer; in a source
checkout there is no build arg, so git answers. Neither present is ``unknown``,
not an exception: the datasets index is a browse surface.
"""

from __future__ import annotations

from data_scout import build


def test_the_environment_wins(monkeypatch):
    monkeypatch.setenv(build.BUILD_ENV, "abc123")
    monkeypatch.setattr(build, "_git_commit", lambda: "deadbeef")
    assert build.build_id() == "abc123"


def test_a_blank_stamp_is_not_an_answer(monkeypatch):
    monkeypatch.setenv(build.BUILD_ENV, "   ")
    monkeypatch.setattr(build, "_git_commit", lambda: "deadbeef")
    assert build.build_id() == "deadbeef"


def test_no_stamp_and_no_checkout_degrades_to_unknown(monkeypatch):
    monkeypatch.delenv(build.BUILD_ENV, raising=False)
    monkeypatch.setattr(build, "_git_commit", lambda: None)
    assert build.build_id() == build.UNKNOWN


def test_a_missing_git_binary_is_unknown_not_an_exception(monkeypatch):
    """The runtime stage is python:3.12-slim: git is not installed there."""
    monkeypatch.delenv(build.BUILD_ENV, raising=False)
    build._git_commit.cache_clear()

    def boom(*_args, **_kwargs):
        raise OSError("git: not found")

    monkeypatch.setattr(build.subprocess, "run", boom)
    try:
        assert build.build_id() == build.UNKNOWN
    finally:
        build._git_commit.cache_clear()


def test_the_short_form_is_a_label_not_a_wall():
    assert build.short("a" * 40) == "a" * 12
    assert build.short("abc123") == "abc123"
