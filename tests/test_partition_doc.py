"""``partition_doc`` must own its task, not abandon it at a server's ceiling.

The defect this file pins down (item 4b): the synchronous convert endpoint
abandons the request at docling-serve's ``DOCLING_SERVE_MAX_SYNC_WAIT`` while
the conversion keeps running. Every document past ~229 pages returned a clean
HTTP 504 **and** left an unowned conversion competing with the next one, so a
retry would amplify load rather than help.

These tests are hermetic: ``httpx.MockTransport`` stands in for the sidecar, so
the real request construction and poll loop run with no network. The 563-page
acceptance run is a deployment check, not something this suite can reproduce.
"""

from __future__ import annotations

import importlib

import diskcache
import httpx
import pytest

from data_scout import cache
from data_scout.tools._common import ToolUnavailable
from data_scout.tools.partition_doc import cached_parts, partition_doc

# The package re-exports the *function* under this name, so the module has to be
# fetched from sys.modules to be monkeypatched.
partition_doc_module = importlib.import_module("data_scout.tools.partition_doc")

BASE = "http://docling.test"

DOC = {
    "filename": "big-book.pdf",
    "status": "success",
    "document": {
        "filename": "big-book.pdf",
        "json_content": {
            "body": {
                "children": [
                    {"$ref": "#/texts/0"},
                    {"$ref": "#/texts/1"},
                    {"$ref": "#/texts/2"},
                ]
            },
            "texts": [
                {"self_ref": "#/texts/0", "label": "title", "text": "A Big Book", "prov": [{"page_no": 1}]},
                {"self_ref": "#/texts/1", "label": "text", "text": "Prose on the first page.", "prov": [{"page_no": 1}]},
                {"self_ref": "#/texts/2", "label": "text", "text": "Prose much later.", "prov": [{"page_no": 563}]},
            ],
            "tables": [],
            "pictures": [],
        },
    },
}


@pytest.fixture
def sidecar(monkeypatch):
    """Install a fake sidecar and record every request it sees."""
    calls: list[tuple[str, str]] = []
    state = {
        "polls": 0,
        "poll_statuses": ["pending", "started", "success"],
        "fail": None,
        "doc": DOC,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/v1/convert/file/async":
            return httpx.Response(200, json={"task_id": "task-1", "task_status": "pending", "task_position": 1})
        if request.url.path == "/v1/status/poll/task-1":
            status = state["poll_statuses"][min(state["polls"], len(state["poll_statuses"]) - 1)]
            state["polls"] += 1
            body: dict = {"task_id": "task-1", "task_status": status}
            if status == "failure":
                body["failure"] = {"message": "layout model exploded", "phase": "conversion"}
            return httpx.Response(200, json=body)
        if request.url.path == "/v1/result/task-1":
            if state["fail"]:
                return httpx.Response(500, text=state["fail"])
            return httpx.Response(200, json=state["doc"])
        return httpx.Response(404, text=f"unexpected {request.url.path}")

    def client(_httpx):
        return _httpx.Client(transport=_httpx.MockTransport(handler))

    monkeypatch.setenv("DOCLING_URL", BASE)
    monkeypatch.setenv("DATA_SCOUT_DOCLING_POLL_INTERVAL", "0")
    monkeypatch.setenv("DATA_SCOUT_DOCLING_TASK_TIMEOUT", "30")
    monkeypatch.setattr(partition_doc_module, "_client", client)
    return calls, state, None


def _pdf(tmp_path) -> str:
    path = tmp_path / "big-book.pdf"
    path.write_bytes(b"%PDF-1.4\n% not a real pdf, but real bytes\n")
    return str(path)


def _doc_with_n_parts(count: int) -> dict:
    """A sidecar response with ``count`` narrative parts, for the 200-part split."""
    texts = [
        {
            "self_ref": f"#/texts/{index}",
            "label": "text",
            "text": f"Prose part {index}.",
            "prov": [{"page_no": index // 10 + 1}],
        }
        for index in range(count)
    ]
    return {
        "filename": "many-parts.pdf",
        "status": "success",
        "document": {
            "filename": "many-parts.pdf",
            "json_content": {
                "body": {"children": [{"$ref": item["self_ref"]} for item in texts]},
                "texts": texts,
                "tables": [],
                "pictures": [],
            },
        },
    }


def test_partition_doc_submits_async_polls_and_fetches_the_result(sidecar, tmp_path):
    calls, _state, _ = sidecar

    result = partition_doc(_pdf(tmp_path))

    assert result["value"] == 3
    assert result["n"] == 3
    assert result["origin"] == "docling-json"
    assert result["docling_task_status"] == "success"
    assert [part["type"] for part in result["parts"]] == ["Title", "NarrativeText", "NarrativeText"]
    assert [part["page"] for part in result["parts"]] == [1, 1, 563]

    paths = [path for _method, path in calls]
    # The sync endpoint is not touched: its ceiling is the defect.
    assert "/v1/convert/file" not in paths
    assert paths[0] == "/v1/convert/file/async"
    assert paths.count("/v1/status/poll/task-1") == 3
    assert paths[-1] == "/v1/result/task-1"


def test_partition_doc_does_not_retry_a_failed_task(sidecar, tmp_path):
    """A retry would submit a second conversion for the same bytes."""
    calls, state, _ = sidecar
    state["poll_statuses"] = ["failure"]

    with pytest.raises(RuntimeError) as exc:
        partition_doc(_pdf(tmp_path))

    message = str(exc.value)
    assert "failure" in message
    assert "layout model exploded" in message
    assert [path for _method, path in calls].count("/v1/convert/file/async") == 1
    # A failed task is not fetched as if it had a result.
    assert "/v1/result/task-1" not in [path for _method, path in calls]


def test_partition_doc_timing_out_names_the_task_so_the_work_is_not_orphaned(sidecar, tmp_path, monkeypatch):
    """Giving up must leave a handle, not a conversion nobody can find again."""
    calls, state, _ = sidecar
    state["poll_statuses"] = ["started"]  # never terminal
    monkeypatch.setenv("DATA_SCOUT_DOCLING_TASK_TIMEOUT", "0.05")

    with pytest.raises(RuntimeError) as exc:
        partition_doc(_pdf(tmp_path))

    message = str(exc.value)
    assert "task-1" in message
    assert "/v1/result/task-1" in message
    assert "still running" in message or "not cancelled" in message
    assert [path for _method, path in calls].count("/v1/convert/file/async") == 1


def test_partition_doc_accepts_partial_success_and_says_so(sidecar, tmp_path):
    calls, state, _ = sidecar
    state["poll_statuses"] = ["partial_success"]

    result = partition_doc(_pdf(tmp_path))

    assert result["docling_task_status"] == "partial_success"
    assert result["n"] == 3


def test_a_result_fetch_failure_is_reported_with_the_task_id(sidecar, tmp_path):
    _calls, state, _ = sidecar
    state["fail"] = "result store unavailable"

    with pytest.raises(RuntimeError) as exc:
        partition_doc(_pdf(tmp_path))

    assert "task-1" in str(exc.value)
    assert "result store unavailable" in str(exc.value)


def test_the_inline_result_is_capped_while_the_cached_observation_is_complete(sidecar, tmp_path):
    """Item 4's split: 200 inline, every part on disk, one honest denominator.

    A 2,246-part document was cached truncated at 200 before this, so a reader
    over the cache read a truncated observation and could not tell. ``n`` is
    the denominator for both views.
    """
    _calls, state, _ = sidecar
    state["doc"] = _doc_with_n_parts(250)
    path = _pdf(tmp_path)

    inline = partition_doc(path)

    assert inline["n"] == 250
    assert inline["value"] == 250
    assert len(inline["parts"]) == 200
    assert inline["parts_truncated"] is True

    full = cached_parts(path)

    assert full["n"] == 250
    assert len(full["parts"]) == 250
    assert full["observation_source"] == "cache"
    # The two views are the same observation, not two measurements.
    assert full["cache_key"] == inline["cache_key"]


def test_the_reader_fails_loudly_on_a_missing_entry_and_never_converts(tmp_path, monkeypatch):
    """Absence is not an empty corpus, and the reader has no refresh path.

    ``DOCLING_URL`` is removed, so a reader that tried to convert would fail
    with a different message. The error must name the cache and the one thing
    that can create the entry.
    """
    monkeypatch.delenv("DOCLING_URL", raising=False)

    with pytest.raises(ToolUnavailable) as exc:
        cached_parts(_pdf(tmp_path))

    message = str(exc.value)
    assert "partition_doc" in message
    assert "cached observation" in message
    assert "Call partition_doc" in message
    # No side effect: a missing entry is not a reason to start converting.
    assert not cache.cache_directory().exists()


def test_the_reader_takes_its_key_from_the_writer_not_a_version_literal(
    sidecar, tmp_path, monkeypatch
):
    """The key is whatever the decorator derives, including the version.

    v1 and v2 keys differ for identical bytes, so a literal here would rot at
    the next bump. This writes an entry under a key no version string could
    produce and points the writer's key function at it: only a reader that
    delegates to the writer can find it.
    """
    path = _pdf(tmp_path)
    directory = cache.cache_directory()
    directory.mkdir(parents=True, exist_ok=True)
    observation = {
        "value": 2,
        "n": 2,
        "unit_notes": "parts",
        "part_type_counts": {"NarrativeText": 2},
        "parts": [{"type": "NarrativeText", "text": "one"}, {"type": "NarrativeText", "text": "two"}],
        "origin": "docling-json",
        "docling_task_status": "success",
        "doc_name": "many-parts.pdf",
        "docling_url": BASE,
        "model_version": "unknown",
    }
    with diskcache.Cache(str(directory)) as store:
        store["key-derived-by-the-writer"] = observation
    monkeypatch.setattr(
        partition_doc_module._observation, "cache_key", lambda *a, **k: "key-derived-by-the-writer"
    )

    full = cached_parts(path)

    assert full["n"] == 2
    assert len(full["parts"]) == 2
    assert full["cache_key"] == "key-derived-by-the-writer"
