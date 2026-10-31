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

import httpx
import pytest

from data_scout.tools.partition_doc import partition_doc

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
    state = {"polls": 0, "poll_statuses": ["pending", "started", "success"], "fail": None}

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
            return httpx.Response(200, json=DOC)
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
