"""One port, four surfaces: the MCP façade, the reports, health, and the /a2a seam.

Written against a real socket rather than a test client on purpose: the encoded
traversal case (``%2e%2e%2f``) is refused *by the router*, and a client that
normalises paths before sending would make that test assert nothing.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time

import pytest

from data_scout import __main__ as app
from data_scout import registry, workspace

DATASET_ID = "demo-dataset"


@pytest.fixture
def server(dataset_workspace):
    import uvicorn

    (dataset_workspace / "report-2026-10-26.html").write_text("<html>report</html>", encoding="utf-8")
    (dataset_workspace / "findings.jsonl").write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")

    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    config = uvicorn.Config(app.build_app(), log_level="warning", lifespan="on")
    httpd = uvicorn.Server(config)
    thread = threading.Thread(target=httpd.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    for _ in range(500):
        if httpd.started:
            break
        time.sleep(0.02)
    try:
        yield port
    finally:
        httpd.should_exit = True
        thread.join(timeout=10)
        sock.close()


def request(port: int, path: str, method: str = "GET", body: str | None = None, headers: dict | None = None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    content_type = response.getheader("Content-Type", "")
    status = response.status
    connection.close()
    return status, content_type, payload


def get(port: int, path: str) -> tuple[int, str, str]:
    """Text-typed helper: most routes here are HTML or JSON."""
    status, content_type, payload = request(port, path)
    return status, content_type, payload.decode("utf-8")


def test_healthz_reports_the_two_resolved_roots_and_the_belt(server, roots):
    status, content_type, body = get(server, "/healthz")
    assert status == 200
    assert "application/json" in content_type
    payload = json.loads(body)
    assert payload["status"] == "ok"
    assert payload["workspace"] == str(roots.workspace)
    assert payload["datasets_root"] == str(roots.datasets)
    assert payload["mcp"] == "/mcp"
    assert payload["belt"] == {"size": 15, "cap": registry.MAX_BELT_SIZE}


def test_the_index_lists_the_frozen_artifacts(server):
    status, _content_type, body = get(server, "/report/")
    assert status == 200
    assert DATASET_ID in body
    assert "report-2026-10-26.html" in body
    assert "2 finding lines" in body


def test_index_json_is_the_same_index_for_machines(server):
    status, _content_type, body = get(server, "/report/index.json")
    assert status == 200
    payload = json.loads(body)
    assert payload["datasets"][0]["dataset_id"] == DATASET_ID
    assert payload["datasets"][0]["findings_lines"] == 2
    assert payload["datasets"][0]["scouted"] is True


def test_the_index_lists_datasets_that_are_available_to_scout(server, roots):
    waiting = roots.datasets / "waiting-dataset"
    (waiting / "docs").mkdir(parents=True)
    (waiting / "a.txt").write_text("hello", encoding="utf-8")
    (waiting / "b.pdf").write_bytes(b"%PDF-1.4")

    status, _content_type, body = get(server, "/report/")
    assert status == 200
    assert "waiting-dataset" in body
    assert "Available to scout" in body
    assert "no frame.md yet" in body
    assert "2 files" in body
    assert "1 dir" in body
    assert "txt 1, pdf 1" in body
    assert "No reports yet." in body

    _status, _content_type, body = get(server, "/report/index.json")
    payload = json.loads(body)
    waiting_record = next(item for item in payload["datasets"] if item["dataset_id"] == "waiting-dataset")
    assert waiting_record["files"] == 2
    assert waiting_record["directories"] == 1
    assert waiting_record["bytes"] == 13
    assert waiting_record["extensions"] == {".txt": 1, ".pdf": 1}
    assert waiting_record["frame"] is False
    assert waiting_record["in_root"] is True
    assert waiting_record["scouted"] is False


def test_a_workspace_with_no_dataset_under_the_root_is_still_listed(server):
    _status, _content_type, body = get(server, "/")
    assert DATASET_ID in body
    assert "Workspace only" in body
    assert "report-2026-10-26.html" in body


def test_an_unavailable_datasets_root_is_stated_not_hidden(server, roots, monkeypatch):
    monkeypatch.setenv(workspace.DATASETS_ROOT_ENV, str(roots.tmp / "gone"))

    status, _content_type, body = get(server, "/report/")
    assert status == 200
    assert "datasets root is unavailable" in body

    _status, _content_type, body = get(server, "/report/index.json")
    payload = json.loads(body)
    assert payload["datasets_error"] is not None
    # The workspace's frozen artifacts survive a mount being down.
    assert any(item["dataset_id"] == DATASET_ID for item in payload["datasets"])


def test_a_frozen_artifact_is_served(server):
    status, content_type, body = get(server, f"/report/{DATASET_ID}/report-2026-10-26.html")
    assert status == 200
    assert content_type.startswith("text/html")
    assert body == "<html>report</html>"


def test_a_head_request_has_no_body(server):
    status, content_type, body = request(server, "/healthz", method="HEAD")
    assert status == 200
    assert "application/json" in content_type
    assert body == b""


def test_the_ledger_is_not_servable_even_from_the_right_dataset(server):
    status, _content_type, _body = get(server, f"/report/{DATASET_ID}/findings.jsonl")
    assert status == 404


def test_traversal_is_refused(server):
    for path in (
        f"/report/{DATASET_ID}/../../etc/passwd",
        "/report/../etc/passwd",
        f"/report/{DATASET_ID}/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    ):
        status, _content_type, _body = get(server, path)
        assert status == 404, path


def test_unknown_routes_are_404(server):
    assert get(server, "/nope")[0] == 404


def test_the_a2a_route_is_a_documented_seam_not_an_agent_loop(server):
    status, _content_type, body = get(server, "/a2a")
    assert status == 501
    payload = json.loads(body)
    assert payload["error"] == "not_implemented"
    assert "recipe" in payload["detail"]
    # And it names the surface that *is* implemented, so the seam is actionable.
    assert "/mcp" in payload["detail"]


def test_resolve_artifact_rejects_an_invalid_dataset_id():
    assert app.resolve_artifact("/report/..%2f..%2fx/report-a.html") is None
    assert app.resolve_artifact("/report/demo-dataset/notes.txt") is None


def test_the_app_never_writes(dataset_workspace):
    """Nothing this module imports opens a file for writing."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(app))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "open":
            modes = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
            assert "w" not in modes and "a" not in modes
    _ = workspace  # imported for its env contract only


def test_the_home_screen_icon_is_served_as_a_png(server):
    """iOS fetches this path for an Add-to-Home-Screen tile, and only accepts an image."""
    for path in ("/apple-touch-icon.png", "/apple-touch-icon-precomposed.png", "/icon-512.png"):
        status, content_type, body = request(server, path)
        assert status == 200, path
        assert content_type == "image/png", path
        assert body[:8] == b"\x89PNG\r\n\x1a\n", path


def test_the_index_declares_the_icon_and_a_short_caption(server):
    status, _content_type, body = get(server, "/")
    assert status == 200
    assert '<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">' in body
    assert 'name="apple-mobile-web-app-title" content="data-scout"' in body
    assert 'name="theme-color" content="#1b1d22"' in body


def test_an_unknown_asset_path_is_404(server):
    assert get(server, "/icon-999.png")[0] == 404
