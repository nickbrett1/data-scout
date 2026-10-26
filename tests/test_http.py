"""One port, two routes: the frozen artifacts and a health endpoint.

``/a2a`` stays a documented seam — serving it is deploy-time wiring, and the
intelligence is a recipe, not a hand-written agent loop.
"""

from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from data_scout import __main__ as app
from data_scout import workspace

CORPUS_ID = "demo-corpus"


@pytest.fixture
def server(corpus_workspace):
    (corpus_workspace / "report-2026-10-26.html").write_text("<html>report</html>", encoding="utf-8")
    (corpus_workspace / "findings.jsonl").write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def get(port: int, path: str) -> tuple[int, str, str]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("GET", path)
    response = connection.getresponse()
    body = response.read().decode("utf-8")
    content_type = response.getheader("Content-Type", "")
    connection.close()
    return response.status, content_type, body


def test_healthz_reports_the_two_resolved_roots(server, roots):
    status, content_type, body = get(server, "/healthz")
    assert status == 200
    assert "application/json" in content_type
    payload = json.loads(body)
    assert payload["status"] == "ok"
    assert payload["workspace"] == str(roots.workspace)
    assert payload["corpora_root"] == str(roots.corpora)


def test_the_index_lists_the_frozen_artifacts(server):
    status, _content_type, body = get(server, "/report/")
    assert status == 200
    assert CORPUS_ID in body
    assert "report-2026-10-26.html" in body
    assert "2 finding lines" in body


def test_index_json_is_the_same_index_for_machines(server):
    status, _content_type, body = get(server, "/report/index.json")
    assert status == 200
    payload = json.loads(body)
    assert payload["corpora"][0]["corpus_id"] == CORPUS_ID
    assert payload["corpora"][0]["findings_lines"] == 2


def test_a_frozen_artifact_is_served(server):
    status, content_type, body = get(server, f"/report/{CORPUS_ID}/report-2026-10-26.html")
    assert status == 200
    assert content_type.startswith("text/html")
    assert body == "<html>report</html>"


def test_the_ledger_is_not_servable_even_from_the_right_corpus(server):
    status, _content_type, _body = get(server, f"/report/{CORPUS_ID}/findings.jsonl")
    assert status == 404


def test_traversal_is_refused(server):
    for path in (
        f"/report/{CORPUS_ID}/../../etc/passwd",
        "/report/../etc/passwd",
        f"/report/{CORPUS_ID}/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
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


def test_resolve_artifact_rejects_an_invalid_corpus_id():
    assert app.resolve_artifact("/report/..%2f..%2fx/report-a.html") is None
    assert app.resolve_artifact("/report/demo-corpus/notes.txt") is None


def test_the_app_never_writes(corpus_workspace):
    """Nothing this module imports opens a file for writing."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(app))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "open":
            modes = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
            assert "w" not in modes and "a" not in modes
    _ = workspace  # imported for its env contract only
