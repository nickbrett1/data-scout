"""Item 5: the sidecar's ceilings are recorded, and the unknown stays unknown.

The defect this pins down is the shape that let 4b hide: ``timeout 3600`` in the
job, a 600 s httpx read timeout in the belt, and a 120 s ``MAX_SYNC_WAIT`` on the
sidecar — three owners, and the smallest one silently governed. The fix is not to
guess the sidecar's value: it is to record what the sidecar actually says, name
which ceiling binds now that the client is asynchronous, and never let "nobody
told us" read as a number.

The primary source matters here and is asserted rather than assumed: docling-serve
reports package versions at ``/version`` and a request schema at
``/openapi.json``, and **no** endpoint reports ``max_sync_wait``. So the honest
record is ``None`` with a reason — 120 is the pydantic default, not an
observation, and "never configured" is not distinguishable from "configured to
120" over the API. These tests are hermetic: ``httpx.MockTransport`` stands in for
the sidecar.
"""

from __future__ import annotations

import httpx
import pytest

from data_scout import docling_service

BASE = "http://docling.test"

VERSIONS = {
    "docling": "2.60.0",
    "docling-core": "2.45.0",
    "docling-serve": "1.9.0",
}

OPENAPI = {
    "openapi": "3.1.0",
    "paths": {
        "/v1/convert/file": {"post": {}},
        "/v1/convert/file/async": {"post": {}},
        "/v1/status/poll/{task_id}": {"get": {}},
        "/v1/result/{task_id}": {"get": {}},
    },
}


def _client(*, version=200, openapi=200, payloads=None, fail=None) -> tuple[httpx.Client, list]:
    """A fake sidecar. ``fail`` raises for every request, like a dead host."""
    seen: list[tuple[str, str]] = []
    bodies = payloads or {"/version": VERSIONS, "/openapi.json": OPENAPI}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if fail is not None:
            raise fail
        if request.url.path == "/version":
            if version != 200:
                return httpx.Response(version, json={"detail": "refused"})
            return httpx.Response(200, json=bodies["/version"])
        if request.url.path == "/openapi.json":
            if openapi != 200:
                return httpx.Response(openapi, text="not found")
            return httpx.Response(200, json=bodies["/openapi.json"])
        return httpx.Response(404, text=f"unexpected {request.url.path}")

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


@pytest.fixture(autouse=True)
def _no_declared_wait(monkeypatch):
    monkeypatch.delenv(docling_service.DECLARED_MAX_SYNC_WAIT_ENV, raising=False)
    monkeypatch.delenv("DATA_SCOUT_DOCLING_TASK_TIMEOUT", raising=False)


def test_a_reachable_sidecar_records_versions_and_leaves_the_sync_ceiling_unknown():
    client, seen = _client()

    record = docling_service.observe(client=client, url=BASE)

    assert record["state"] == "reachable"
    assert record["version"]["state"] == "reported"
    assert record["version"]["values"]["docling-serve"] == "1.9.0"
    assert record["openapi"]["state"] == "reported"
    assert record["openapi"]["async_convert"] is True

    # The point of the item: reachable does not mean the ceiling is knowable.
    ceiling = record["max_sync_wait"]
    assert ceiling["seconds"] is None
    assert ceiling["source"] == "unobservable"
    assert "no endpoint" in ceiling["detail"]
    # And specifically not the pydantic default masquerading as an observation.
    assert ceiling["seconds"] != 120
    assert record["timeouts"]["sidecar_max_sync_wait"]["seconds"] is None

    # A probe reads; it never converts.
    assert seen == [("GET", "/version"), ("GET", "/openapi.json")]
    assert not any(path.startswith("/v1/convert") for _method, path in seen)


def test_the_binding_timeout_is_the_belts_own_task_deadline():
    client, _seen = _client()

    record = docling_service.observe(client=client, url=BASE)

    binding = record["timeouts"]["binding"]
    assert binding["seconds"] == 3600.0
    assert binding["env"] == "DATA_SCOUT_DOCLING_TASK_TIMEOUT"
    assert binding["owner"] == "data_scout.tools.partition_doc"
    # The read timeout is per request and must not be reported as the deadline.
    assert record["timeouts"]["request_read"]["seconds"] == 600.0
    assert record["timeouts"]["request_read"]["seconds"] != binding["seconds"]


def test_a_configured_task_timeout_is_what_the_record_says_governs(monkeypatch):
    monkeypatch.setenv("DATA_SCOUT_DOCLING_TASK_TIMEOUT", "90")
    client, _seen = _client()

    record = docling_service.observe(client=client, url=BASE)

    assert record["timeouts"]["binding"]["seconds"] == 90.0


def test_a_version_refusal_is_recorded_as_a_configured_no_not_as_absence():
    """403 is ``show_version_info=false``: the server exists and says nothing."""
    client, _seen = _client(version=403)

    record = docling_service.observe(client=client, url=BASE)

    assert record["state"] == "reachable"
    assert record["version"]["state"] == "refused"
    assert record["version"]["http_status"] == 403
    assert record["version"]["values"] is None
    assert "show_version_info" in record["version"]["detail"]
    # The ceiling is still unknown, and still not a zero.
    assert record["max_sync_wait"]["seconds"] is None


def test_a_missing_openapi_schema_is_missing_rather_than_empty():
    """``enable_api_docs=false`` gives 404: an unavailable schema, not an empty one."""
    client, _seen = _client(openapi=404)

    record = docling_service.observe(client=client, url=BASE)

    assert record["openapi"]["state"] == "missing"
    assert record["openapi"]["http_status"] == 404
    assert record["openapi"]["paths"] is None
    assert record["openapi"]["async_convert"] is None
    assert "enable_api_docs" in record["openapi"]["detail"]


def test_an_unreachable_sidecar_is_not_recorded_as_a_ceiling_of_zero():
    client, _seen = _client(fail=httpx.ConnectError("connection refused"))

    record = docling_service.observe(client=client, url=BASE)

    assert record["state"] == "unreachable"
    assert record["version"]["state"] == "not_checked"
    assert record["openapi"]["state"] == "not_checked"
    assert "ConnectError" in record["version"]["detail"]
    assert record["max_sync_wait"]["seconds"] is None
    # An unreachable sidecar must not take the server down with it.
    assert record["timeouts"]["binding"]["seconds"] == 3600.0


def test_not_configured_is_distinct_from_unreachable_and_from_a_number(monkeypatch):
    monkeypatch.delenv("DOCLING_URL", raising=False)

    record = docling_service.observe()

    assert record["state"] == "not_configured"
    assert record["url"] is None
    assert record["version"]["state"] == "not_checked"
    assert "DOCLING_URL is not set" in record["version"]["detail"]
    assert record["max_sync_wait"]["seconds"] is None
    # The binding ceiling is known even when the sidecar is not: it is ours.
    assert record["timeouts"]["binding"]["seconds"] == 3600.0


def test_a_declared_value_is_labelled_declared_and_never_observed(monkeypatch):
    monkeypatch.setenv(docling_service.DECLARED_MAX_SYNC_WAIT_ENV, "540")
    client, _seen = _client()

    record = docling_service.observe(client=client, url=BASE)

    assert record["max_sync_wait"]["seconds"] == 540.0
    assert record["max_sync_wait"]["source"] == "declared"
    assert "not an observation" in record["max_sync_wait"]["detail"]


def test_a_declaration_that_is_not_a_number_leaves_the_ceiling_unknown(monkeypatch):
    monkeypatch.setenv(docling_service.DECLARED_MAX_SYNC_WAIT_ENV, "not-a-number")
    client, _seen = _client()

    record = docling_service.observe(client=client, url=BASE)

    assert record["max_sync_wait"]["seconds"] is None
    assert record["max_sync_wait"]["source"] == "unobservable"
    assert "was ignored" in record["max_sync_wait"]["detail"]


def test_the_summary_line_names_the_binding_timeout_and_the_unknown_ceiling():
    client, _seen = _client()

    line = docling_service.summary(docling_service.observe(client=client, url=BASE))

    assert "docling: reachable" in line
    assert "docling-serve 1.9.0" in line
    assert "sidecar sync ceiling unknown" in line
    assert "binding timeout 3600s" in line
