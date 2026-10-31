"""The MCP façade: the belt as tools, generated from the registry, one port.

The claims worth testing are the ones a client would notice: the tool set *is*
the belt (no drift, no sixteenth), the schemas are callable, a refusal arrives as
an error rather than a zero, and a real MCP client can complete the handshake and
call a tool over the socket the container actually publishes.

Refusals are asserted **over the wire**, not in-process: a tool that raises is an
``isError`` result on the client side and an exception inside the server, so an
in-process test would be asserting the wrong contract.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import pytest

from data_scout import registry
from data_scout.mcp_server import build_server


@pytest.fixture(scope="module")
def mcp():
    return build_server()


def names(server) -> list[str]:
    return sorted(tool.name for tool in asyncio.run(server.list_tools()))


def text_of(result) -> str:
    return "\n".join(block.text for block in result.content if hasattr(block, "text"))


def test_the_facade_is_exactly_the_belt(mcp):
    assert names(mcp) == sorted(registry.names())
    assert len(names(mcp)) == registry.MAX_BELT_SIZE == 15


def test_every_tool_has_a_description_and_an_object_schema(mcp):
    for tool in asyncio.run(mcp.list_tools()):
        assert tool.description and len(tool.description) > 20, tool.name
        assert tool.input_schema["type"] == "object", tool.name


def test_measurement_register_is_callable_through_the_facade(mcp):
    result = asyncio.run(mcp.call_tool("measurement_register", {"validated_only": False}))
    payload = json.loads(text_of(result))
    assert payload["value"] == 15
    assert payload["belt_cap"] == registry.MAX_BELT_SIZE
    # The honest state of a new belt: built, nothing validated.
    assert sorted(payload["unvalidated"]) == sorted(registry.names())


# ---- a real client over the real socket ------------------------------------


@pytest.fixture
def live_mcp(dataset_workspace):
    import uvicorn

    from data_scout.__main__ import build_app

    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    httpd = uvicorn.Server(uvicorn.Config(build_app(), log_level="warning", lifespan="on"))
    thread = threading.Thread(target=httpd.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    for _ in range(500):
        if httpd.started:
            break
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        httpd.should_exit = True
        thread.join(timeout=10)
        sock.close()


def run_session(url: str, scenario):
    """One client session, one server_info, one call to ``scenario``."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async def main():
        async with streamable_http_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await scenario(session)

    return asyncio.run(main())


def test_a_client_completes_the_handshake_and_lists_the_belt(live_mcp):
    """The container's real transport: streamable HTTP on the one published port."""

    async def scenario(session):
        listed = await session.list_tools()
        return session.server_info.name, sorted(tool.name for tool in listed.tools)

    name, tools = run_session(live_mcp, scenario)
    assert name == "data-scout"
    assert tools == sorted(registry.names())


def test_over_the_wire_a_path_outside_the_roots_is_an_error_naming_both_roots(live_mcp, roots):
    async def scenario(session):
        return await session.call_tool("file_census", {"path": "/etc"})

    result = run_session(live_mcp, scenario)
    assert result.is_error
    message = text_of(result)
    assert str(roots.datasets) in message
    assert str(roots.workspace) in message


def test_over_the_wire_a_missing_extra_is_an_error_not_zero_clusters(live_mcp):
    async def scenario(session):
        return await session.call_tool("cluster_topics", {"vectors": [[0.1, 0.2]] * 40})

    result = run_session(live_mcp, scenario)
    if not result.is_error:
        pytest.skip("the [semantic] extra is installed here, so the refusal cannot be exercised")
    assert 'pip install -e ".[semantic]"' in text_of(result)


def test_over_the_wire_the_ledger_refuses_an_undecided_line(live_mcp):
    async def scenario(session):
        return await session.call_tool(
            "findings_append",
            {
                "record": {
                    "dataset_id": "demo-dataset",
                    "session": 1,
                    "unit": "documents",
                    "tool": "file_census",
                    "tool_version": "1",
                    "params": {},
                    "value": 1,
                    "n": 1,
                    "sample": {"rule": "census", "n": 1},
                    "promoted": True,
                }
            },
        )

    result = run_session(live_mcp, scenario)
    assert result.is_error
    assert "script" in text_of(result).lower()


def test_over_the_wire_a_decided_line_lands_in_the_workspace_ledger(live_mcp, dataset_workspace):
    async def scenario(session):
        return await session.call_tool(
            "findings_append",
            {
                "record": {
                    "dataset_id": "demo-dataset",
                    "session": 1,
                    "unit": "documents",
                    "tool": "file_census",
                    "tool_version": "1",
                    "params": {"mime": "text/plain"},
                    "value": 0.5,
                    "n": 2,
                    "sample": {"rule": "census", "n": 2},
                    "promoted": True,
                    "script": "data_scout.tools.file_census",
                }
            },
        )

    result = run_session(live_mcp, scenario)
    assert not result.is_error, text_of(result)
    lines = [
        line
        for line in (dataset_workspace / "findings.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    written = json.loads(lines[0])
    assert written["tool"] == "file_census"
    assert written["script"] == "data_scout.tools.file_census"
