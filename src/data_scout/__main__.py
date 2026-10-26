"""The HTTP app: the MCP façade, the frozen reports, and a documented ``/a2a`` seam.

One container, one published port, path-routed — because a second published port
would be a second service (``data-scout-genproj-scaffold`` §6.2):

``/mcp``
    The belt as an MCP tool server (``data_scout.mcp_server``), so an agent — or
    Open WebUI through MCPHub — calls the same functions the notebooks call.
    Same port, same process: the façade is generated from the registry, not
    deployed beside it.
``/healthz``
    Liveness plus the two resolved roots and the belt size, so a misconfigured
    mount is visible without opening a shell.
``/`` and ``/report/``
    The reports index: every corpus workspace and its frozen artifacts.
``/report/<corpus>/<file>.html``
    A rendered artifact. Only ``report-*.html`` and ``session-*.html`` are
    servable, and only from a corpus directory directly under the workspace —
    static file serving, no templates, no directory listing of the workspace.
``/report/index.json``
    The same index as JSON, for anything that is not a phone.
``/a2a``
    **Deliberately unimplemented.** The A2A listener is deploy-time wiring and
    the dial address is a fact about an observer, not about this service
    (``data-scout-convention`` §7). Do not write an agent loop here: the
    intelligence is a goose recipe driving these tools (design §2.1). The belt is
    reachable as tools, which is what a recipe needs and what an A2A card would
    only front.

This serves the frozen artifacts and the tools. It never writes to the corpora
root, and the only thing that ever writes at all is ``findings_append``, into the
workspace it already owns.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from data_scout import __version__, registry, workspace
from data_scout.mcp_server import MCP_PATH, build_server

HOST_ENV = "DATA_SCOUT_HTTP_HOST"
PORT_ENV = "DATA_SCOUT_HTTP_PORT"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8710

_SERVABLE_RE = re.compile(r"^(report|session)-[A-Za-z0-9._-]*\.html$")

# Named routes rather than a static directory: three known files, no path input.
_ICONS = {
    "/apple-touch-icon.png": "apple-touch-icon.png",
    "/apple-touch-icon-precomposed.png": "apple-touch-icon.png",
    "/icon-512.png": "icon-512.png",
    "/favicon.ico": "favicon-32.png",
    "/favicon-32.png": "favicon-32.png",
}


def asset_path(filename: str) -> Path:
    """The icon files, from the installed package rather than from a working tree."""
    from importlib.resources import files

    resource = files("data_scout") / "assets" / filename
    return Path(str(resource))

_A2A_DETAIL = (
    "The A2A listener is deploy-time wiring, not repo work. The correct card URL is "
    "the container's name on the shared network (an address is a fact about an "
    "observer, not about a service). The intelligence is a goose recipe driving these "
    f"tools, and the tools are served as MCP at {MCP_PATH} — that is the surface to "
    f"attach to. A card pointing here would list green and fail every turn."
)

_INDEX_CSS = """
body{margin:0 auto;padding:1.6rem 1.1rem;max-width:38em;color:#1b1b1b;background:#fdfdfc;
font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
h1{font-size:1.4rem}h2{font-size:1.1rem;margin-top:1.8rem}
li{margin:.25rem 0}a{color:#2a5d8f}code{background:#f2f2ee;padding:.1em .3em;border-radius:3px}
.muted{color:#6b6b6b;font-size:.9em}
@media (prefers-color-scheme:dark){body{color:#e6e4df;background:#16171a}a{color:#7fb3e0}
code{background:#22242a}.muted{color:#9b9b96}}
"""


def index_payload() -> dict[str, Any]:
    """Every corpus workspace and the frozen artifacts it holds."""
    root = workspace.workspace_root()
    corpora: list[dict[str, Any]] = []
    try:
        directories = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError:
        directories = []
    for directory in directories:
        if directory.name.startswith("."):
            continue
        try:
            workspace.validate_corpus_id(directory.name)
        except workspace.WorkspaceError:
            continue
        files = sorted(
            path.name
            for path in directory.iterdir()
            if path.is_file() and _SERVABLE_RE.match(path.name)
        )
        ledger = directory / workspace.LEDGER_FILENAME
        corpora.append(
            {
                "corpus_id": directory.name,
                "artifacts": [f"/report/{directory.name}/{name}" for name in files],
                "names": files,
                "findings_lines": _count_lines(ledger),
            }
        )
    return {"version": __version__, "workspace": str(root), "corpora_root": str(workspace.corpora_root()), "corpora": corpora}


def index_html(payload: dict[str, Any]) -> str:
    parts = [
        "<!DOCTYPE html><html lang=en><head><meta charset=utf-8>",
        '<meta name=viewport content="width=device-width, initial-scale=1">',
        # The home-screen contract: iOS reads apple-touch-icon for the bookmark
        # tile and apple-mobile-web-app-title for its caption, so the caption is
        # "data-scout" and the page title stays descriptive.
        '<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">',
        '<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png">',
        '<link rel="icon" type="image/png" sizes="512x512" href="/icon-512.png">',
        '<meta name="theme-color" content="#1b1d22">',
        '<meta name="apple-mobile-web-app-title" content="data-scout">',
        '<meta name="apple-mobile-web-app-capable" content="yes">',
        "<title>data-scout reports</title><style>",
        _INDEX_CSS,
        "</style></head><body><h1>data-scout reports</h1>",
        f'<p class="muted">workspace <code>{_escape(payload["workspace"])}</code> · '
        f'corpora <code>{_escape(payload["corpora_root"])}</code> · v{_escape(payload["version"])}</p>',
    ]
    if not payload["corpora"]:
        parts.append("<p>No corpus workspaces yet.</p>")
    for corpus in payload["corpora"]:
        parts.append(f'<h2>{_escape(corpus["corpus_id"])}</h2>')
        parts.append(f'<p class="muted">{corpus["findings_lines"]} finding lines</p>')
        if corpus["names"]:
            parts.append("<ul>")
            for name, href in zip(corpus["names"], corpus["artifacts"], strict=False):
                parts.append(f'<li><a href="{_escape(href)}">{_escape(name)}</a></li>')
            parts.append("</ul>")
        else:
            parts.append("<p class=muted>No rendered artifacts yet.</p>")
    parts.append("</body></html>")
    return "".join(parts)


def resolve_artifact(path: str) -> Path | None:
    """Map ``/report/<corpus>/<file>`` to a file on disk, or ``None``.

    Two independent checks, both cheap: the corpus component must be a valid
    corpus id (no separators, no dots-dots), and the filename must match the
    artifact pattern. There is no path joining of untrusted input beyond that.
    """
    parts = [part for part in path.split("/") if part]
    if len(parts) != 3 or parts[0] != "report":
        return None
    _, corpus_id, filename = parts
    try:
        workspace.validate_corpus_id(corpus_id)
    except workspace.WorkspaceError:
        return None
    if not _SERVABLE_RE.match(filename):
        return None
    candidate = workspace.workspace_dir(corpus_id) / filename
    return candidate if candidate.is_file() else None


def _json(payload: dict[str, Any], status: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status, headers={"Cache-Control": "no-store"})


def healthz(request: Request) -> Response:
    del request
    return _json(
        {
            "status": "ok",
            "version": __version__,
            "mcp": MCP_PATH,
            "belt": {"size": len(registry.BELT), "cap": registry.MAX_BELT_SIZE},
            **index_payload(),
        }
    )


def index_json(request: Request) -> Response:
    del request
    return _json(index_payload())


def index_page(request: Request) -> Response:
    del request
    return HTMLResponse(index_html(index_payload()), headers={"Cache-Control": "no-store"})


def a2a(request: Request) -> Response:
    del request
    return _json({"error": "not_implemented", "detail": _A2A_DETAIL}, status=501)


def artifact(request: Request) -> Response:
    corpus_id = request.path_params["corpus_id"]
    filename = request.path_params["filename"]
    found = resolve_artifact(f"/report/{corpus_id}/{filename}")
    if found is None:
        return _json({"error": "not_found", "path": request.url.path}, status=404)
    return FileResponse(found, media_type="text/html; charset=utf-8")


def icon(request: Request) -> Response:
    filename = _ICONS.get(request.url.path)
    if filename is None:
        return _json({"error": "not_found", "path": request.url.path}, status=404)
    return FileResponse(asset_path(filename), media_type="image/png")


def build_app() -> Starlette:
    """The one app: MCP at ``/mcp``, reports under ``/report/``, on one port.

    The MCP routes are mounted first, so ``/mcp`` is the server's own handler and
    never falls through to the report routes. DNS-rebinding protection is off
    because this is dialed by container name on an internal network
    (``http://data-scout:8710/mcp``); the transport is not exposed publicly.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    server = build_server()
    app = server.streamable_http_app(
        streamable_http_path=MCP_PATH,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    app.router.routes.extend(
        [
            Route("/healthz", healthz),
            Route("/report/index.json", index_json),
            Route("/index.json", index_json),
            Route("/", index_page),
            Route("/report", index_page),
            Route("/report/", index_page),
            Route("/a2a", a2a),
            *[Route(path, icon) for path in _ICONS],
            Route("/report/{corpus_id}/{filename}", artifact),
        ]
    )
    app.router.redirect_slashes = False
    return app


def _count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m data_scout``: serve, and take no arguments.

    The container's command is this module, so the process that runs in the
    image is the same one both routes belong to — one container, one port.
    """
    del argv  # accepted so a future flag does not change the contract
    return serve()


def serve(host: str | None = None, port: int | None = None) -> int:
    import uvicorn

    host = host or os.environ.get(HOST_ENV, DEFAULT_HOST)
    port = port or int(os.environ.get(PORT_ENV, DEFAULT_PORT))
    print(
        f"data-scout {__version__} serving on http://{host}:{port} "
        f"(mcp={MCP_PATH}, belt={len(registry.BELT)}, workspace={workspace.workspace_root()}, "
        f"corpora={workspace.corpora_root()})",
        flush=True,
    )
    uvicorn.run(build_app(), host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
