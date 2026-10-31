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
    The datasets index: what is available to scout, and the frozen artifacts
    each dataset already holds.
``/report/<dataset>/<file>.html``
    A rendered artifact. Only ``report-*.html`` and ``session-*.html`` are
    servable, and only from a dataset directory directly under the workspace —
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

This serves the frozen artifacts and the tools. It never writes to the datasets
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

from data_scout import __version__, inventory, registry, workspace
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
.warn{color:#8a4b00;background:#fff6e6;padding:.5rem .7rem;border-radius:4px;font-size:.92em}
.stats{color:#3a3a3a;font-size:.92em}
.status{font-size:.82em;text-transform:uppercase;letter-spacing:.05em;color:#6b6b6b;margin:.2rem 0 0}
.status.scouted{color:#2a6b3f}
@media (prefers-color-scheme:dark){body{color:#e6e4df;background:#16171a}a{color:#7fb3e0}
code{background:#22242a}.muted{color:#9b9b96}.warn{color:#e0b070;background:#2a2416}
.stats{color:#c9c7c2}.status{color:#9b9b96}.status.scouted{color:#8fce9f}}
"""


def index_payload() -> dict[str, Any]:
    """Every dataset available to scout, and the frozen artifacts it holds.

    The inventory comes from the read-only datasets root; the artifacts and
    ledger come from the writable workspace. They are joined on the id, and a
    workspace entry with no counterpart under the root is still listed — a
    mount being down must not make a frozen report disappear.
    """
    root = workspace.workspace_root()
    datasets_root = workspace.datasets_root()
    summaries, datasets_error = inventory.discover(datasets_root)
    by_id = {summary.dataset_id: summary.as_record() for summary in summaries}

    try:
        directories = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError:
        directories = []
    for directory in directories:
        if directory.name.startswith("."):
            continue
        try:
            workspace.validate_dataset_id(directory.name)
        except workspace.WorkspaceError:
            continue
        record = by_id.get(directory.name) or inventory.WorkspaceOnlyDataset(directory.name).as_record()
        files = sorted(
            path.name
            for path in directory.iterdir()
            if path.is_file() and _SERVABLE_RE.match(path.name)
        )
        record.update(
            {
                "artifacts": [f"/report/{directory.name}/{name}" for name in files],
                "names": files,
                "findings_lines": _count_lines(directory / workspace.LEDGER_FILENAME),
                "workspace": True,
            }
        )
        by_id[directory.name] = record

    datasets: list[dict[str, Any]] = []
    for dataset_id in sorted(by_id):
        record = by_id[dataset_id]
        record.setdefault("artifacts", [])
        record.setdefault("names", [])
        record.setdefault("findings_lines", 0)
        record.setdefault("workspace", False)
        # "Scouted" means something has begun: a frame, a workspace or a
        # finding. Counting files is not scouting.
        record["scouted"] = bool(
            record["workspace"] or record["frame"] or record["artifacts"] or record["findings_lines"]
        )
        datasets.append(record)

    return {
        "version": __version__,
        "workspace": str(root),
        "datasets_root": str(datasets_root),
        "datasets_error": datasets_error,
        "datasets": datasets,
    }


def index_html(payload: dict[str, Any]) -> str:
    datasets = payload["datasets"]
    scouted = sum(1 for dataset in datasets if dataset["scouted"])
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
        "<title>data-scout datasets</title><style>",
        _INDEX_CSS,
        "</style></head><body><h1>data-scout datasets</h1>",
        f'<p class="muted">workspace <code>{_escape(payload["workspace"])}</code> · '
        f'datasets <code>{_escape(payload["datasets_root"])}</code> · v{_escape(payload["version"])}</p>',
    ]
    if payload["datasets_error"]:
        parts.append(f'<p class="warn">The datasets root is unavailable: {_escape(payload["datasets_error"])}</p>')
    else:
        parts.append(
            f'<p class="muted">{len(datasets)} dataset{"" if len(datasets) == 1 else "s"} · '
            f'{scouted} scouted · {len(datasets) - scouted} available</p>'
        )
    if not datasets:
        parts.append("<p>No datasets under the datasets root yet.</p>")
    for dataset in datasets:
        parts.extend(_dataset_html(dataset))
    parts.append("</body></html>")
    return "".join(parts)


def _dataset_html(dataset: dict[str, Any]) -> list[str]:
    """One dataset: what it is, what it costs to look, and what has been made."""
    name = dataset["name"]
    dataset_id = dataset["dataset_id"]
    parts = [f"<h2>{_escape(name)}"]
    if name != dataset_id:
        parts.append(f' <span class="muted">({_escape(dataset_id)})</span>')
    parts.append("</h2>")

    if not dataset["in_root"]:
        parts.append(
            '<p class="warn">Workspace only — this dataset is not under the datasets '
            "root right now.</p>"
        )
    else:
        facts = [
            f'{dataset["files"]:,} files',
            f'{dataset["directories"]:,} dirs',
            inventory.format_bytes(dataset["bytes"]),
        ]
        extensions = inventory.format_extensions(dataset["extensions"])
        if extensions:
            facts.append(extensions)
        if dataset["newest"]:
            facts.append(f'newest {_escape(dataset["newest"])}')
        if dataset["truncated"]:
            facts.append(f'partial — first {inventory.MAX_SCAN_ENTRIES:,} entries')
        parts.append(f'<p class="stats">{_escape(" · ".join(facts))}</p>')

    if dataset["scouted"]:
        status = "Scouted"
        if dataset["findings_lines"]:
            status += f' · {dataset["findings_lines"]} finding lines'
    else:
        status = "Available to scout"
        if dataset["in_root"] and not dataset["frame"]:
            status += " · no frame.md yet"
    status_class = "status scouted" if dataset["scouted"] else "status"
    parts.append(f'<p class="{status_class}">{_escape(status)}</p>')

    if dataset["names"]:
        parts.append("<ul>")
        for label, href in zip(dataset["names"], dataset["artifacts"], strict=False):
            parts.append(f'<li><a href="{_escape(href)}">{_escape(label)}</a></li>')
        parts.append("</ul>")
    else:
        parts.append('<p class="muted">No reports yet.</p>')
    return parts


def resolve_artifact(path: str) -> Path | None:
    """Map ``/report/<dataset>/<file>`` to a file on disk, or ``None``.

    Two independent checks, both cheap: the dataset component must be a valid
    dataset id (no separators, no dots-dots), and the filename must match the
    artifact pattern. There is no path joining of untrusted input beyond that.
    """
    parts = [part for part in path.split("/") if part]
    if len(parts) != 3 or parts[0] != "report":
        return None
    _, dataset_id, filename = parts
    try:
        workspace.validate_dataset_id(dataset_id)
    except workspace.WorkspaceError:
        return None
    if not _SERVABLE_RE.match(filename):
        return None
    candidate = workspace.workspace_dir(dataset_id) / filename
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
    dataset_id = request.path_params["dataset_id"]
    filename = request.path_params["filename"]
    found = resolve_artifact(f"/report/{dataset_id}/{filename}")
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
            Route("/report/{dataset_id}/{filename}", artifact),
        ]
    )
    app.router.redirect_slashes = False
    return app


def _plural(count: int, noun: str) -> str:
    return noun if count == 1 else f"{noun}s"


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
        f"datasets={workspace.datasets_root()})",
        flush=True,
    )
    uvicorn.run(build_app(), host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
