"""The HTTP app: the MCP façade, the frozen reports, and a documented ``/a2a`` seam.

One container, one published port, path-routed — because a second published port
would be a second service (``data-scout-genproj-scaffold`` §6.2):

``/mcp``
    The belt as an MCP tool server (``data_scout.mcp_server``), so an agent — or
    Open WebUI through MCPHub — calls the same functions the notebooks call.
    Same port, same process: the façade is generated from the registry, not
    deployed beside it.
``/healthz``
    Liveness plus the two resolved roots, the belt size and the recorded
    docling-serve observation, so a misconfigured mount — or a sidecar whose
    timeouts cannot be seen from its API — is visible without opening a shell.
``/livez``
    Liveness only, and O(1). ``/healthz`` re-scans the datasets root (up to
    ``inventory.MAX_SCAN_ENTRIES`` per dataset) because it is a browse surface
    for a person; that is the wrong cost for a 30-second Docker probe, so the
    image's ``HEALTHCHECK`` asks this instead.
``/`` and ``/report/``
    The datasets index: what is available to scout, and the frozen artifacts
    each dataset already holds.
``/report/<dataset>/<file>.html``
    A rendered artifact. Only ``report-*.html`` and ``session-*.html`` are
    servable, and only from a dataset directory directly under the workspace,
    with no directory listing of the workspace. The file is served as written,
    plus the reading chrome (nav bar, current stylesheet) when it was rendered
    before that chrome existed — a presentational splice at the edge, never a
    rewrite of the frozen file (``render.with_chrome``).
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
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from data_scout import (
    __version__,
    build,
    docling_service,
    inventory,
    registry,
    render,
    workspace,
)
from data_scout.mcp_server import MCP_PATH, build_server

HOST_ENV = "DATA_SCOUT_HTTP_HOST"
PORT_ENV = "DATA_SCOUT_HTTP_PORT"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8710

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
        # The writer and the reader must not each know a location: ask the one
        # function that owns the convention's two directories.
        files = _order_artifacts([path.name for path in workspace.artifact_paths(directory.name)])
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
        # The pin, read through read_pin — not read_frame. A dataset may be
        # pinned and unprofiled, and that combination must be visible rather
        # than raising DatasetUnprofiled. A corrupt pin degrades to pin_error
        # for this one entry: the index is a browse surface and must stay alive
        # when one dataset's pin cannot be read, while the frame path stays
        # strict.
        try:
            pin = workspace.read_pin(dataset_id)
        except workspace.PinUnreadable as exc:
            record["pinned"] = False
            record["dataset_commit"] = None
            record["pin_error"] = str(exc)
        else:
            record["pinned"] = pin is not None
            record["dataset_commit"] = pin.dataset_commit if pin else None
        # "Scouted" means something has begun: a frame, a workspace or a
        # finding. Counting files is not scouting.
        record["scouted"] = bool(
            record["workspace"] or record["frame"] or record["artifacts"] or record["findings_lines"]
        )
        datasets.append(record)

    return {
        "version": __version__,
        # The version says which release; the build says which commit. Rendered
        # reports are frozen, so "which image made this one" is a real question.
        "build": build.build_id(),
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
        '<meta name="color-scheme" content="dark">',
        "<title>data-scout datasets</title><style>",
        render.stylesheet(),
        "</style></head><body>",
        # The same chrome the reports carry, so the site is one thing on a phone.
        '<nav class="data-scout-nav"><span class="brand">data-scout</span>'
        '<span class="brand">datasets</span></nav>',
        "<h1>datasets</h1>",
        f'<p class="muted">workspace <code>{_escape(payload["workspace"])}</code> · '
        f'datasets <code>{_escape(payload["datasets_root"])}</code> · v{_escape(payload["version"])}'
        # Short for the eye, full in the tooltip: the hash is here to answer
        # "what am I running", and 40 characters inline is a wall.
        f' · build <code title="{_escape(payload["build"])}">'
        f'{_escape(build.short(payload["build"]))}</code></p>',
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


#: Artifacts in the order a reader wants them: the decision first, then the
#: evidence, then the raw session transcripts. Sorted alphabetically, a dataset
#: page leads with ``session-1.html`` — the least publishable thing in the
#: directory — and buries the brief it was written to support.
_ARTIFACT_ORDER = ("brief-", "report-", "session-")


def _order_artifacts(names: list[str]) -> list[str]:
    def rank(name: str) -> tuple[int, str]:
        for position, prefix in enumerate(_ARTIFACT_ORDER):
            if name.startswith(prefix):
                return (position, name)
        return (len(_ARTIFACT_ORDER), name)

    return sorted(names, key=rank)


def _dataset_html(dataset: dict[str, Any]) -> list[str]:
    """One dataset: what it is, what it costs to look, and what has been made."""
    name = dataset["name"]
    dataset_id = dataset["dataset_id"]
    parts = ['<section class="card">', f"<h2>{_escape(name)}"]
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
    # A frame number carries its rules where a reader sees it: "frame: true" over
    # a ledger holding both a v1 whole-checkout line and a v2 payload line hides
    # the distinction that makes those numbers incomparable.
    schemas = dataset.get("frame_schemas") or []
    if schemas:
        labels = ", ".join(f"v{version}" if version >= 0 else "v?" for version in schemas)
        status += f" · frame schema {labels}"
    elif dataset["frame"]:
        status += " · frame.md present, no frame ledger line"
    status_class = "status scouted" if dataset["scouted"] else "status"
    parts.append(f'<p class="{status_class}">{_escape(status)}</p>')

    # The brief is the front page: a dataset page should answer "what should I
    # run on this?" before it lists transcripts. It is pulled out of the list so
    # the reader is not asked to choose between it and its own link.
    pairs = list(zip(dataset["names"], dataset["artifacts"], strict=False))
    briefs = [pair for pair in pairs if pair[0].startswith("brief-")]
    rest = [pair for pair in pairs if not pair[0].startswith("brief-")]
    if briefs:
        _label, href = briefs[0]
        parts.append(
            f'<p class="brief-link"><a href="{_escape(href)}">'
            "What to run on this dataset &rarr;</a></p>"
        )
    if rest:
        parts.append("<ul>")
        for label, href in rest:
            parts.append(f'<li><a href="{_escape(href)}">{_escape(label)}</a></li>')
        parts.append("</ul>")
    else:
        parts.append('<p class="muted">No reports yet.</p>')
    parts.append("</section>")
    return parts


def resolve_artifact(path: str) -> Path | None:
    """Map ``/report/<dataset>/<file>`` to a file on disk, or ``None``.

    Two independent checks, both cheap: the dataset component must be a valid
    dataset id (no separators, no dots-dots), and the filename must resolve
    through :func:`workspace.artifact_path` — the same rule the writer uses.
    There is no path joining of untrusted input beyond that.
    """
    parts = [part for part in path.split("/") if part]
    if len(parts) != 3 or parts[0] != "report":
        return None
    _, dataset_id, filename = parts
    try:
        workspace.validate_dataset_id(dataset_id)
        candidate = workspace.artifact_path(dataset_id, filename)
    except workspace.WorkspaceError:
        return None
    return candidate if candidate.is_file() else None


def _json(payload: dict[str, Any], status: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status, headers={"Cache-Control": "no-store"})


def healthz(request: Request) -> Response:
    # The probe ran once in build_app; a health tick must not re-probe the sidecar.
    record = getattr(
        request.app.state,
        "docling",
        {"state": "not_probed", "detail": "This app was not built by build_app()."},
    )
    return _json(
        {
            "status": "ok",
            "version": __version__,
            "mcp": MCP_PATH,
            "belt": {"size": len(registry.BELT), "cap": registry.MAX_BELT_SIZE},
            "docling": record,
            **index_payload(),
        }
    )


def livez(request: Request) -> Response:
    """Liveness, without the inventory scan ``/healthz`` is entitled to do."""
    del request
    return _json({"status": "ok", "version": __version__, "build": build.build_id()})


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
    html = found.read_text(encoding="utf-8", errors="replace")
    return HTMLResponse(render.with_chrome(html), headers={"Cache-Control": "no-cache"})


def icon(request: Request) -> Response:
    filename = _ICONS.get(request.url.path)
    if filename is None:
        return _json({"error": "not_found", "path": request.url.path}, status=404)
    return FileResponse(asset_path(filename), media_type="image/png")


def build_app(docling: dict[str, Any] | None = None) -> Starlette:
    """The one app: MCP at ``/mcp``, reports under ``/report/``, on one port.

    The MCP routes are mounted first, so ``/mcp`` is the server's own handler and
    never falls through to the report routes. DNS-rebinding protection is off
    because this is dialed by container name on an internal network
    (``http://data-scout:8710/mcp``); the transport is not exposed publicly.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    observation = docling if docling is not None else docling_service.observe()
    server = build_server()
    app = server.streamable_http_app(
        streamable_http_path=MCP_PATH,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    app.router.routes.extend(
        [
            Route("/healthz", healthz),
            Route("/livez", livez),
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
    # One probe per process, taken before the first request: /healthz reports what
    # startup saw rather than re-probing a sidecar on every liveness tick.
    app.state.docling = observation
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
    observation = docling_service.observe()
    print(docling_service.summary(observation), flush=True)
    print(
        f"data-scout {__version__} serving on http://{host}:{port} "
        f"(mcp={MCP_PATH}, belt={len(registry.BELT)}, workspace={workspace.workspace_root()}, "
        f"datasets={workspace.datasets_root()})",
        flush=True,
    )
    uvicorn.run(build_app(docling=observation), host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
