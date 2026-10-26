"""The HTTP app: frozen reports on one port, and a documented ``/a2a`` seam.

One container, one published port, path-routed — because a second published port
would be a second service (``data-scout-genproj-scaffold`` §6.2):

``/healthz``
    Liveness plus the two resolved roots, so a misconfigured mount is visible
    without opening a shell.
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
    intelligence is a goose recipe driving these tools (design §2.1).

This serves the frozen artifacts and nothing else. It never writes, and it never
touches the corpora root.
"""

from __future__ import annotations

import json
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from data_scout import __version__, workspace

HOST_ENV = "DATA_SCOUT_HTTP_HOST"
PORT_ENV = "DATA_SCOUT_HTTP_PORT"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8710

_SERVABLE_RE = re.compile(r"^(report|session)-[A-Za-z0-9._-]*\.html$")

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


class Handler(BaseHTTPRequestHandler):
    server_version = f"data-scout/{__version__}"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
        route = self.path.split("?", 1)[0].rstrip("/") or "/"

        if route == "/healthz":
            self._json(HTTPStatus.OK, {"status": "ok", "version": __version__, **index_payload()})
            return
        if route == "/report/index.json" or route == "/index.json":
            self._json(HTTPStatus.OK, index_payload())
            return
        if route in {"/", "/report"}:
            self._text(HTTPStatus.OK, index_html(index_payload()), "text/html; charset=utf-8")
            return
        if route.startswith("/a2a"):
            self._json(
                HTTPStatus.NOT_IMPLEMENTED,
                {
                    "error": "not_implemented",
                    "detail": (
                        "The A2A listener is deploy-time wiring, not repo work. The "
                        "correct card URL is the container's name on the shared "
                        "network (an address is a fact about an observer, not about "
                        "a service). The intelligence is a goose recipe driving these "
                        "tools, not a hand-written agent loop."
                    ),
                },
            )
            return

        artifact = resolve_artifact(route)
        if artifact is not None:
            self._file(artifact)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found", "path": route})

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self._text(status, json.dumps(payload, indent=2, sort_keys=True), "application/json; charset=utf-8")

    def _text(self, status: HTTPStatus, body: str, content_type: str) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _file(self, path: Path) -> None:
        try:
            data = path.read_bytes()
        except OSError:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found", "path": str(path)})
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - base signature
        print(f"[data-scout] {self.address_string()} {format % args}", flush=True)


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
    image is the same one the report route belongs to — one container, one port.
    """
    del argv  # accepted so a future flag does not change the contract
    return serve()


def serve(host: str | None = None, port: int | None = None) -> int:
    host = host or os.environ.get(HOST_ENV, DEFAULT_HOST)
    port = port or int(os.environ.get(PORT_ENV, DEFAULT_PORT))
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(
        f"data-scout {__version__} serving on http://{host}:{port} "
        f"(workspace={workspace.workspace_root()}, corpora={workspace.corpora_root()})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
