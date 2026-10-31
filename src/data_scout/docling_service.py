"""``docling_service`` — what the sidecar says about itself, recorded at startup.

Item 5 of the belt-defects order, and the shape that let 4b hide for a whole
session: the job grants each document ``timeout 3600``, the belt's httpx client
keeps a 600 s read timeout, and docling-serve abandons a *synchronous* request at
``MAX_SYNC_WAIT`` (120 s by pydantic default). Three timeouts, three owners, and
the smallest one silently governed. Nothing in the belt could see that.

What this module does:

* probes ``/version`` and ``/openapi.json`` once at startup and records exactly
  what came back, including the HTTP status when it came back as a refusal;
* records **which** timeout governs a conversion now that ``partition_doc`` is
  asynchronous — the belt's own ``DATA_SCOUT_DOCLING_TASK_TIMEOUT`` — so the
  binding ceiling is a named fact instead of an inference;
* keeps "not configured", "unreachable" and "reports nothing" as three distinct
  states, in the same way item 2 keeps "no images" from "no exiftool".

What it deliberately does **not** do: invent a ``max_sync_wait``. Checked against
docling-serve upstream (``docling_serve/app.py``, ``settings.py``,
``capabilities.py``):

* ``/version`` returns ``DOCLING_VERSIONS`` — package versions — and 403 when
  ``show_version_info`` is false;
* ``/openapi.json`` is the request schema; ``max_sync_wait`` is a server setting,
  not a schema field, and the sync endpoints do not even document their 504;
* ``/v1/capabilities`` reports ``limits`` (``max_document_timeout``,
  ``max_sources_per_request``, …) but not this one.

So no endpoint reports it. The record therefore carries ``seconds: None`` with
``source: "unobservable"`` — never ``120``, because 120 is the pydantic default,
not an observation, and "never configured" and "configured to 120" are genuinely
indistinguishable from outside. A deployment that *declares* the value in this
process's environment is recorded as ``source: "declared"`` and is still never
called observed.
"""

from __future__ import annotations

import importlib
import os
from datetime import UTC, datetime
from typing import Any

from data_scout.tools._common import ToolUnavailable, seconds_env

#: The sidecar's own variable name, read here only as a *declaration*: if this
#: process happens to be told the value, the record can say so and label it.
DECLARED_MAX_SYNC_WAIT_ENV = "DOCLING_SERVE_MAX_SYNC_WAIT"

_VERSION_PATH = "/version"
_OPENAPI_PATH = "/openapi.json"
_ASYNC_SUBMIT_PATH = "/v1/convert/file/async"

#: A probe is a courtesy, not a dependency: short enough that an unreachable
#: sidecar delays startup by seconds rather than by the belt's task timeout.
PROBE_TIMEOUT = 5.0
CONNECT_TIMEOUT = 2.0

_UNOBSERVABLE_DETAIL = (
    "docling-serve exposes no endpoint that reports this: /version returns package "
    "versions, /openapi.json is the request schema, and /v1/capabilities reports "
    "other limits. It is a server setting, so the effective value is unknown from "
    "here — not zero, and not the 120 s pydantic default."
)

_DECLARED_DETAIL = (
    "Declared in this process's environment. That is a statement about the sidecar, "
    "not an observation of it: nothing on the sidecar reports this setting, so a "
    "declaration that disagrees with the deployment cannot be detected from here."
)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _endpoint(state: str, *, http_status: int | None = None, detail: str | None = None) -> dict[str, Any]:
    return {"state": state, "http_status": http_status, "detail": detail}


def _max_sync_wait() -> dict[str, Any]:
    """The sidecar's sync ceiling: declared by us, or honestly unknown."""
    record: dict[str, Any] = {
        "seconds": None,
        "source": "unobservable",
        "enforced_by": "docling-serve, on the synchronous convert endpoints only",
        "detail": _UNOBSERVABLE_DETAIL,
    }
    declared = os.environ.get(DECLARED_MAX_SYNC_WAIT_ENV, "").strip()
    if not declared:
        return record
    try:
        seconds = float(declared)
    except ValueError:
        record["detail"] = (
            f"{_UNOBSERVABLE_DETAIL} The declaration {DECLARED_MAX_SYNC_WAIT_ENV}="
            f"{declared!r} is not a number of seconds and was ignored."
        )
        return record
    record["seconds"] = seconds
    record["source"] = "declared"
    record["detail"] = _DECLARED_DETAIL
    return record


def _timeouts() -> dict[str, Any]:
    """The three ceilings, each named with its owner, and which one binds."""
    belt = importlib.import_module("data_scout.tools.partition_doc")
    task = seconds_env(belt.TASK_TIMEOUT_ENV, belt.DEFAULT_TASK_TIMEOUT, tool="partition_doc")
    return {
        "binding": {
            "seconds": task,
            "owner": "data_scout.tools.partition_doc",
            "env": belt.TASK_TIMEOUT_ENV,
            "scope": "one asynchronous task, from submit to result",
            "detail": (
                "This governs, and it is the belt's own deadline. The async submit "
                "returns in milliseconds, so the sidecar's sync ceiling is never "
                "waited on and cannot bind."
            ),
        },
        "request_read": {
            "seconds": belt.READ_TIMEOUT,
            "owner": "data_scout.tools.partition_doc",
            "scope": "one HTTP request — submit, poll or result fetch",
            "detail": "Bounds a single slow request, not a task: a poll returns in milliseconds.",
        },
        "sidecar_max_sync_wait": {
            "seconds": None,
            "owner": "docling-serve",
            "scope": "synchronous convert endpoints only; not used by this belt",
            "detail": _UNOBSERVABLE_DETAIL,
        },
    }


def _version_record(client: Any, base: str) -> dict[str, Any]:
    response = client.get(base + _VERSION_PATH)
    if response.status_code == 403:
        return {
            **_endpoint(
                "refused",
                http_status=403,
                detail=(
                    "The sidecar is up but configured with show_version_info=false, so it "
                    "refuses to name its versions. That is a configured 'no', not an absent "
                    "server and not an unknown one."
                ),
            ),
            "values": None,
        }
    if response.status_code != 200:
        return {
            **_endpoint(
                "missing",
                http_status=response.status_code,
                detail=f"{_VERSION_PATH} answered {response.status_code}; no versions were reported.",
            ),
            "values": None,
        }
    payload = response.json()
    values = payload if isinstance(payload, dict) else {"value": payload}
    return {**_endpoint("reported", http_status=200), "values": values}


def _openapi_record(client: Any, base: str) -> dict[str, Any]:
    response = client.get(base + _OPENAPI_PATH)
    if response.status_code in {403, 404}:
        return {
            **_endpoint(
                "missing",
                http_status=response.status_code,
                detail=(
                    f"{_OPENAPI_PATH} answered {response.status_code}; the deployment has "
                    "enable_api_docs=false. The schema is unavailable, which is not the same "
                    "as an empty schema."
                ),
            ),
            "paths": None,
            "async_convert": None,
        }
    if response.status_code != 200:
        return {
            **_endpoint(
                "missing",
                http_status=response.status_code,
                detail=f"{_OPENAPI_PATH} answered {response.status_code}.",
            ),
            "paths": None,
            "async_convert": None,
        }
    payload = response.json()
    paths = payload.get("paths") if isinstance(payload, dict) else None
    found = isinstance(paths, dict)
    return {
        **_endpoint("reported", http_status=200),
        "paths": len(paths) if found else None,
        "async_convert": (_ASYNC_SUBMIT_PATH in paths) if found else None,
    }


def _resolve_url(url: str | None) -> tuple[str | None, bool]:
    if url:
        return url.rstrip("/"), True
    try:
        belt = importlib.import_module("data_scout.tools.partition_doc")
        return belt.docling_url(), True
    except ToolUnavailable:
        return None, False


def observe(*, client: Any | None = None, url: str | None = None) -> dict[str, Any]:
    """Probe the sidecar once and return the record for ``/healthz`` and the log.

    Never raises for a sidecar that is absent, unreachable or refusing: the belt
    must still serve, and a probe that took the server down with it would be a
    worse defect than the one it reports. Loudness lives in the record's
    ``state`` and per-endpoint ``detail``, and in ``partition_doc``'s refusal at
    call time.
    """
    base, configured = _resolve_url(url)
    record: dict[str, Any] = {
        "state": "not_configured" if not configured else "probing",
        "url": base,
        "checked_at": _now(),
        "version": _endpoint("not_checked"),
        "openapi": _endpoint("not_checked"),
        "max_sync_wait": _max_sync_wait(),
        "timeouts": _timeouts(),
    }
    if not configured:
        reason = (
            "DOCLING_URL is not set, so the sidecar could not be probed. That is a "
            "deployment choice, not an empty sidecar: partition_doc will say the same "
            "thing when it is called."
        )
        record["version"]["detail"] = reason
        record["openapi"]["detail"] = reason
        return record

    owned = client is None
    try:
        if owned:
            import httpx

            client = httpx.Client(
                timeout=httpx.Timeout(PROBE_TIMEOUT, connect=CONNECT_TIMEOUT)
            )
        record["version"] = _version_record(client, base)
        record["openapi"] = _openapi_record(client, base)
    except Exception as exc:  # noqa: BLE001 - a probe must never take the server down
        detail = f"The probe failed ({type(exc).__name__}: {exc})."
        record["state"] = "unreachable"
        record["version"] = {**_endpoint("not_checked", detail=detail), "values": None}
        record["openapi"] = {
            **_endpoint("not_checked", detail=detail),
            "paths": None,
            "async_convert": None,
        }
        return record
    finally:
        if owned and client is not None:
            client.close()

    record["state"] = "reachable"
    return record


def summary(record: dict[str, Any]) -> str:
    """One startup line: the state, the version if known, and the binding ceiling."""
    binding = record["timeouts"]["binding"]
    ceiling = record["max_sync_wait"]
    if ceiling["seconds"] is None:
        sync = "sidecar sync ceiling unknown"
    else:
        sync = f"sidecar sync ceiling {ceiling['seconds']:g}s ({ceiling['source']})"
    version = record["version"].get("values") or {}
    named = version.get("docling-serve") if isinstance(version, dict) else None
    head = f"docling: {record['state']}"
    if named:
        head += f" (docling-serve {named})"
    return (
        f"{head}; {sync}; binding timeout {binding['seconds']:g}s "
        f"({binding['owner']}, {binding['env']})"
    )
