"""Which build is running: the commit the process was built from.

A version number answers "which release", not "which commit". Everything that
matters on this service — a render fix, a ledger rule — lands between releases,
so the datasets index shows the build stamp and ``/healthz`` carries it too.

Resolution order is deliberate:

``DATA_SCOUT_BUILD``
    The image stamps this from a build arg at build time. In a container this is
    the only source that can answer: the final stage has no ``.git`` and no git
    binary, which is exactly why the stamp is baked in rather than read.
git
    A source checkout (``pip install -e .``, a developer's venv) has no build
    arg, so the commit is asked for once and cached. Absent git or a repository,
    this degrades to ``unknown`` — a browse surface must not raise because
    provenance is unavailable (same rule as a corrupt pin).
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

BUILD_ENV = "DATA_SCOUT_BUILD"
UNKNOWN = "unknown"

# ``src/data_scout/build.py`` -> the repository root. Only meaningful for a
# source checkout; in an installed package there is simply no ``.git`` above it.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def build_id() -> str:
    """The commit this process was built from, or ``unknown``.

    The environment is read per call because it is the deployment's answer and
    costs nothing; only the git fallback is cached.
    """
    stamped = os.environ.get(BUILD_ENV, "").strip()
    if stamped:
        return stamped
    return _git_commit() or UNKNOWN


def short(build: str | None = None, length: int = 12) -> str:
    """The form to show a person: a 40-character SHA is a wall, not a label."""
    value = build_id() if build is None else build
    return value if len(value) <= length else value[:length]


@lru_cache(maxsize=1)
def _git_commit() -> str | None:
    """``git rev-parse HEAD``, or ``None`` when this is not a checkout.

    Cached for the process: the answer cannot change without a restart, and
    spawning a subprocess per request would be a cost on a browse surface.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None
