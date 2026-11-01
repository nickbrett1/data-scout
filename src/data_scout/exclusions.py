"""What is not the dataset.

A dataset in a checkout arrives wrapped in version control, and that wrapper is
not corpus: git rewrites its index and logs, DVC rewrites its own lockfiles, and
none of it is a book. Measuring it produces numbers that are *true and useless* —
"the most common file type is text/plain" is a statement about git hook scripts,
and a whole-tree hash reports a change every time a lockfile ticks over.

The rule therefore lives here, once, and everything that walks a dataset applies
it: the frame's identity, the toolbelt's file walks, the inventory's counts and
the report. `frame.EXCLUSION_NAMES` is still the name a reader sees in a frame
line, because the ledger already holds frames that used it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Exclusion:
    """One declared exclusion: a name, the path it covers, and why.

    ``name`` is what the frame records. ``path`` is a path relative to the
    dataset root. The pair is data, so a reader can see *what* was left out
    without reading a module's walk.
    """

    name: str
    path: str
    why: str


#: The explicit exclusion set. VCS transient state and the host's pin, none of
#: which is part of the corpus, and all of which change for reasons unrelated to
#: the data.
FRAME_EXCLUSIONS: tuple[Exclusion, ...] = (
    Exclusion(
        name="git-internals",
        path=".git",
        why="git rewrites its index, logs and refs on ordinary VCS activity",
    ),
    Exclusion(
        name="dvc-internals",
        path=".dvc",
        why=(
            "DVC rewrites .dvc/tmp/lock and .dvc/tmp/rwlock.lock on ordinary "
            "DVC activity; measured 9 -> 8 bytes with the payload untouched"
        ),
    ),
    Exclusion(
        name="dataset-pin",
        path="pin.json",
        why=(
            "written by /srv/bin/datasets on the host when a pin is recorded; it is a "
            "claim about the payload, not part of it, and it changes on ordinary "
            "pinning activity with the payload untouched"
        ),
    ),
)

#: The exclusion names in force. Recorded in every v2 frame.
EXCLUSION_NAMES: tuple[str, ...] = tuple(item.name for item in FRAME_EXCLUSIONS)

#: Directory names a walk must not descend into. A file's directory is a
#: version-control internal if any part of its relative path is one of these.
EXCLUDED_DIRS: frozenset[str] = frozenset(
    Path(item.path).parts[0]
    for item in FRAME_EXCLUSIONS
    if "/" not in item.path and not item.path.endswith(".json")
)

#: File names at the *root* of a dataset that are not part of it. These are not
#: pruned anywhere in the tree: ``pin.json`` is the host's claim about the
#: payload, and a file of that name deeper down is just a file.
EXCLUDED_FILES: frozenset[str] = frozenset(
    item.path for item in FRAME_EXCLUSIONS if "/" not in item.path and item.path.endswith(".json")
)

#: The same names as a tuple, for messages that read better than a set.
EXCLUDED_DIR_NAMES: tuple[str, ...] = tuple(sorted(EXCLUDED_DIRS))


def is_excluded_name(name: str) -> bool:
    """Is this directory (or root file) name outside the dataset?"""
    return name in EXCLUDED_DIRS or name in EXCLUDED_FILES


def relative_parts(path: str | Path, root: str | Path) -> tuple[str, ...]:
    """The parts of ``path`` below ``root``; the whole path if it is not below it."""
    try:
        return Path(path).relative_to(root).parts
    except ValueError:
        return Path(path).parts


def in_vcs(path: str | Path, root: str | Path) -> bool:
    """Does ``path`` sit inside a version-control internal of ``root``?"""
    return any(part in EXCLUDED_DIRS for part in relative_parts(path, root))
