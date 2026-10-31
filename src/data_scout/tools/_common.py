"""Shared plumbing for the instrument belt.

Nothing here measures anything. It exists so that the three rules the order
states in prose are enforced in one place instead:

* an unavailable binary or optional extra **fails loudly, naming what is
  missing and how to install it** — never a zero, never an empty list, never a
  plausible-looking default;
* every instrument returns the same ``{value, n, unit_notes}`` shape, with the
  detail it carries as **first-class fields** beside it;
* payloads are capped, because Docling output is megabytes per document and the
  rendered HTML inherits every byte of it.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

#: The three keys every instrument returns (``data-scout-convention`` §5).
RESULT_KEYS = ("value", "n", "unit_notes")


class ToolUnavailable(RuntimeError):
    """A tool's instrument is not present. Loud, named, actionable."""

    def __init__(self, tool: str, missing: str, hint: str) -> None:
        self.tool = tool
        self.missing = missing
        self.hint = hint
        super().__init__(f"{tool}: {missing} is not available. {hint}")


def require_binary(tool: str, binary: str, *, package: str) -> str:
    """Return the path to ``binary`` or fail loudly.

    A missing tier-0 binary must never degrade into a zero: "no fonts" and "no
    pdffonts" are the same number and opposite facts.
    """
    found = shutil.which(binary)
    if found is None:
        raise ToolUnavailable(
            tool,
            f"the `{binary}` binary",
            f"Install {package} (apt: apt-get install -y --no-install-recommends "
            f"{package}). It is in the runtime image's tier-0 belt.",
        )
    return found


def require_import(tool: str, module: str, *, extra: str, packages: str | None = None) -> Any:
    """Import an optional dependency or fail loudly, naming the extra."""
    try:
        return __import__(module, fromlist=["_"])
    except ImportError:
        names = packages or module
        raise ToolUnavailable(
            tool,
            f"the `{module}` package",
            f"It needs the optional `{extra}` extra: pip install -e \".[{extra}]\" "
            f"(brings {names}). Refusing to return a misleading default instead.",
        ) from None


def measurement(value: Any, n: int, unit_notes: str, **extra: Any) -> dict[str, Any]:
    """The one result shape: the number, what it is a number of, and the detail.

    ``extra`` keys must be named, never a generic ``detail`` blob — a caller has
    to be able to test ``font_count == 0`` without parsing prose.
    """
    result: dict[str, Any] = {"value": _stable(value), "n": int(n), "unit_notes": unit_notes}
    result.update({key: _stable(item) for key, item in extra.items()})
    return result


def _stable(value: Any) -> Any:
    """Round floats so a re-execution reproduces a line byte for byte.

    Six decimal places is well past any decision this project makes, and it
    keeps floating-point summation order from failing the §9 diff for a reason
    nobody cares about.
    """
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _stable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable(item) for item in value]
    if isinstance(value, tuple):
        return [_stable(item) for item in value]
    return value


def share(part: int, whole: int) -> float:
    return float(part) / float(whole) if whole else 0.0


def capped(items: Iterable[Any], limit: int) -> tuple[list[Any], bool]:
    """Trim a payload to ``limit`` items, saying whether it was trimmed."""
    out: list[Any] = []
    for item in items:
        if len(out) >= limit:
            return out, True
        out.append(item)
    return out, False


def iter_files(
    path: str | os.PathLike[str],
    *,
    suffixes: Iterable[str] | None = None,
    limit: int | None = None,
    follow_symlinks: bool = False,
) -> Iterator[Path]:
    """Yield files under ``path`` (or ``path`` itself), read-only, sorted.

    Sorted so a sample or a census is deterministic without a seed, and
    symlinks are not followed by default so a dataset cannot walk out of its own
    mount through a link.

    **Opens nothing for writing.** There is deliberately no helper here that
    creates a file under a dataset.
    """
    wanted = {suffix.lower() for suffix in suffixes} if suffixes else None
    root = Path(path)
    if root.is_file():
        candidates: Iterator[Path] = iter([root])
    elif root.is_dir():
        candidates = (
            Path(directory) / name
            for directory, _dirs, files in os.walk(root, followlinks=follow_symlinks)
            for name in sorted(files)
        )
    else:
        candidates = iter(())
    count = 0
    for candidate in candidates:
        if wanted is not None and candidate.suffix.lower() not in wanted:
            continue
        if limit is not None and count >= limit:
            return
        count += 1
        yield candidate


def read_text(path: str | os.PathLike[str], *, max_bytes: int | None = None) -> str:
    """Read a text-ish file for measurement. Read-only, always."""
    data = Path(path).read_bytes()
    if max_bytes is not None:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def as_texts(source: Any) -> list[str]:
    """Accept a path, a string, or a sequence of either, and return texts.

    The instruments take ``measure(text | rows | path)``; this is the one place
    that decides what those three words mean, so no tool grows its own dialect.
    """
    if isinstance(source, str | os.PathLike):
        # A string is a path only if it can be one: a 4 KB paragraph is text,
        # and asking the filesystem about it raises "file name too long".
        if isinstance(source, str) and (len(source) > 4096 or "\n" in source):
            return [source]
        candidate = Path(source)
        try:
            if candidate.is_file():
                return [read_text(candidate)]
        except OSError:
            pass
        return [str(source)]
    if isinstance(source, list | tuple):
        texts: list[str] = []
        for item in source:
            texts.extend(as_texts(item))
        return texts
    raise TypeError(f"expected text, a path, or a list of those, got {type(source).__name__}")
