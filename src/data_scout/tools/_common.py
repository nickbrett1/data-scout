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
from typing import Any, NoReturn

#: The three keys every instrument returns (``data-scout-convention`` §5).
RESULT_KEYS = ("value", "n", "unit_notes")


class NotATextSource(ValueError):
    """A source that is not text and is not a file either.

    Raised instead of silently measuring a pathname as if it were prose: that
    failure shape is a plausible number with a false unit and no signal.
    """


class ToolUnavailable(RuntimeError):
    """A tool's instrument is not present. Loud, named, actionable."""

    def __init__(self, tool: str, missing: str, hint: str) -> None:
        self.tool = tool
        self.missing = missing
        self.hint = hint
        super().__init__(f"{tool}: {missing} is not available. {hint}")


def absent_is_not_zero(*, tool: str, what: str, hint: str) -> NoReturn:
    """Raise, rather than let "we did not observe this" read as a zero.

    One helper and one error shape for the whole belt. "No images" and "no
    exiftool" are the same number and opposite facts (item 2); so are "no parts"
    and "no cached observation" (item 4). Every path that could otherwise
    return an empty list or a clean zero calls this instead, naming what is
    missing and how to get it, so absence is never mistaken for a measurement.
    """
    raise ToolUnavailable(tool, what, hint)


def require_binary(tool: str, binary: str, *, package: str) -> str:
    """Return the path to ``binary`` or fail loudly.

    A missing tier-0 binary must never degrade into a zero: "no fonts" and "no
    pdffonts" are the same number and opposite facts.
    """
    found = shutil.which(binary)
    if found is None:
        absent_is_not_zero(
            tool=tool,
            what=f"the `{binary}` binary",
            hint=(
                f"Install {package} (apt: apt-get install -y --no-install-recommends "
                f"{package}). It is in the runtime image's tier-0 belt."
            ),
        )
    return found


def require_import(tool: str, module: str, *, extra: str, packages: str | None = None) -> Any:
    """Import an optional dependency or fail loudly, naming the extra."""
    try:
        return __import__(module, fromlist=["_"])
    except ImportError:
        names = packages or module
        absent_is_not_zero(
            tool=tool,
            what=f"the `{module}` package",
            hint=(
                f"It needs the optional `{extra}` extra: pip install -e \".[{extra}]\" "
                f"(brings {names}). Refusing to return a misleading default instead."
            ),
        )


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


#: Suffixes that make a bare string unambiguously "I meant a file". A string
#: carrying one of these is never silently measured as prose.
_PATH_SUFFIXES = frozenset(
    {
        ".azw3", ".bmp", ".csv", ".doc", ".docx", ".epub", ".gif", ".gz", ".heic",
        ".htm", ".html", ".ipynb", ".jpeg", ".jpg", ".json", ".jsonl", ".md",
        ".mobi", ".odt", ".pdf", ".png", ".ppt", ".pptx", ".rdf", ".rst", ".rtf",
        ".tar", ".tif", ".tiff", ".tsv", ".txt", ".webp", ".xlsx", ".xml",
        ".yaml", ".yml", ".zip",
    }
)


def _looks_like_path(value: str) -> bool:
    """Whether a short string is a *candidate path* rather than literal text.

    Deliberately conservative in the other direction: a long string or one with
    a newline is text, because asking the filesystem about a 4 KB paragraph
    raises "file name too long" and a whole document is obviously not a path.
    """
    if value.startswith(("http://", "https://")):
        return False
    if "/" in value or "\\" in value:
        return True
    return Path(value).suffix.lower() in _PATH_SUFFIXES


def as_texts(source: Any) -> list[str]:
    """Accept a path, a string, or a sequence of either, and return texts.

    The instruments take ``measure(text | rows | path)``; this is the one place
    that decides what those three words mean, so no tool grows its own dialect.

    **A pathname is never measured as if it were text.** The old behaviour fell
    back to the literal string whenever ``Path(source).is_file()`` was false, so
    a relative path under a different CWD became a clean measurement of a
    pathname — a plausible number with a false unit and no error. Now a string
    that looks like a path must be a file, and a path-shaped object must be a
    file too; otherwise :class:`NotATextSource` names the CWD and asks for the
    real path. Literal text still passes through when it is clearly prose.
    """
    if isinstance(source, os.PathLike) and not isinstance(source, str):
        path = Path(source)
        if path.is_dir():
            raise NotATextSource(
                f"as_texts: {path} is a directory, not a text source. Pass the "
                "files to measure (or a list of them); a directory is not text."
            )
        if not path.is_file():
            raise NotATextSource(f"as_texts: {path} is not a file")
        return [read_text(path)]
    if isinstance(source, str):
        if len(source) > 4096 or "\n" in source:
            return [source]
        candidate = Path(source)
        try:
            if candidate.is_file():
                return [read_text(candidate)]
            if candidate.is_dir():
                raise NotATextSource(
                    f"as_texts: {source!r} is a directory, not a text source. Pass "
                    "the files to measure; a directory is not text."
                )
        except OSError:
            pass
        if _looks_like_path(source):
            raise NotATextSource(
                f"as_texts: {source!r} looks like a path but is not a file "
                f"(the current directory is {Path.cwd()}). Refusing to measure the "
                "pathname as if it were text: pass an absolute path to the file, "
                "or pass the text itself."
            )
        return [source]
    if isinstance(source, list | tuple):
        texts: list[str] = []
        for item in source:
            texts.extend(as_texts(item))
        return texts
    raise TypeError(f"expected text, a path, or a list of those, got {type(source).__name__}")
