"""``exif_census`` — exiftool over a tree, aggregated. Tier 0, exhaustive.

The distinction it exists for: a **photograph** carries a camera Make/Model, a
**screenshot** carries none, and a **page render** carries none either. So the
headline number is ``photographic_share`` — the share of images with camera
metadata — which separates a photo library from a dataset of rendered pages
before anyone looks at a single pixel.

Metadata is aggregated only; a full exiftool dump is megabytes per file and the
report inherits every byte of it.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from data_scout.tools._common import (
    capped,
    iter_files,
    measurement,
    require_binary,
    share,
)

_MAX_DETAIL = 25
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".webp", ".gif", ".bmp", ".dng", ".cr2", ".nef", ".arw")


def _empty_census() -> dict[str, Any]:
    """No images is a measurement of zero, not a failed instrument."""
    return measurement(
        value=0.0,
        n=0,
        unit_notes="share of images carrying camera metadata",
        photographic_share=0.0,
        by_make={},
        by_model={},
        by_software={},
        by_file_type={},
        orientation={},
        gps_share=0.0,
        with_camera_metadata=0,
        without_camera_metadata=0,
        resolution={},
        date_range={},
        examples=[],
    )


def _records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    exiftool = require_binary("exif_census", "exiftool", package="libimage-exiftool-perl")
    args = [exiftool, "-json", "-q", "-q", "-r"]
    # Only the tags we aggregate: a full dump is the payload problem in person.
    args += [
        "-FileName",
        "-FileType",
        "-Make",
        "-Model",
        "-Software",
        "-Orientation",
        "-ImageWidth",
        "-ImageHeight",
        "-GPSLatitude",
        "-DateTimeOriginal",
        "-CreateDate",
        "-XResolution",
        "-YResolution",
    ]
    if limit is not None:
        args += ["-fileOrder", "FileName"]
    args.append(str(path))
    try:
        completed = subprocess.run(  # noqa: S603 - argv list, no shell
            args, capture_output=True, text=True, check=False, timeout=1800
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"exif_census: exiftool failed on {path}: {exc}") from exc
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"exif_census: exiftool exited {completed.returncode}: {completed.stderr.strip()[:500]}"
        )
    try:
        data = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"exif_census: exiftool returned non-JSON output: {exc}") from exc
    records = data if isinstance(data, list) else [data]
    if limit is not None:
        records = records[:limit]
    return records


def exif_census(
    path: str | os.PathLike[str],
    *,
    max_files: int | None = None,
    limit_details: bool = True,
) -> dict[str, Any]:
    """Census the EXIF of one image or of every image under ``path``.

    ``value`` is ``photographic_share``: the share of files carrying a camera
    Make or Model. ``n`` is the number of files exiftool reported on.
    """
    target = Path(path)
    if target.is_dir() and not any(iter_files(target, suffixes=_IMAGE_SUFFIXES, limit=1)):
        return _empty_census()
    if target.is_dir():
        # exiftool -r walks the tree itself; the suffix filter below only trims
        # the *aggregate*, so a dataset of screenshots is still counted.
        files = [item for item in iter_files(target, suffixes=_IMAGE_SUFFIXES, limit=max_files)]
        records = _records(target, max_files)
        if files and len(records) > len(files):
            records = records[: len(files)]
    else:
        records = _records(target, max_files)

    if not records:
        return _empty_census()

    by_make: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    by_software: Counter[str] = Counter()
    by_file_type: Counter[str] = Counter()
    orientation: Counter[str] = Counter()
    widths: Counter[int] = Counter()
    dates: list[str] = []
    photographic = 0
    gps = 0
    examples: list[dict[str, Any]] = []

    for record in records:
        make = str(record.get("Make") or "").strip()
        model = str(record.get("Model") or "").strip()
        software = str(record.get("Software") or "").strip()
        file_type = str(record.get("FileType") or "<unknown>").strip()
        if make or model:
            photographic += 1
            by_make[make or "<unstated>"] += 1
            by_model[model or "<unstated>"] += 1
        if software:
            by_software[software[:60]] += 1
        by_file_type[file_type] += 1
        if record.get("Orientation") is not None:
            orientation[str(record.get("Orientation"))] += 1
        if record.get("GPSLatitude") is not None:
            gps += 1
        width = record.get("ImageWidth")
        if isinstance(width, int):
            widths[width] += 1
        for key in ("DateTimeOriginal", "CreateDate"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                dates.append(value.strip())
                break
        if len(examples) < 5:
            examples.append(
                {
                    "file": record.get("FileName"),
                    "type": file_type,
                    "make": make or None,
                    "model": model or None,
                    "software": software or None,
                }
            )

    makes, makes_trimmed = capped(by_make.most_common(), _MAX_DETAIL) if limit_details else (by_make.most_common(), False)
    models, models_trimmed = capped(by_model.most_common(), _MAX_DETAIL) if limit_details else (by_model.most_common(), False)
    software, software_trimmed = capped(by_software.most_common(), _MAX_DETAIL) if limit_details else (by_software.most_common(), False)

    return measurement(
        value=share(photographic, len(records)),
        n=len(records),
        unit_notes="share of images carrying camera metadata",
        photographic_share=share(photographic, len(records)),
        with_camera_metadata=photographic,
        without_camera_metadata=len(records) - photographic,
        by_make=dict(makes),
        by_make_trimmed=makes_trimmed,
        by_model=dict(models),
        by_model_trimmed=models_trimmed,
        by_software=dict(software),
        by_software_trimmed=software_trimmed,
        by_file_type=dict(by_file_type.most_common(_MAX_DETAIL)),
        orientation=dict(orientation.most_common()),
        gps_share=share(gps, len(records)),
        resolution={str(width): count for width, count in widths.most_common(_MAX_DETAIL)},
        date_range={"earliest": min(dates), "latest": max(dates)} if dates else {},
        examples=examples,
    )
