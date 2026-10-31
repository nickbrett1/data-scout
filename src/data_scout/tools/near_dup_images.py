"""``near_dup_images`` — phash clusters. Tier 1, pure.

Reuses the **approach** of reader's image dedupe (perceptual hash, cluster by
Hamming distance), not its code: the one-way import rule again.

The number that matters is ``duplicate_share``. Above ~0.5 the dataset is
mostly re-renders of the same pages and the unit is probably a page, not an
image; that is a phase-3 decision this instrument informs.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from data_scout.tools._common import (
    capped,
    iter_files,
    measurement,
    require_import,
    share,
)

_MAX_CLUSTER_MEMBERS = 20
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".gif", ".bmp")


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def near_dup_images(
    source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    *,
    max_distance: int = 8,
    hash_size: int = 8,
    max_files: int | None = None,
) -> dict[str, Any]:
    """Cluster images by perceptual hash.

    ``value`` is the share of images that are a near duplicate of another.
    ``max_distance`` is Hamming distance over a 64-bit phash; 8 is the usual
    "same picture, re-encoded or resized" threshold.

    A file that cannot be decoded is counted as ``undecodable`` and excluded
    from the denominator's claim — it is a finding, not a duplicate.
    """
    imagehash = require_import("near_dup_images", "imagehash", extra="dev", packages="imagehash")
    require_import("near_dup_images", "PIL", extra="dev", packages="pillow")

    if isinstance(source, str | os.PathLike):
        target = Path(source)
        paths = list(iter_files(target, suffixes=None if target.is_file() else _IMAGE_SUFFIXES, limit=max_files))
    else:
        paths = [Path(item) for item in source]

    hashes: list[tuple[Path, Any]] = []
    undecodable: list[str] = []
    for path in paths:
        try:
            with path.open("rb") as handle:
                digest = imagehash.phash(_load_image(handle), hash_size=hash_size)
        except Exception as exc:  # noqa: BLE001 - an unreadable image is a finding
            undecodable.append(f"{path}: {type(exc).__name__}: {exc}"[:200])
            continue
        hashes.append((path, digest))

    union = _UnionFind(len(hashes))
    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            if hashes[left][1] - hashes[right][1] <= max_distance:
                union.union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(len(hashes)):
        groups.setdefault(union.find(index), []).append(index)
    clusters = [sorted(members) for members in groups.values() if len(members) > 1]
    clusters.sort(key=lambda members: (-len(members), members[0]))

    duplicates = sum(len(members) - 1 for members in clusters)
    detail = [
        {
            "size": len(members),
            "paths": [str(hashes[index][0]) for index in members[:_MAX_CLUSTER_MEMBERS]],
            "truncated": len(members) > _MAX_CLUSTER_MEMBERS,
        }
        for members in clusters
    ]
    inline, trimmed = capped(detail, 50)

    return measurement(
        value=share(duplicates, len(hashes)),
        n=len(paths),
        unit_notes="share of decoded images that are a near duplicate of another",
        duplicate_share=share(duplicates, len(hashes)),
        duplicate_count=duplicates,
        cluster_count=len(clusters),
        clusters=inline,
        clusters_truncated=trimmed,
        decoded=len(hashes),
        undecodable=len(undecodable),
        undecodable_examples=undecodable[:10],
        max_distance=max_distance,
        hash_size=hash_size,
    )


def _load_image(handle: Any) -> Any:
    from PIL import Image

    image = Image.open(handle)
    image.load()
    if image.mode not in {"L", "RGB"}:
        image = image.convert("RGB")
    return image
