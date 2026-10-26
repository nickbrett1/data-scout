"""``near_dup_docs`` — MinHash/LSH over documents. Tier 1, pure.

Catches the two redundancy cases that matter and look different:

* **one work in three manifestations** — the same book as epub, mobi and pdf;
* **boilerplate chunks** — the same front matter, copyright page or OCR header
  repeated across a hundred files.

The headline is the **duplicate share**: the share of documents that are a near
duplicate of an earlier one. That is the number a session acts on when it
decides the unit is a *work* and not a *file* (phase 3).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from data_scout.tools._common import (
    as_texts,
    capped,
    measurement,
    require_import,
    share,
)

_SHINGLE = 4
_MAX_CLUSTER_MEMBERS = 20


def _shingles(words: Sequence[str], size: int = _SHINGLE) -> set[str]:
    if len(words) <= size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}


def _fingerprints(texts: Iterable[str], num_perm: int) -> list[Any]:
    module = require_import(
        "near_dup_docs",
        "datasketch",
        extra="dev",
        packages="datasketch",
    )
    fingerprints = []
    for text in texts:
        words = text.lower().split()
        minhash = module.MinHash(num_perm=num_perm)
        shingles = _shingles(words)
        if not shingles:
            fingerprints.append(None)
            continue
        for shingle in shingles:
            minhash.update(shingle.encode("utf-8"))
        fingerprints.append(minhash)
    return fingerprints


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


def near_dup_docs(
    source: Any,
    *,
    threshold: float = 0.9,
    num_perm: int = 64,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Cluster documents by near-duplicate text.

    ``value`` is the share of documents in a cluster of more than one. ``n`` is
    the document count. Documents with no text are counted and reported
    separately: an empty document is not a duplicate of another empty one, it is
    a missing measurement.
    """
    module = require_import("near_dup_docs", "datasketch", extra="dev", packages="datasketch")
    texts = as_texts(source)
    if labels is None:
        labels = [str(index) for index in range(len(texts))]
    if len(labels) != len(texts):
        raise ValueError("near_dup_docs: labels must have one entry per document")

    fingerprints = _fingerprints(texts, num_perm)
    lsh = module.MinHashLSH(threshold=threshold, num_perm=num_perm)
    present = [(index, fingerprint) for index, fingerprint in enumerate(fingerprints) if fingerprint is not None]
    for index, fingerprint in present:
        lsh.insert(str(index), fingerprint)

    union = _UnionFind(len(texts))
    for index, fingerprint in present:
        for candidate in lsh.query(fingerprint):
            other = int(candidate)
            if other != index:
                union.union(index, other)

    groups: dict[int, list[int]] = {}
    for index in range(len(texts)):
        groups.setdefault(union.find(index), []).append(index)
    clusters = [sorted(members) for members in groups.values() if len(members) > 1]
    clusters.sort(key=lambda members: (-len(members), members[0]))

    duplicates = sum(len(members) - 1 for members in clusters)
    detail = [
        {
            "size": len(members),
            "labels": [labels[index] for index in members[:_MAX_CLUSTER_MEMBERS]],
            "truncated": len(members) > _MAX_CLUSTER_MEMBERS,
        }
        for members in clusters
    ]
    inline, trimmed = capped(detail, 50)

    return measurement(
        value=share(duplicates, len(texts)),
        n=len(texts),
        unit_notes="share of documents that are a near duplicate of another",
        duplicate_share=share(duplicates, len(texts)),
        duplicate_count=duplicates,
        cluster_count=len(clusters),
        clusters=inline,
        clusters_truncated=trimmed,
        empty_documents=sum(1 for fingerprint in fingerprints if fingerprint is None),
        threshold=threshold,
        num_perm=num_perm,
        shingle_words=_SHINGLE,
    )
