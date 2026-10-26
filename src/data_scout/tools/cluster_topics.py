"""``cluster_topics`` — UMAP -> HDBSCAN -> c-TF-IDF labels over a sample.

Tier 2, pure over embeddings. **The outlier share is the number that matters**:
the percentage of the sample that fits no cluster is the finding for boilerplate
and noise, and it is returned as ``value`` rather than left for a caller to
derive from a topic table.

Needs the optional ``[semantic]`` extra (BERTopic, UMAP, HDBSCAN, scikit-learn)
and **fails loudly** without it — naming the extra and the install command.
There is no fallback clustering: a k-means substitute would produce a different
number under the same name, which is measurement drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from data_scout.tools._common import capped, measurement, require_import, share

_MAX_TOPIC_TERMS = 10


def cluster_topics(
    vectors: Sequence[Sequence[float]],
    texts: Sequence[str] | None = None,
    *,
    min_topic_size: int = 10,
    random_state: int = 42,
    top_n_words: int = _MAX_TOPIC_TERMS,
) -> dict[str, Any]:
    """Cluster embeddings and label the clusters.

    ``value`` is the outlier share (documents assigned no cluster), ``n`` the
    number of documents. ``texts`` is optional; with it the topic labels are
    c-TF-IDF terms, without it they are topic ids only.
    """
    require_import(
        "cluster_topics",
        "bertopic",
        extra="semantic",
        packages="bertopic, umap-learn, hdbscan, scikit-learn, numpy, scipy",
    )
    require_import("cluster_topics", "numpy", extra="semantic", packages="numpy")
    from bertopic import BERTopic

    documents = list(vectors)
    if not documents:
        raise ValueError("cluster_topics: no vectors to cluster")
    if len(documents) < min_topic_size * 2:
        raise ValueError(
            f"cluster_topics: {len(documents)} documents is too few to cluster "
            f"honestly with min_topic_size={min_topic_size}; HDBSCAN would report "
            "one topic and an outlier share that means nothing. Draw a larger sample."
        )
    if texts is not None and len(texts) != len(documents):
        raise ValueError(
            f"cluster_topics: {len(documents)} vectors but {len(texts)} texts"
        )

    model = BERTopic(
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=False,
        nr_topics="auto",
        top_n_words=top_n_words,
    )
    topics, _probabilities = model.fit_transform(list(texts) if texts else None, documents)

    counts: dict[int, int] = {}
    for topic in topics:
        counts[int(topic)] = counts.get(int(topic), 0) + 1
    outliers = counts.pop(-1, 0)

    topic_rows: list[dict[str, Any]] = []
    for topic_id, size in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        terms: list[str] = []
        if texts:
            try:
                terms = [word for word, _weight in model.get_topic(topic_id) or []][:top_n_words]
            except Exception:  # noqa: BLE001 - labels are enrichment, the cluster is the measurement
                terms = []
        topic_rows.append({"topic_id": topic_id, "size": size, "share": share(size, len(documents)), "terms": terms})
    inline, trimmed = capped(topic_rows, 50)

    label = model.get_topic_info() if texts else None

    return measurement(
        value=share(outliers, len(documents)),
        n=len(documents),
        unit_notes="share of documents fitting no cluster",
        outlier_share=share(outliers, len(documents)),
        outlier_count=outliers,
        topic_count=len(counts),
        topics=inline,
        topics_truncated=trimmed,
        topic_info_size=0 if label is None else int(len(label)),
        min_topic_size=min_topic_size,
        random_state=random_state,
        labelled=texts is not None,
    )
