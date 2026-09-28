"""Retrieval quality metrics, including the shortfall measures that motivate vecverdict.

Most vector-search benchmarks report latency and recall. They rarely report that
a graph-based index *returned fewer results than asked for* — a filtered query
against HNSW can quietly come back with 4 hits when k=10, and standard recall
averaging hides this inside a lower mean. Shortfall is broken out explicitly here
because "returned fewer results than requested" is a correctness failure, not a
quality gradient.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# A result id meaning "the backend returned no candidate in this slot". Backends
# that pad short result sets must map their padding onto this sentinel so the
# shortfall accounting stays honest.
MISSING = -1


@dataclass(frozen=True)
class RetrievalQuality:
    """Quality of one backend's results for one query set.

    Attributes:
        k: Number of results requested per query.
        n_queries: Number of queries evaluated.
        recall_at_k: Mean fraction of exact top-k neighbours retrieved.
        recall_at_1: Mean fraction of queries whose exact top-1 was retrieved.
        mean_returned: Mean number of non-missing results per query.
        shortfall_rate: Fraction of queries returning fewer than k results.
        mean_shortfall: Mean count of missing results per query.
        worst_shortfall: Largest number of missing results for any single query.
        attainable_recall_at_k: Recall measured against min(k, n_allowed)
            neighbours. When a filter allows fewer than k vectors, no backend can
            return k, so this separates "the filter was narrow" from "the index
            failed".
    """

    k: int
    n_queries: int
    recall_at_k: float
    recall_at_1: float
    mean_returned: float
    shortfall_rate: float
    mean_shortfall: float
    worst_shortfall: int
    attainable_recall_at_k: float
    per_query_recall: np.ndarray = field(repr=False, compare=False)
    per_query_returned: np.ndarray = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, float | int]:
        """Serialisable summary, excluding the per-query arrays."""
        data = asdict(self)
        data.pop("per_query_recall", None)
        data.pop("per_query_returned", None)
        return data


def evaluate(
    retrieved: np.ndarray,
    truth: np.ndarray,
    k: int,
    n_allowed: int | np.ndarray | None = None,
) -> RetrievalQuality:
    """Score retrieved neighbours against exact ground truth.

    Args:
        retrieved: Backend results, shape (n_queries, k_returned). Slots with no
            result must be MISSING. Ragged result sets should be padded with
            MISSING to a rectangular array.
        truth: Exact neighbours from groundtruth.compute, shape (n_queries, k_true).
        k: Number of results that were requested.
        n_allowed: Number of vectors the filter permitted — an int if constant
            across queries, or a per-query array. Used for attainable recall.

    Returns:
        RetrievalQuality with both standard recall and shortfall measures.
    """
    retrieved = np.atleast_2d(np.asarray(retrieved, dtype=np.int64))
    truth = np.atleast_2d(np.asarray(truth, dtype=np.int64))
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if retrieved.shape[0] != truth.shape[0]:
        raise ValueError(
            f"query count mismatch: retrieved {retrieved.shape[0]} vs truth {truth.shape[0]}"
        )
    if retrieved.shape[1] > k:
        raise ValueError(
            f"backend returned {retrieved.shape[1]} results for k={k}; "
            "returning more than requested is a backend bug"
        )

    n_queries = retrieved.shape[0]
    truth_k = truth[:, :k]

    valid = retrieved != MISSING
    returned = valid.sum(axis=1).astype(np.int64)

    # Ground truth itself may be short when the allowlist was narrow, so the
    # denominator is per-query rather than a flat k.
    truth_valid = truth_k != MISSING
    truth_counts = truth_valid.sum(axis=1).astype(np.int64)

    hits = np.empty(n_queries, dtype=np.int64)
    top1_hits = np.zeros(n_queries, dtype=np.bool_)
    for i in range(n_queries):
        expected = truth_k[i][truth_valid[i]]
        actual = retrieved[i][valid[i]]
        hits[i] = np.intersect1d(expected, actual, assume_unique=False).size
        if expected.size:
            top1_hits[i] = bool(np.isin(expected[0], actual))

    denom = np.maximum(truth_counts, 1)
    per_query_recall = (hits / denom).astype(np.float64)
    # A query with no valid ground truth is undefined, not zero; exclude it.
    scored = truth_counts > 0
    recall_at_k = float(per_query_recall[scored].mean()) if scored.any() else 0.0
    recall_at_1 = float(top1_hits[scored].mean()) if scored.any() else 0.0

    if n_allowed is None:
        attainable = np.full(n_queries, k, dtype=np.int64)
    else:
        attainable = np.minimum(np.asarray(n_allowed, dtype=np.int64), k)
        attainable = np.broadcast_to(attainable, (n_queries,))
    # Only count a shortfall the backend could actually have avoided.
    avoidable = np.maximum(attainable - returned, 0)

    attainable_denom = np.maximum(np.minimum(truth_counts, attainable), 1)
    attainable_recall = (
        float((hits[scored] / attainable_denom[scored]).mean()) if scored.any() else 0.0
    )

    return RetrievalQuality(
        k=k,
        n_queries=int(n_queries),
        recall_at_k=recall_at_k,
        recall_at_1=recall_at_1,
        mean_returned=float(returned.mean()),
        shortfall_rate=float((avoidable > 0).mean()),
        mean_shortfall=float(avoidable.mean()),
        worst_shortfall=int(avoidable.max()) if n_queries else 0,
        attainable_recall_at_k=min(attainable_recall, 1.0),
        per_query_recall=per_query_recall,
        per_query_returned=returned,
    )


def pad_to_k(results: list[np.ndarray] | list[list[int]], k: int) -> np.ndarray:
    """Pad ragged per-query result lists into an (n_queries, k) MISSING-padded array."""
    out = np.full((len(results), k), MISSING, dtype=np.int64)
    for i, row in enumerate(results):
        row_array = np.asarray(row, dtype=np.int64).ravel()
        if row_array.size > k:
            raise ValueError(f"query {i} returned {row_array.size} results, more than k={k}")
        out[i, : row_array.size] = row_array
    return out
