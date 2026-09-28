"""Exact nearest-neighbour ground truth.

Every recall number in vecverdict is measured against exhaustive, unquantised
search computed here. No approximate index is ever used as a reference, because
an approximate reference silently caps the recall a candidate can be shown to
achieve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

Metric = Literal["ip", "l2", "cosine"]

# Chosen so a float32 score block stays comfortably inside L3 on typical
# machines: 4096 queries x 65536 vectors x 4 bytes is ~1 GiB, so we go smaller.
_QUERY_BLOCK = 512
_CORPUS_BLOCK = 65_536


@dataclass(frozen=True)
class GroundTruth:
    """Exact top-k neighbours for a query set.

    Attributes:
        indices: Row indices into the corpus, shape (n_queries, k), best first.
        scores: Similarity/distance for each neighbour, shape (n_queries, k).
        metric: The metric these neighbours were computed under.
    """

    indices: np.ndarray
    scores: np.ndarray
    metric: Metric

    def __post_init__(self) -> None:
        if self.indices.shape != self.scores.shape:
            raise ValueError(
                f"indices {self.indices.shape} and scores {self.scores.shape} "
                "must have the same shape"
            )

    @property
    def k(self) -> int:
        return int(self.indices.shape[1])


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows, leaving zero rows untouched rather than producing NaN."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # A zero vector has no direction; dividing by 1.0 keeps it zero instead of NaN.
    np.maximum(norms, np.finfo(np.float32).tiny, out=norms)
    return (vectors / norms).astype(np.float32, copy=False)


def _as_float32_2d(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return np.ascontiguousarray(array, dtype=np.float32)


def compute(
    corpus: np.ndarray,
    queries: np.ndarray,
    k: int,
    metric: Metric = "ip",
    allowlist: np.ndarray | None = None,
) -> GroundTruth:
    """Compute exact top-k neighbours by exhaustive search.

    Args:
        corpus: Corpus vectors, shape (n, dim).
        queries: Query vectors, shape (n_queries, dim).
        k: Number of neighbours to return. Clamped to the number of candidates.
        metric: "ip" (inner product), "cosine" (normalised inner product), or
            "l2" (squared Euclidean; smaller is better).
        allowlist: Optional corpus row indices to restrict the search to. This is
            what makes filtered-recall measurement possible: the exact answer
            *within the allowed set* is the only fair reference for a filtered
            query.

    Returns:
        GroundTruth whose indices are always absolute corpus row indices, even
        when an allowlist was applied.
    """
    corpus = _as_float32_2d(corpus, "corpus")
    queries = _as_float32_2d(queries, "queries")
    if corpus.shape[1] != queries.shape[1]:
        raise ValueError(
            f"dimension mismatch: corpus dim {corpus.shape[1]} != query dim {queries.shape[1]}"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    if metric == "cosine":
        corpus = normalize(corpus)
        queries = normalize(queries)
        metric = "ip"

    if allowlist is not None:
        allowlist = np.unique(np.ascontiguousarray(allowlist, dtype=np.int64))
        if allowlist.size and (allowlist[0] < 0 or allowlist[-1] >= corpus.shape[0]):
            raise ValueError("allowlist contains out-of-range corpus indices")
        candidates = allowlist
    else:
        candidates = None

    n_candidates = corpus.shape[0] if candidates is None else int(candidates.size)
    if n_candidates == 0:
        raise ValueError("no candidates to search: allowlist is empty")

    # Returning fewer than k is correct, not a failure: it is exactly the
    # shortfall behaviour we hold approximate backends to.
    effective_k = min(k, n_candidates)
    if effective_k < k:
        logger.info(
            "k reduced from %d to %d: only %d candidates available", k, effective_k, n_candidates
        )

    higher_is_better = metric == "ip"
    n_queries = queries.shape[0]
    best_scores = np.empty((n_queries, effective_k), dtype=np.float32)
    best_indices = np.empty((n_queries, effective_k), dtype=np.int64)

    for q_start in range(0, n_queries, _QUERY_BLOCK):
        q_block = queries[q_start : q_start + _QUERY_BLOCK]
        acc_scores: np.ndarray | None = None
        acc_indices: np.ndarray | None = None

        for c_start in range(0, n_candidates, _CORPUS_BLOCK):
            if candidates is None:
                rows = corpus[c_start : c_start + _CORPUS_BLOCK]
                absolute = np.arange(
                    c_start, min(c_start + _CORPUS_BLOCK, n_candidates), dtype=np.int64
                )
            else:
                absolute = candidates[c_start : c_start + _CORPUS_BLOCK]
                rows = corpus[absolute]

            scores = _score(q_block, rows, metric)
            take = min(effective_k, scores.shape[1])
            part = _top_k(scores, take, higher_is_better)
            part_scores = np.take_along_axis(scores, part, axis=1)
            part_indices = absolute[part]

            if acc_scores is None:
                acc_scores, acc_indices = part_scores, part_indices
            else:
                acc_scores = np.concatenate([acc_scores, part_scores], axis=1)
                acc_indices = np.concatenate([acc_indices, part_indices], axis=1)
                merged = _top_k(acc_scores, effective_k, higher_is_better)
                acc_scores = np.take_along_axis(acc_scores, merged, axis=1)
                acc_indices = np.take_along_axis(acc_indices, merged, axis=1)

        assert acc_scores is not None and acc_indices is not None
        order = _sort_order(acc_scores, higher_is_better)
        q_end = q_start + q_block.shape[0]
        ranked_scores = np.take_along_axis(acc_scores, order, axis=1)
        ranked_indices = np.take_along_axis(acc_indices, order, axis=1)
        best_scores[q_start:q_end] = ranked_scores[:, :effective_k]
        best_indices[q_start:q_end] = ranked_indices[:, :effective_k]

    return GroundTruth(indices=best_indices, scores=best_scores, metric=metric)


def _score(queries: np.ndarray, corpus: np.ndarray, metric: Metric) -> np.ndarray:
    if metric == "ip":
        return queries @ corpus.T
    if metric == "l2":
        # ||q - c||^2 expanded so the bulk of the work is one GEMM.
        q_sq = np.einsum("ij,ij->i", queries, queries)[:, None]
        c_sq = np.einsum("ij,ij->i", corpus, corpus)[None, :]
        dist = q_sq + c_sq - 2.0 * (queries @ corpus.T)
        return np.maximum(dist, 0.0, out=dist)
    raise ValueError(f"unsupported metric: {metric!r}")


def _top_k(scores: np.ndarray, k: int, higher_is_better: bool) -> np.ndarray:
    """Unordered indices of the k best columns per row."""
    if k >= scores.shape[1]:
        return np.tile(np.arange(scores.shape[1], dtype=np.int64), (scores.shape[0], 1))
    ordered = scores if not higher_is_better else -scores
    return np.argpartition(ordered, kth=k - 1, axis=1)[:, :k]


def _sort_order(scores: np.ndarray, higher_is_better: bool) -> np.ndarray:
    ordered = scores if not higher_is_better else -scores
    return np.argsort(ordered, axis=1, kind="stable")
