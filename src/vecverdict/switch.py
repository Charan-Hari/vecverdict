"""The switch verdict: should these vectors move to a compressed index?

Answering honestly requires refusing some questions. Vectors reconstructed from
a product-quantised FAISS index have already lost information; re-quantising
them measures the damage of two codecs stacked, not the damage the user would
actually incur. Any recall figure produced that way understates the target
index and is not reported.

The memory figure is arithmetic and certain. The recall figure is the one that
requires measurement, and it is measured on the user's own vectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import groundtruth, metrics
from .backends import create

logger = logging.getLogger(__name__)

# FAISS index classes that store quantised codes rather than full vectors.
# Reconstructing from these loses information before this tool sees it.
_LOSSY_MARKERS = ("PQ", "SQ", "LSH", "RQ", "LocalSearch", "Residual")

# Below this recall, compression is doing visible damage to result quality.
_RECALL_FLOOR = 0.90
# Savings smaller than this rarely justify a migration.
_SAVINGS_FLOOR = 2.0


@dataclass
class SwitchReport:
    """The verdict and the evidence behind it."""

    source: str
    n_vectors: int
    dim: int
    source_bytes: int
    target_bytes: int
    bit_width: int
    k: int
    recall: float | None = None
    recall_at_1: float | None = None
    verdict: str = ""
    reasons: list[str] = field(default_factory=list)
    refused: bool = False

    @property
    def savings_ratio(self) -> float:
        return self.source_bytes / self.target_bytes if self.target_bytes else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "n_vectors": self.n_vectors,
            "dim": self.dim,
            "source_mb": round(self.source_bytes / 1024**2, 1),
            "target_mb": round(self.target_bytes / 1024**2, 1),
            "savings_ratio": round(self.savings_ratio, 2),
            "bit_width": self.bit_width,
            "k": self.k,
            "recall": round(self.recall, 4) if self.recall is not None else None,
            "recall_at_1": round(self.recall_at_1, 4) if self.recall_at_1 is not None else None,
            "verdict": self.verdict,
            "reasons": self.reasons,
            "refused": self.refused,
        }


def inspect_faiss_index(path: str | Path) -> tuple[np.ndarray, str]:
    """Read vectors out of a FAISS index, refusing lossy sources.

    Args:
        path: Path to a FAISS index file.

    Returns:
        The reconstructed vectors and the index class name.

    Raises:
        ValueError: If the index stores quantised codes, since reconstructing
            and re-quantising would stack two lossy codecs.
    """
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "reading FAISS indexes requires: pip install 'vecverdict[faiss]'"
        ) from exc

    index = faiss.read_index(str(path))
    class_name = type(index).__name__

    inner = index
    while hasattr(inner, "index") and inner.index is not None:
        inner = inner.index
        class_name = f"{class_name}->{type(inner).__name__}"

    if any(marker in class_name for marker in _LOSSY_MARKERS):
        raise ValueError(
            f"{class_name} stores quantised codes, not full vectors. Reconstructing "
            "them and re-quantising would measure two lossy codecs stacked, so the "
            "recall reported would be worse than a real migration and would misstate "
            "the target index. Supply the original float32 vectors with --vectors "
            "instead, or pass --allow-lossy to override."
        )

    total = index.ntotal
    if total == 0:
        raise ValueError("index is empty")

    vectors = np.empty((total, index.d), dtype=np.float32)
    inner.reconstruct_n(0, total, vectors)
    return vectors, class_name


def evaluate(
    vectors: np.ndarray,
    queries: np.ndarray | None = None,
    k: int = 10,
    bit_width: int = 4,
    metric: str = "ip",
    source_name: str = "float32 vectors",
    n_queries: int = 100,
    seed: int = 0,
) -> SwitchReport:
    """Measure what compressing these vectors would cost.

    Args:
        vectors: Corpus, shape (n, dim), float32.
        queries: Queries to evaluate with. Sampled from the corpus if omitted.
        k: Results per query.
        bit_width: Target quantisation width, 2 or 4.
        metric: Similarity metric.
        source_name: Label for the source index in the report.
        n_queries: Queries to sample when none are supplied.
        seed: Seed for query sampling.

    Returns:
        A SwitchReport whose verdict may recommend against switching.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
    n, dim = vectors.shape

    if queries is None:
        rng = np.random.default_rng(seed)
        picked = rng.choice(n, size=min(n_queries, n), replace=False)
        queries = vectors[picked]
    queries = np.ascontiguousarray(queries, dtype=np.float32)

    report = SwitchReport(
        source=source_name,
        n_vectors=n,
        dim=dim,
        source_bytes=n * dim * 4,
        target_bytes=n * dim * bit_width // 8 + n * 8,
        bit_width=bit_width,
        k=k,
    )

    ids = np.arange(n, dtype=np.int64)
    truth = groundtruth.compute(vectors, queries, k=k, metric=metric)

    with create("turbovec", dim=dim, metric=metric, bit_width=bit_width) as backend:
        backend.build(vectors, ids)
        retrieved = backend.search(queries, k=k)

    quality = metrics.evaluate(retrieved, ids[truth.indices], k=k)
    report.recall = quality.recall_at_k
    report.recall_at_1 = quality.recall_at_1
    _decide(report)
    return report


def _decide(report: SwitchReport) -> None:
    """Form a verdict, which may be to stay put."""
    recall = report.recall if report.recall is not None else 0.0
    savings = report.savings_ratio
    reasons: list[str] = []

    if recall >= _RECALL_FLOOR and savings >= _SAVINGS_FLOOR:
        verdict = "switch"
        reasons.append(
            f"{savings:.1f}x less memory at recall@{report.k} {recall:.3f} "
            f"({recall * 100:.1f}% of exact results retained)"
        )
        if report.source_bytes < 512 * 1024**2:
            reasons.append(
                f"the index is only {report.source_bytes / 1024**2:.0f} MB, so the "
                "absolute saving may not justify the migration"
            )
    elif recall < _RECALL_FLOOR:
        verdict = "stay"
        reasons.append(
            f"recall@{report.k} of {recall:.3f} is below {_RECALL_FLOOR:.2f}: "
            f"compression loses {(1 - recall) * 100:.1f}% of correct results on these vectors"
        )
        reasons.append(
            "embeddings whose information is spread evenly across dimensions "
            "quantise poorly; a higher bit width may recover this"
        )
    else:
        verdict = "stay"
        reasons.append(f"{savings:.1f}x saving does not justify a migration")

    report.verdict = verdict
    report.reasons = reasons
