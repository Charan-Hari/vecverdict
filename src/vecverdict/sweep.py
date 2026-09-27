"""The selectivity sweep: how retrieval degrades as a filter narrows.

A filter that allows most of the corpus is easy for any index. The interesting
region is the tail — allowing 1%, 0.1%, 0.01% of vectors — where graph-based
indexes can lose reachability to the allowed set and return fewer results than
requested without reporting an error.

The sweep holds k fixed, varies only the fraction of the corpus allowed, and
compares every backend against exact ground truth computed over the same
allowlist. Allowlists are drawn from a seeded generator so a run is reproducible
and every backend sees byte-identical filters.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__, environment, groundtruth, metrics
from .backends import Backend

logger = logging.getLogger(__name__)

# Spans the region where filtered search is trivial down to where it fails.
DEFAULT_SELECTIVITIES: tuple[float, ...] = (1.0, 0.5, 0.1, 0.01, 0.001, 0.0001)


@dataclass(frozen=True)
class SweepPoint:
    """One backend measured at one selectivity."""

    backend: str
    index_type: str
    is_graph_based: bool
    selectivity: float
    n_allowed: int
    k: int
    quality: metrics.RetrievalQuality
    latency_ms: float
    memory_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "index_type": self.index_type,
            "is_graph_based": self.is_graph_based,
            "selectivity": self.selectivity,
            "n_allowed": self.n_allowed,
            "k": self.k,
            "latency_ms": round(self.latency_ms, 3),
            "memory_bytes": self.memory_bytes,
            **{
                key: (round(value, 4) if isinstance(value, float) else value)
                for key, value in self.quality.to_dict().items()
            },
        }


@dataclass
class SweepResult:
    """A complete sweep, ready to serialise or plot."""

    points: list[SweepPoint] = field(default_factory=list)
    dataset: str = "unknown"
    n_vectors: int = 0
    dim: int = 0
    n_queries: int = 0
    k: int = 0
    metric: str = "ip"
    seed: int = 0

    def to_dict(self) -> dict[str, object]:
        """Serialisable form, including non-identifying machine metadata."""
        return {
            "schema": "vecverdict/sweep/1",
            "vecverdict_version": __version__,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": self.dataset,
            "n_vectors": self.n_vectors,
            "dim": self.dim,
            "n_queries": self.n_queries,
            "k": self.k,
            "metric": self.metric,
            "seed": self.seed,
            "machine": environment.collect().to_dict(),
            "points": [point.to_dict() for point in self.points],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def backends(self) -> list[str]:
        """Backend names in first-seen order."""
        seen: dict[str, None] = {}
        for point in self.points:
            seen.setdefault(point.backend, None)
        return list(seen)

    def for_backend(self, name: str) -> list[SweepPoint]:
        """Points for one backend, ordered from widest to narrowest filter."""
        chosen = [point for point in self.points if point.backend == name]
        return sorted(chosen, key=lambda point: -point.selectivity)


def make_allowlist(
    ids: np.ndarray,
    selectivity: float,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the ids a filter permits.

    Args:
        ids: All external ids in the corpus.
        selectivity: Fraction of the corpus to allow, in (0, 1].
        k: Results requested per query. The allowlist is never smaller than k,
            so that a shortfall is always the index's doing rather than an
            arithmetic consequence of the filter.
        rng: Seeded generator, so every backend sees the same allowlist.

    Returns:
        Sorted array of allowed ids.
    """
    if not 0 < selectivity <= 1:
        raise ValueError(f"selectivity must be in (0, 1], got {selectivity}")
    n_allowed = min(max(k, round(ids.size * selectivity)), ids.size)
    return np.sort(rng.choice(ids, size=n_allowed, replace=False))


def run(
    backends: list[Backend],
    vectors: np.ndarray,
    ids: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    selectivities: tuple[float, ...] = DEFAULT_SELECTIVITIES,
    metric: str = "ip",
    dataset: str = "unknown",
    seed: int = 0,
    build: bool = True,
) -> SweepResult:
    """Measure every backend across every selectivity.

    Args:
        backends: Already-constructed backends. Built here unless build=False.
        vectors: Corpus, shape (n, dim), float32.
        ids: External ids, shape (n,).
        queries: Query vectors, shape (n_queries, dim).
        k: Results requested per query.
        selectivities: Corpus fractions to allow, each in (0, 1].
        metric: Similarity metric shared by ground truth and backends.
        dataset: Dataset name recorded in the result.
        seed: Seed for allowlist sampling.
        build: Whether to index `vectors` before measuring.

    Returns:
        SweepResult containing one point per backend per selectivity.
    """
    if not backends:
        raise ValueError("no backends to measure")
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64).ravel()

    if build:
        for backend in backends:
            logger.info("building %s over %d vectors", backend.info.name, vectors.shape[0])
            backend.build(vectors, ids)

    result = SweepResult(
        dataset=dataset,
        n_vectors=int(vectors.shape[0]),
        dim=int(vectors.shape[1]),
        n_queries=int(queries.shape[0]),
        k=k,
        metric=metric,
        seed=seed,
    )

    # Position of each id, so ground-truth row indices map back to external ids.
    for selectivity in selectivities:
        # Reseeded per selectivity so a backend list change cannot alter filters.
        rng = np.random.default_rng([seed, int(selectivity * 1e9)])
        allowed = make_allowlist(ids, selectivity, k, rng)
        allowed_rows = np.searchsorted(ids, allowed) if _is_sorted(ids) else _rows_for(ids, allowed)

        truth = groundtruth.compute(
            vectors, queries, k=k, metric=metric, allowlist=allowed_rows
        )
        truth_ids = ids[truth.indices]
        # Ground truth may be narrower than k; pad so shapes stay comparable.
        if truth_ids.shape[1] < k:
            padded = np.full((truth_ids.shape[0], k), metrics.MISSING, dtype=np.int64)
            padded[:, : truth_ids.shape[1]] = truth_ids
            truth_ids = padded

        for backend in backends:
            start = time.perf_counter()
            retrieved = backend.search_filtered(queries, k=k, allowed=allowed)
            elapsed_ms = (time.perf_counter() - start) * 1000 / max(queries.shape[0], 1)

            quality = metrics.evaluate(retrieved, truth_ids, k=k, n_allowed=allowed.size)
            info = backend.info
            result.points.append(
                SweepPoint(
                    backend=info.name,
                    index_type=info.index_type,
                    is_graph_based=info.is_graph_based,
                    selectivity=selectivity,
                    n_allowed=int(allowed.size),
                    k=k,
                    quality=quality,
                    latency_ms=elapsed_ms,
                    memory_bytes=backend.memory_bytes(),
                )
            )
            logger.info(
                "%s @ %.4f: returned %.1f/%d recall %.3f",
                info.name,
                selectivity,
                quality.mean_returned,
                k,
                quality.recall_at_k,
            )

    return result


def _is_sorted(ids: np.ndarray) -> bool:
    return bool(np.all(np.diff(ids) > 0))


def _rows_for(ids: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Row indices of `allowed` within unsorted `ids`."""
    order = np.argsort(ids)
    positions = np.searchsorted(ids[order], allowed)
    return order[positions]
