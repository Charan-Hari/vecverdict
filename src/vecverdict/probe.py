"""Export a single filtered query, result by result, for the demo site.

The sweep reports averages. An average of 0.26 results out of 10 is easy to
read past; a list of ten slots where nine are empty is not. This module
captures one query in that detail so the failure can be inspected rather than
summarised.

Everything written here is measured. Nothing is illustrative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import groundtruth
from .backends import Backend
from .environment import collect
from .metrics import MISSING
from .sweep import rows_for_ids

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

SCHEMA = "vecverdict/probe/1"


@dataclass(frozen=True)
class QueryProbe:
    """One backend's answer to one filtered query, slot by slot.

    Attributes:
        backend: Backend name.
        index_type: Index structure, e.g. "HNSW" or "flat-tq4".
        is_graph_based: Whether the index traverses a graph at search time.
        returned: Ids in rank order, MISSING for slots left empty.
        correct: Per slot, whether that id is in the exact top-k.
        truth: The exact top-k ids under the same filter.
        n_returned: Slots actually filled.
        n_correct: Filled slots holding a true top-k id.
    """

    backend: str
    index_type: str
    is_graph_based: bool
    returned: list[int]
    correct: list[bool]
    truth: list[int]
    n_returned: int
    n_correct: int

    def to_dict(self) -> dict[str, object]:
        """Plain dict for JSON."""
        return {
            "backend": self.backend,
            "index_type": self.index_type,
            "is_graph_based": self.is_graph_based,
            "returned": self.returned,
            "correct": self.correct,
            "truth": self.truth,
            "n_returned": self.n_returned,
            "n_correct": self.n_correct,
        }


@dataclass(frozen=True)
class ProbeResult:
    """Every backend's answer to the same filtered query.

    Attributes:
        dataset: Dataset label.
        n_vectors: Corpus size.
        dim: Vector dimension.
        k: Results requested.
        selectivity: Fraction of the corpus the filter allowed.
        n_allowed: Vectors the filter allowed.
        metric: Similarity metric.
        query_index: Which query this is, within the dataset's query set.
        probes: One entry per backend.
        machine: Sanitised machine metadata.
    """

    dataset: str
    n_vectors: int
    dim: int
    k: int
    selectivity: float
    n_allowed: int
    metric: str
    query_index: int
    probes: list[QueryProbe] = field(default_factory=list)
    machine: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Plain dict for JSON."""
        return {
            "schema": SCHEMA,
            "dataset": self.dataset,
            "n_vectors": self.n_vectors,
            "dim": self.dim,
            "k": self.k,
            "selectivity": self.selectivity,
            "n_allowed": self.n_allowed,
            "metric": self.metric,
            "query_index": self.query_index,
            "machine": self.machine,
            "probes": [probe.to_dict() for probe in self.probes],
        }

    def save(self, path: str | Path) -> Path:
        """Write JSON, creating parent directories."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target


def probe(
    backends: Sequence[Backend],
    vectors: np.ndarray,
    ids: np.ndarray,
    query: np.ndarray,
    allowed: np.ndarray,
    k: int = 10,
    metric: str = "ip",
    dataset: str = "unknown",
    query_index: int = 0,
    build: bool = True,
) -> ProbeResult:
    """Ask every backend one filtered query and record each slot.

    Args:
        backends: Constructed backends. Built here unless build=False.
        vectors: Corpus, shape (n, dim), float32.
        ids: External ids, shape (n,).
        query: One query vector, shape (dim,) or (1, dim).
        allowed: Ids the filter permits.
        k: Results requested.
        metric: Metric shared by ground truth and backends.
        dataset: Label recorded in the output.
        query_index: Index of this query, recorded for reproducibility.
        build: Set False when backends already hold the corpus.

    Returns:
        A ProbeResult holding one QueryProbe per backend.

    Raises:
        ValueError: If the allowlist is smaller than k, which would make a
            shortfall arithmetic rather than a property of the index.
    """
    if allowed.size < k:
        raise ValueError(
            f"allowlist of {allowed.size} is smaller than k={k}; "
            "a shortfall would then be unavoidable and prove nothing"
        )

    query_2d = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)

    # Ground truth works in corpus row indices; the rest of this module works
    # in external ids. Translate the same way the sweep does, so probe results
    # and sweep results describe the same neighbours.
    allowed_rows = rows_for_ids(ids, allowed)
    truth = groundtruth.compute(
        vectors, query_2d, k=k, metric=metric, allowlist=allowed_rows
    )
    truth_ids = [int(value) for value in ids[truth.indices][0]]
    truth_set = set(truth_ids)

    probes: list[QueryProbe] = []
    for backend in backends:
        if build:
            backend.build(vectors, ids)

        found = backend.search_filtered(query_2d, k, allowed)[0]
        returned = [int(value) for value in found]
        correct = [value != MISSING and value in truth_set for value in returned]

        info = backend.info
        probes.append(
            QueryProbe(
                backend=info.name,
                index_type=info.index_type,
                is_graph_based=info.is_graph_based,
                returned=returned,
                correct=correct,
                truth=truth_ids,
                n_returned=sum(value != MISSING for value in returned),
                n_correct=sum(correct),
            )
        )

    return ProbeResult(
        dataset=dataset,
        n_vectors=int(vectors.shape[0]),
        dim=int(vectors.shape[1]),
        k=k,
        selectivity=float(allowed.size / ids.size),
        n_allowed=int(allowed.size),
        metric=metric,
        query_index=query_index,
        probes=probes,
        machine=collect().to_dict(),
    )
