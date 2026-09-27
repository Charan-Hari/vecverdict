"""The backend interface every vector index is measured through.

Each library reports "no result here" differently: FAISS pads short result sets
with -1 to a full k columns, Chroma returns a shorter list, turbovec returns an
array whose width is the number of results it actually found. Comparing them
fairly requires one representation, so every adapter normalises onto a
MISSING-padded (n_queries, k) array of external ids.

Adapters must never invent a result to fill a slot. A backend that finds four
neighbours reports four and leaves the rest MISSING; that gap is the measurement.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from ..metrics import MISSING

# Metric names are normalised across libraries that spell them differently.
Metric = str


@dataclass(frozen=True)
class BackendInfo:
    """Static description of a backend for reporting.

    Attributes:
        name: Short identifier used in results and charts.
        version: Installed library version.
        index_type: The concrete index construction used, e.g. "HNSW" or
            "flat-4bit". Reported because the failure modes being measured are
            properties of the index structure, not of the vendor.
        is_graph_based: True for navigable-graph indexes (HNSW and relatives),
            whose reachability under selective filters is the central subject of
            the filter sweep.
    """

    name: str
    version: str
    index_type: str
    is_graph_based: bool


class Backend(abc.ABC):
    """A vector index under measurement.

    Implementations are constructed cheaply and do all real work in build().
    """

    def __init__(self, dim: int, metric: Metric = "ip") -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if metric not in ("ip", "l2", "cosine"):
            raise ValueError(f"unsupported metric: {metric!r}")
        self.dim = dim
        self.metric = metric
        self._size = 0

    @property
    def size(self) -> int:
        """Number of vectors currently indexed."""
        return self._size

    @property
    @abc.abstractmethod
    def info(self) -> BackendInfo:
        """Static description of this backend."""

    @abc.abstractmethod
    def build(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Index `vectors` under the external `ids`.

        Args:
            vectors: Corpus, shape (n, dim), float32.
            ids: External int64 ids, shape (n,). These are the ids returned by
                search and referenced by allowlists.
        """

    @abc.abstractmethod
    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Return the top-k external ids per query.

        Returns:
            Array of shape (n_queries, k), MISSING where no result was produced.
        """

    @abc.abstractmethod
    def search_filtered(self, queries: np.ndarray, k: int, allowed: np.ndarray) -> np.ndarray:
        """Return the top-k external ids restricted to `allowed`.

        Implementations must not post-filter unrestricted results: that would
        measure this tool's filtering rather than the backend's. Each backend
        uses its own native filtering path.

        Args:
            queries: Query vectors, shape (n_queries, dim), float32.
            k: Number of results requested per query.
            allowed: External ids the search may return, shape (n_allowed,).

        Returns:
            Array of shape (n_queries, k), MISSING-padded.
        """

    @abc.abstractmethod
    def memory_bytes(self) -> int:
        """Resident size of the index payload in bytes.

        Measures the vector payload the index holds, excluding interpreter and
        allocator overhead, so backends remain comparable.
        """

    def close(self) -> None:
        """Release any resources held by the backend.

        Not abstract: most backends hold nothing that needs releasing, and
        forcing every adapter to define an empty method adds noise.
        """

    def __enter__(self) -> Backend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _validate_queries(self, queries: np.ndarray, k: int) -> np.ndarray:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        queries = np.atleast_2d(np.ascontiguousarray(queries, dtype=np.float32))
        if queries.shape[1] != self.dim:
            raise ValueError(
                f"query dim {queries.shape[1]} does not match index dim {self.dim}"
            )
        if self._size == 0:
            raise RuntimeError("search called before build")
        return queries

    def _validate_build(
        self, vectors: np.ndarray, ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"vectors must have shape (n, {self.dim}), got {vectors.shape}"
            )
        ids = np.ascontiguousarray(ids, dtype=np.int64).ravel()
        if ids.shape[0] != vectors.shape[0]:
            raise ValueError(
                f"ids length {ids.shape[0]} does not match vector count {vectors.shape[0]}"
            )
        if np.unique(ids).size != ids.size:
            raise ValueError("ids must be unique")
        if ids.min() < 0:
            raise ValueError(f"ids must be non-negative, got minimum {ids.min()}")
        return vectors, ids


def empty_results(n_queries: int, k: int) -> np.ndarray:
    """An all-MISSING result array, the starting point for adapter normalisation."""
    return np.full((n_queries, k), MISSING, dtype=np.int64)
