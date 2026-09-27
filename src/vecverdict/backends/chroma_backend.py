"""Chroma adapter.

Chroma is an HNSW index behind a document store. It reports short result sets by
returning a shorter list rather than padding, so normalisation here widens the
list back to k — the missing entries are the measurement, and must survive.

Chroma addresses vectors by string id; the int64 external ids used throughout
this package are mapped to decimal strings and back.
"""

from __future__ import annotations

import contextlib

import numpy as np

from ..groundtruth import normalize
from .base import Backend, BackendInfo, empty_results

# Chroma rejects collection names shorter than three characters.
_COLLECTION = "vecverdict"

# Chroma names inner-product space "ip" and Euclidean "l2"; cosine is handled by
# normalising vectors and using inner product, matching the other backends.
_SPACES = {"ip": "ip", "cosine": "ip", "l2": "l2"}


class ChromaBackend(Backend):
    """Ephemeral in-memory Chroma collection using its native id filter."""

    def __init__(self, dim: int, metric: str = "ip", batch_size: int = 5000) -> None:
        super().__init__(dim, metric)
        self.batch_size = batch_size
        self._client = None
        self._collection = None
        self._version = "unknown"

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            name="chroma",
            version=self._version,
            index_type="HNSW",
            is_graph_based=True,
        )

    def build(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        import chromadb

        vectors, ids = self._validate_build(vectors, ids)
        self._version = getattr(chromadb, "__version__", "unknown")
        if self.metric == "cosine":
            vectors = normalize(vectors)

        self._client = chromadb.EphemeralClient()
        # A fresh measurement must not inherit a previous run's vectors.
        with contextlib.suppress(Exception):
            self._client.delete_collection(_COLLECTION)
        self._collection = self._client.create_collection(
            _COLLECTION,
            configuration={"hnsw": {"space": _SPACES[self.metric]}},
        )

        # Chroma enforces a per-call payload limit, so large corpora are chunked.
        for start in range(0, vectors.shape[0], self.batch_size):
            stop = start + self.batch_size
            self._collection.add(
                ids=[str(i) for i in ids[start:stop]],
                embeddings=vectors[start:stop].tolist(),
            )
        self._size = int(vectors.shape[0])

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        queries = self._prepare(queries, k)
        result = self._collection.query(query_embeddings=queries.tolist(), n_results=k)
        return self._normalise(result["ids"], queries.shape[0], k)

    def search_filtered(self, queries: np.ndarray, k: int, allowed: np.ndarray) -> np.ndarray:
        queries = self._prepare(queries, k)
        allowed = np.unique(np.ascontiguousarray(allowed, dtype=np.int64))
        if allowed.size == 0:
            return empty_results(queries.shape[0], k)

        result = self._collection.query(
            query_embeddings=queries.tolist(),
            n_results=k,
            ids=[str(i) for i in allowed],
        )
        return self._normalise(result["ids"], queries.shape[0], k)

    def memory_bytes(self) -> int:
        # HNSW over float32 vectors; M defaults to 16 in Chroma, giving 2*M
        # links of 4 bytes each at the base layer.
        vector_bytes = self._size * self.dim * 4
        link_bytes = self._size * 16 * 2 * 4
        return vector_bytes + link_bytes

    def close(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.delete_collection(_COLLECTION)
        self._collection = None
        self._client = None

    def _prepare(self, queries: np.ndarray, k: int) -> np.ndarray:
        queries = self._validate_queries(queries, k)
        return normalize(queries) if self.metric == "cosine" else queries

    @staticmethod
    def _normalise(rows: list[list[str]], n_queries: int, k: int) -> np.ndarray:
        """Widen Chroma's truncated per-query lists to a MISSING-padded array."""
        out = empty_results(n_queries, k)
        for i, row in enumerate(rows[:n_queries]):
            if not row:
                continue
            values = np.array([int(x) for x in row[:k]], dtype=np.int64)
            out[i, : values.size] = values
        return out
