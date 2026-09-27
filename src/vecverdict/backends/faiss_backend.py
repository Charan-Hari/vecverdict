"""FAISS adapters: an exact flat index and an HNSW graph index.

Two FAISS configurations are measured because they fail differently. The flat
index is exhaustive, so a filter can only reduce how many candidates exist. The
HNSW index navigates a graph, and a selective filter can leave much of that
graph unreachable — the failure the filter sweep exists to quantify.

FAISS signals "no result" by padding with -1, which coincides with this
package's MISSING sentinel; the mapping is made explicit rather than assumed.
"""

from __future__ import annotations

import numpy as np

from ..groundtruth import normalize
from ..metrics import MISSING
from .base import Backend, BackendInfo, empty_results

_FAISS_EMPTY = -1

# FAISS HNSW defaults. efSearch is the candidate-list size at query time and is
# the parameter that most directly governs filtered-search reachability.
_DEFAULT_M = 32
_DEFAULT_EF_CONSTRUCTION = 200
_DEFAULT_EF_SEARCH = 100


class _FaissBase(Backend):
    """Shared id-mapping and result normalisation for FAISS indexes."""

    def __init__(self, dim: int, metric: str = "ip") -> None:
        super().__init__(dim, metric)
        self._index = None
        self._version = "unknown"

    def build(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        import faiss

        vectors, ids = self._validate_build(vectors, ids)
        self._version = getattr(faiss, "__version__", "unknown")
        if self.metric == "cosine":
            vectors = normalize(vectors)

        self._index = faiss.IndexIDMap(self._make_index(faiss))
        self._index.add_with_ids(vectors, ids)
        self._size = int(vectors.shape[0])

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        queries = self._prepare(queries, k)
        _, ids = self._index.search(queries, k)
        return self._normalise(ids, queries.shape[0], k)

    def search_filtered(self, queries: np.ndarray, k: int, allowed: np.ndarray) -> np.ndarray:
        import faiss

        queries = self._prepare(queries, k)
        allowed = np.unique(np.ascontiguousarray(allowed, dtype=np.int64))
        if allowed.size == 0:
            return empty_results(queries.shape[0], k)

        # IDSelectorArray requires the array to outlive the search call, so the
        # reference is held locally rather than inlined into SearchParameters.
        selector = faiss.IDSelectorArray(allowed)
        params = self._search_params(faiss, selector)
        _, ids = self._index.search(queries, k, params=params)
        return self._normalise(ids, queries.shape[0], k)

    def memory_bytes(self) -> int:
        return self._payload_bytes() + self._size * 8

    def _prepare(self, queries: np.ndarray, k: int) -> np.ndarray:
        queries = self._validate_queries(queries, k)
        return normalize(queries) if self.metric == "cosine" else queries

    @staticmethod
    def _normalise(ids: np.ndarray, n_queries: int, k: int) -> np.ndarray:
        """Translate FAISS -1 padding onto MISSING without assuming they are equal."""
        out = empty_results(n_queries, k)
        ids = np.atleast_2d(np.asarray(ids, dtype=np.int64))
        width = min(k, ids.shape[1])
        block = ids[:, :width]
        out[:, :width] = np.where(block == _FAISS_EMPTY, MISSING, block)
        return out

    def _make_index(self, faiss):
        raise NotImplementedError

    def _search_params(self, faiss, selector):
        params = faiss.SearchParameters()
        params.sel = selector
        return params

    def _payload_bytes(self) -> int:
        raise NotImplementedError


class FaissFlatBackend(_FaissBase):
    """Exhaustive float32 index — exact, and the memory baseline to beat."""

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            name="faiss-flat",
            version=self._version,
            index_type="flat-f32",
            is_graph_based=False,
        )

    def _make_index(self, faiss):
        if self.metric == "l2":
            return faiss.IndexFlatL2(self.dim)
        return faiss.IndexFlatIP(self.dim)

    def _payload_bytes(self) -> int:
        return self._size * self.dim * 4


class FaissHnswBackend(_FaissBase):
    """HNSW graph index — the structure whose filtered reachability is in question."""

    def __init__(
        self,
        dim: int,
        metric: str = "ip",
        m: int = _DEFAULT_M,
        ef_construction: int = _DEFAULT_EF_CONSTRUCTION,
        ef_search: int = _DEFAULT_EF_SEARCH,
    ) -> None:
        super().__init__(dim, metric)
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            name=f"faiss-hnsw-m{self.m}-ef{self.ef_search}",
            version=self._version,
            index_type="HNSW",
            is_graph_based=True,
        )

    def _make_index(self, faiss):
        space = faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT
        index = faiss.IndexHNSWFlat(self.dim, self.m, space)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        return index

    def _search_params(self, faiss, selector):
        # HNSW needs its own parameter type so efSearch accompanies the selector;
        # a bare SearchParameters would silently drop the graph settings.
        params = faiss.SearchParametersHNSW()
        params.sel = selector
        params.efSearch = self.ef_search
        return params

    def _payload_bytes(self) -> int:
        # Full-precision vectors plus graph links: roughly 2*M neighbours at
        # level 0, each a 4-byte node id.
        vector_bytes = self._size * self.dim * 4
        link_bytes = self._size * self.m * 2 * 4
        return vector_bytes + link_bytes
