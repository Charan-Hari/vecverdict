"""turbovec adapter.

turbovec is a flat quantised index: every query scans every vector with SIMD
kernels, and an allowlist is honoured inside the kernel rather than applied to
results afterwards. It therefore has no graph to disconnect, which is why it
serves as the filtered-search control in the sweep.
"""

from __future__ import annotations

import numpy as np

from ..groundtruth import normalize
from .base import Backend, BackendInfo, empty_results

# turbovec stores 2-bit or 4-bit codes; 4-bit is the higher-recall default.
_DEFAULT_BIT_WIDTH = 4


class TurbovecBackend(Backend):
    """IdMapIndex with native allowlist filtering."""

    def __init__(self, dim: int, metric: str = "ip", bit_width: int = _DEFAULT_BIT_WIDTH) -> None:
        super().__init__(dim, metric)
        if bit_width not in (2, 4):
            raise ValueError(f"bit_width must be 2 or 4, got {bit_width}")
        self.bit_width = bit_width
        self._index = None
        self._version = "unknown"

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            name=f"turbovec-{self.bit_width}bit",
            version=self._version,
            index_type=f"flat-tq{self.bit_width}",
            is_graph_based=False,
        )

    def build(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        import turbovec

        vectors, ids = self._validate_build(vectors, ids)
        self._version = getattr(turbovec, "__version__", "unknown")
        if self.metric == "cosine":
            vectors = normalize(vectors)

        self._index = turbovec.IdMapIndex(dim=self.dim, bit_width=self.bit_width)
        self._index.add_with_ids(vectors, ids.astype(np.uint64))
        self._size = int(vectors.shape[0])

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        queries = self._prepare(queries, k)
        _, ids = self._index.search(queries, k=k)
        return self._normalise(ids, n_queries=queries.shape[0], k=k)

    def search_filtered(self, queries: np.ndarray, k: int, allowed: np.ndarray) -> np.ndarray:
        queries = self._prepare(queries, k)
        allowed = np.unique(np.ascontiguousarray(allowed, dtype=np.uint64))
        if allowed.size == 0:
            return empty_results(queries.shape[0], k)
        _, ids = self._index.search(queries, k=k, allowlist=allowed)
        return self._normalise(ids, n_queries=queries.shape[0], k=k)

    def memory_bytes(self) -> int:
        # Quantised codes dominate: one code per dimension per vector, plus the
        # uint64 id map entry that makes ids survive deletion.
        code_bytes = self._size * self.dim * self.bit_width // 8
        id_map_bytes = self._size * 8
        return code_bytes + id_map_bytes

    def _prepare(self, queries: np.ndarray, k: int) -> np.ndarray:
        queries = self._validate_queries(queries, k)
        return normalize(queries) if self.metric == "cosine" else queries

    @staticmethod
    def _normalise(ids: np.ndarray, n_queries: int, k: int) -> np.ndarray:
        """Widen turbovec's exactly-sized output to a MISSING-padded (n, k) array.

        turbovec returns min(k, n_allowed) columns rather than padding, so a
        narrow allowlist yields a narrow array. Widening here keeps the shortfall
        visible in the metrics instead of hidden in an array shape.
        """
        out = empty_results(n_queries, k)
        ids = np.atleast_2d(np.asarray(ids, dtype=np.int64))
        if ids.size:
            out[:, : ids.shape[1]] = ids[:, :k]
        return out
