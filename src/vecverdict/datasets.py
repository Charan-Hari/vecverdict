"""Dataset loading: synthetic for tests, real embeddings for results.

Three tiers, each for a different purpose. Synthetic clustered vectors make the
test suite fast and deterministic. Small real corpora keep the development loop
under a few seconds. A large pre-computed embedding set provides results worth
publishing without anyone paying an embedding API.

Downloaded data is cached outside the repository and is never committed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Cache location is overridable so a large download can live on another drive.
_CACHE_ENV = "VECVERDICT_CACHE"
_DEFAULT_CACHE = Path.home() / ".cache" / "vecverdict"


@dataclass(frozen=True)
class Dataset:
    """A corpus with held-out queries.

    Attributes:
        name: Identifier recorded in results.
        vectors: Corpus, shape (n, dim), float32.
        ids: External ids, shape (n,).
        queries: Query vectors, shape (n_queries, dim), float32.
        source: Human-readable provenance for the writeup.
    """

    name: str
    vectors: np.ndarray
    ids: np.ndarray
    queries: np.ndarray
    source: str

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])


def cache_dir() -> Path:
    path = Path(os.environ.get(_CACHE_ENV, _DEFAULT_CACHE))
    path.mkdir(parents=True, exist_ok=True)
    return path


def synthetic(
    n: int = 20_000,
    dim: int = 128,
    n_queries: int = 50,
    n_clusters: int = 200,
    spread: float = 0.4,
    seed: int = 0,
) -> Dataset:
    """Clustered Gaussian vectors.

    Clustering is not cosmetic. Graph indexes connect nearby vectors, so a
    filter's effect on reachability depends on the corpus having neighbourhood
    structure. Uniform random vectors have none and would understate the
    failure this tool measures.

    Queries are drawn near cluster centres, mimicking real queries that land in
    populated regions rather than empty space.
    """
    rng = np.random.default_rng(seed)
    centers = (rng.standard_normal((n_clusters, dim)) * 3).astype(np.float32)

    per_cluster = np.full(n_clusters, n // n_clusters, dtype=np.int64)
    per_cluster[: n % n_clusters] += 1
    vectors = np.repeat(centers, per_cluster, axis=0)
    vectors += rng.standard_normal(vectors.shape).astype(np.float32) * spread

    query_centers = centers[rng.choice(n_clusters, size=n_queries, replace=n_queries > n_clusters)]
    queries = query_centers + rng.standard_normal(query_centers.shape).astype(np.float32) * spread

    return Dataset(
        name=f"synthetic-{n}x{dim}",
        vectors=np.ascontiguousarray(vectors, dtype=np.float32),
        ids=np.arange(vectors.shape[0], dtype=np.int64),
        queries=np.ascontiguousarray(queries, dtype=np.float32),
        source="generated locally; clustered Gaussian",
    )


def from_numpy(
    path: str | Path,
    n_queries: int = 50,
    query_path: str | Path | None = None,
    seed: int = 0,
) -> Dataset:
    """Load a user's own embeddings from a .npy or .npz file.

    This is the path that answers "does this happen to my data?". Queries are
    held out from the corpus when a separate query file is not supplied.

    Args:
        path: .npy array of shape (n, dim), or .npz containing "vectors".
        n_queries: Queries to hold out when query_path is None.
        query_path: Optional separate query array.
        seed: Seed for held-out query selection.
    """
    path = Path(path)
    vectors = _load_array(path)
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D array in {path.name}, got shape {vectors.shape}")

    if query_path is not None:
        queries = _load_array(Path(query_path))
        if queries.shape[1] != vectors.shape[1]:
            raise ValueError(
                f"query dim {queries.shape[1]} does not match corpus dim {vectors.shape[1]}"
            )
    else:
        if vectors.shape[0] <= n_queries:
            raise ValueError(
                f"corpus of {vectors.shape[0]} vectors is too small to hold out {n_queries} queries"
            )
        rng = np.random.default_rng(seed)
        picked = rng.choice(vectors.shape[0], size=n_queries, replace=False)
        # Queries stay in the corpus: a real system searches over everything it
        # holds, and removing them would make every query artificially hard.
        queries = vectors[picked]

    return Dataset(
        name=path.stem,
        vectors=np.ascontiguousarray(vectors, dtype=np.float32),
        ids=np.arange(vectors.shape[0], dtype=np.int64),
        queries=np.ascontiguousarray(queries, dtype=np.float32),
        source=f"user file {path.name}",
    )


def dbpedia_openai(limit: int = 100_000, n_queries: int = 50, seed: int = 0) -> Dataset:
    """Pre-computed OpenAI text-embedding-ada-002 vectors (1536-d) from DBpedia.

    Chosen because the embeddings already exist publicly: results are
    reproducible by anyone, at a dimension vector databases are actually
    deployed at, without an embedding API bill.

    Requires: pip install 'vecverdict[data]'
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "loading public datasets requires: pip install 'vecverdict[data]'"
        ) from exc

    logger.info("loading dbpedia-entities-openai-1M (limit=%d)", limit)
    stream = load_dataset(
        "KShivendu/dbpedia-entities-openai-1M",
        split="train",
        streaming=True,
        cache_dir=str(cache_dir()),
    )

    rows = []
    for i, row in enumerate(stream):
        if i >= limit:
            break
        rows.append(row["openai"])
    vectors = np.asarray(rows, dtype=np.float32)

    rng = np.random.default_rng(seed)
    picked = rng.choice(vectors.shape[0], size=min(n_queries, vectors.shape[0]), replace=False)

    return Dataset(
        name=f"dbpedia-openai-{vectors.shape[0]}",
        vectors=np.ascontiguousarray(vectors),
        ids=np.arange(vectors.shape[0], dtype=np.int64),
        queries=np.ascontiguousarray(vectors[picked]),
        source="KShivendu/dbpedia-entities-openai-1M (text-embedding-ada-002, 1536-d)",
    )


def load(spec: str, limit: int = 100_000, n_queries: int = 50, seed: int = 0) -> Dataset:
    """Resolve a dataset specification.

    Args:
        spec: "synthetic", "dbpedia", or a path to a .npy/.npz file.
        limit: Maximum vectors to load for downloadable datasets.
        n_queries: Number of queries.
        seed: Seed for sampling.
    """
    if spec == "synthetic":
        return synthetic(n=limit, n_queries=n_queries, seed=seed)
    if spec == "dbpedia":
        return dbpedia_openai(limit=limit, n_queries=n_queries, seed=seed)
    if Path(spec).exists():
        return from_numpy(spec, n_queries=n_queries, seed=seed)
    raise ValueError(
        f"unknown dataset {spec!r}; expected 'synthetic', 'dbpedia', or a path to a .npy file"
    )


def _load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    if path.suffix == ".npz":
        with np.load(path) as bundle:
            key = "vectors" if "vectors" in bundle else next(iter(bundle.keys()))
            return np.ascontiguousarray(bundle[key], dtype=np.float32)
    return np.ascontiguousarray(np.load(path), dtype=np.float32)
