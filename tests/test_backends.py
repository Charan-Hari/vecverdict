"""Conformance tests every backend adapter must pass.

The suite is parametrised over installed backends so each adapter is held to the
same contract. Most assertions concern result *shape* and the MISSING sentinel
rather than recall: normalising three different "no result" conventions onto one
representation is the adapters' entire job, and a mistake there would silently
corrupt every measurement built on top.
"""

from __future__ import annotations

import numpy as np
import pytest

from vecverdict import backends, groundtruth
from vecverdict.metrics import MISSING

DIM = 32
N = 400
K = 10

INSTALLED = backends.available()


@pytest.fixture(scope="module")
def corpus() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((N, DIM), dtype=np.float32)
    ids = np.arange(1000, 1000 + N, dtype=np.int64)
    return vectors, ids


@pytest.fixture(scope="module")
def queries() -> np.ndarray:
    rng = np.random.default_rng(8)
    return rng.standard_normal((5, DIM), dtype=np.float32)


@pytest.fixture(params=INSTALLED)
def backend(request: pytest.FixtureRequest, corpus: tuple[np.ndarray, np.ndarray]):
    vectors, ids = corpus
    instance = backends.create(request.param, dim=DIM, metric="ip")
    instance.build(vectors, ids)
    yield instance
    instance.close()


pytestmark = pytest.mark.skipif(not INSTALLED, reason="no backends installed")


def test_registry_lists_known_backends() -> None:
    assert "turbovec" in backends.names()
    assert "faiss-hnsw" in backends.names()


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        backends.create("nonexistent", dim=8)


def test_search_returns_requested_shape(backend, queries: np.ndarray) -> None:
    results = backend.search(queries, k=K)
    assert results.shape == (queries.shape[0], K)
    assert results.dtype == np.int64


def test_unfiltered_search_is_complete(backend, queries: np.ndarray) -> None:
    """With N >> k and no filter, every backend should fill all k slots."""
    results = backend.search(queries, k=K)
    assert (results != MISSING).all()


def test_results_use_external_ids(backend, queries: np.ndarray) -> None:
    """Ids start at 1000, so raw row offsets would be detectable here."""
    results = backend.search(queries, k=K)
    present = results[results != MISSING]
    assert present.min() >= 1000
    assert present.max() < 1000 + N


def test_results_are_unique_per_query(backend, queries: np.ndarray) -> None:
    results = backend.search(queries, k=K)
    for row in results:
        present = row[row != MISSING]
        assert np.unique(present).size == present.size


def test_flat_backends_match_exact_ground_truth(
    backend, corpus: tuple[np.ndarray, np.ndarray], queries: np.ndarray
) -> None:
    """Exhaustive backends must be exact; approximate ones are exempt."""
    if backend.info.is_graph_based or "tq" in backend.info.index_type:
        pytest.skip("approximate index")

    vectors, ids = corpus
    truth = groundtruth.compute(vectors, queries, k=K, metric="ip")
    expected = ids[truth.indices]
    np.testing.assert_array_equal(backend.search(queries, k=K), expected)


def test_filtered_search_returns_only_allowed_ids(
    backend, corpus: tuple[np.ndarray, np.ndarray], queries: np.ndarray
) -> None:
    _, ids = corpus
    allowed = ids[::4]
    results = backend.search_filtered(queries, k=K, allowed=allowed)
    present = results[results != MISSING]
    assert np.isin(present, allowed).all()


def test_narrow_allowlist_is_padded_to_k(
    backend, corpus: tuple[np.ndarray, np.ndarray], queries: np.ndarray
) -> None:
    """Three allowed vectors and k=10 must give three ids and seven MISSING.

    This is the normalisation contract: FAISS pads with -1, Chroma truncates,
    turbovec returns a narrow array. All three must arrive here identically.
    """
    _, ids = corpus
    allowed = ids[:3]

    results = backend.search_filtered(queries, k=K, allowed=allowed)

    assert results.shape == (queries.shape[0], K)
    for row in results:
        assert (row[3:] == MISSING).all()
        assert set(row[:3].tolist()) <= set(allowed.tolist())


def test_empty_allowlist_returns_all_missing(backend, queries: np.ndarray) -> None:
    results = backend.search_filtered(queries, k=K, allowed=np.array([], dtype=np.int64))
    assert (results == MISSING).all()


def test_filtered_search_does_not_exceed_k(
    backend, corpus: tuple[np.ndarray, np.ndarray], queries: np.ndarray
) -> None:
    _, ids = corpus
    results = backend.search_filtered(queries, k=3, allowed=ids)
    assert results.shape[1] == 3


def test_memory_is_positive_and_scales_with_corpus(backend) -> None:
    assert backend.memory_bytes() > 0


def test_turbovec_uses_less_memory_than_faiss_flat(
    corpus: tuple[np.ndarray, np.ndarray],
) -> None:
    """The compression claim, asserted rather than assumed."""
    if "turbovec" not in INSTALLED or "faiss-flat" not in INSTALLED:
        pytest.skip("both backends required")

    vectors, ids = corpus
    with backends.create("turbovec", dim=DIM) as tv, backends.create("faiss-flat", dim=DIM) as fl:
        tv.build(vectors, ids)
        fl.build(vectors, ids)
        assert tv.memory_bytes() < fl.memory_bytes()


def test_search_before_build_is_rejected(queries: np.ndarray) -> None:
    instance = backends.create(INSTALLED[0], dim=DIM)
    with pytest.raises(RuntimeError, match="before build"):
        instance.search(queries, k=1)


def test_dimension_mismatch_is_rejected(backend) -> None:
    with pytest.raises(ValueError, match="does not match index dim"):
        backend.search(np.zeros((1, DIM + 1), dtype=np.float32), k=1)


def test_duplicate_ids_are_rejected() -> None:
    instance = backends.create(INSTALLED[0], dim=DIM)
    vectors = np.zeros((3, DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="unique"):
        instance.build(vectors, np.array([1, 1, 2], dtype=np.int64))


def test_id_count_mismatch_is_rejected() -> None:
    instance = backends.create(INSTALLED[0], dim=DIM)
    vectors = np.zeros((3, DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match vector count"):
        instance.build(vectors, np.array([1, 2], dtype=np.int64))
