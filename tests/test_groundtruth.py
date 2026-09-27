"""Tests for exact ground-truth computation.

Fixtures are small enough that the correct answer is known by inspection rather
than by running the code under test, which is the only way a reference
implementation can be meaningfully verified.
"""

from __future__ import annotations

import numpy as np
import pytest

from vecverdict import groundtruth


@pytest.fixture
def orthogonal_corpus() -> np.ndarray:
    """Unit basis vectors: the nearest neighbour of e_i is unambiguously e_i."""
    return np.eye(5, dtype=np.float32)


def test_finds_exact_match_first(orthogonal_corpus: np.ndarray) -> None:
    query = orthogonal_corpus[2:3]
    result = groundtruth.compute(orthogonal_corpus, query, k=3, metric="ip")
    assert result.indices[0, 0] == 2
    assert result.scores[0, 0] == pytest.approx(1.0)


def test_scores_are_descending_for_inner_product() -> None:
    rng = np.random.default_rng(0)
    corpus = rng.standard_normal((200, 16), dtype=np.float32)
    queries = rng.standard_normal((5, 16), dtype=np.float32)
    result = groundtruth.compute(corpus, queries, k=10, metric="ip")
    for row in result.scores:
        assert np.all(np.diff(row) <= 1e-5)


def test_scores_are_ascending_for_l2() -> None:
    rng = np.random.default_rng(1)
    corpus = rng.standard_normal((200, 16), dtype=np.float32)
    queries = rng.standard_normal((5, 16), dtype=np.float32)
    result = groundtruth.compute(corpus, queries, k=10, metric="l2")
    for row in result.scores:
        assert np.all(np.diff(row) >= -1e-5)


def test_l2_finds_self_with_zero_distance(orthogonal_corpus: np.ndarray) -> None:
    result = groundtruth.compute(orthogonal_corpus, orthogonal_corpus[1:2], k=1, metric="l2")
    assert result.indices[0, 0] == 1
    assert result.scores[0, 0] == pytest.approx(0.0, abs=1e-5)


def test_blocking_matches_unblocked_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """A corpus spanning several blocks must give the same answer as one block."""
    rng = np.random.default_rng(2)
    corpus = rng.standard_normal((1000, 8), dtype=np.float32)
    queries = rng.standard_normal((20, 8), dtype=np.float32)

    expected = groundtruth.compute(corpus, queries, k=10)
    monkeypatch.setattr(groundtruth, "_CORPUS_BLOCK", 64)
    monkeypatch.setattr(groundtruth, "_QUERY_BLOCK", 7)
    blocked = groundtruth.compute(corpus, queries, k=10)

    np.testing.assert_array_equal(expected.indices, blocked.indices)
    np.testing.assert_allclose(expected.scores, blocked.scores, rtol=1e-5)


def test_allowlist_restricts_and_returns_absolute_indices() -> None:
    rng = np.random.default_rng(3)
    corpus = rng.standard_normal((500, 8), dtype=np.float32)
    queries = rng.standard_normal((4, 8), dtype=np.float32)
    allowed = np.array([10, 20, 30, 40, 50], dtype=np.int64)

    result = groundtruth.compute(corpus, queries, k=3, allowlist=allowed)

    assert np.isin(result.indices, allowed).all()


def test_allowlist_smaller_than_k_returns_fewer_results() -> None:
    """Returning fewer than k is correct when the filter allows fewer than k."""
    rng = np.random.default_rng(4)
    corpus = rng.standard_normal((100, 8), dtype=np.float32)
    queries = rng.standard_normal((2, 8), dtype=np.float32)

    result = groundtruth.compute(corpus, queries, k=10, allowlist=np.array([1, 2, 3]))

    assert result.k == 3


def test_allowlist_result_matches_manual_restriction() -> None:
    rng = np.random.default_rng(5)
    corpus = rng.standard_normal((300, 8), dtype=np.float32)
    queries = rng.standard_normal((3, 8), dtype=np.float32)
    allowed = np.arange(0, 300, 7, dtype=np.int64)

    filtered = groundtruth.compute(corpus, queries, k=5, allowlist=allowed)
    manual = groundtruth.compute(corpus[allowed], queries, k=5)

    np.testing.assert_array_equal(filtered.indices, allowed[manual.indices])


def test_cosine_ignores_magnitude() -> None:
    corpus = np.array([[1.0, 0.0], [100.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = groundtruth.compute(
        corpus, np.array([[2.0, 0.0]], dtype=np.float32), k=2, metric="cosine"
    )
    assert result.scores[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert result.scores[0, 1] == pytest.approx(1.0, abs=1e-5)


def test_normalize_leaves_zero_vector_finite() -> None:
    result = groundtruth.normalize(np.array([[0.0, 0.0]], dtype=np.float32))
    assert np.isfinite(result).all()


def test_k_larger_than_corpus_is_clamped() -> None:
    corpus = np.eye(3, dtype=np.float32)
    result = groundtruth.compute(corpus, corpus[:1], k=99)
    assert result.k == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": 0}, "k must be >= 1"),
        ({"metric": "jaccard"}, "unsupported metric"),
    ],
)
def test_invalid_arguments_raise(kwargs: dict, message: str) -> None:
    corpus = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        groundtruth.compute(corpus, corpus[:1], **{"k": 2, **kwargs})


def test_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        groundtruth.compute(np.eye(4, dtype=np.float32), np.eye(3, dtype=np.float32), k=1)


def test_empty_allowlist_raises() -> None:
    corpus = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError, match="allowlist is empty"):
        groundtruth.compute(corpus, corpus[:1], k=1, allowlist=np.array([], dtype=np.int64))


def test_out_of_range_allowlist_raises() -> None:
    corpus = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError, match="out-of-range"):
        groundtruth.compute(corpus, corpus[:1], k=1, allowlist=np.array([99], dtype=np.int64))
