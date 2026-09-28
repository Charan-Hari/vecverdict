"""Tests for retrieval quality metrics, especially shortfall accounting."""

from __future__ import annotations

import numpy as np
import pytest

from vecverdict import metrics
from vecverdict.metrics import MISSING


def test_perfect_retrieval_scores_one() -> None:
    truth = np.array([[1, 2, 3], [4, 5, 6]])
    result = metrics.evaluate(truth.copy(), truth, k=3)
    assert result.recall_at_k == pytest.approx(1.0)
    assert result.recall_at_1 == pytest.approx(1.0)
    assert result.shortfall_rate == pytest.approx(0.0)


def test_disjoint_retrieval_scores_zero() -> None:
    result = metrics.evaluate(np.array([[7, 8, 9]]), np.array([[1, 2, 3]]), k=3)
    assert result.recall_at_k == pytest.approx(0.0)


def test_order_does_not_affect_recall_at_k() -> None:
    """recall@k is a set measure; ranking within the k is not penalised."""
    result = metrics.evaluate(np.array([[3, 1, 2]]), np.array([[1, 2, 3]]), k=3)
    assert result.recall_at_k == pytest.approx(1.0)


def test_recall_at_1_requires_top_neighbour_present() -> None:
    present = metrics.evaluate(np.array([[9, 9, 1]]), np.array([[1, 2, 3]]), k=3)
    absent = metrics.evaluate(np.array([[2, 3, 9]]), np.array([[1, 2, 3]]), k=3)
    assert present.recall_at_1 == pytest.approx(1.0)
    assert absent.recall_at_1 == pytest.approx(0.0)


def test_partial_recall_is_the_hit_fraction() -> None:
    result = metrics.evaluate(np.array([[1, 2, 99, 98]]), np.array([[1, 2, 3, 4]]), k=4)
    assert result.recall_at_k == pytest.approx(0.5)


def test_padded_results_count_as_shortfall() -> None:
    """The FAISS failure mode: k slots requested, most returned as -1 padding."""
    retrieved = np.array([[1, 2, 3, MISSING, MISSING]])
    truth = np.array([[1, 2, 3, 4, 5]])

    result = metrics.evaluate(retrieved, truth, k=5)

    assert result.mean_returned == pytest.approx(3.0)
    assert result.mean_shortfall == pytest.approx(2.0)
    assert result.worst_shortfall == 2
    assert result.shortfall_rate == pytest.approx(1.0)


def test_shortfall_is_not_counted_when_filter_allows_fewer_than_k() -> None:
    """Returning 3 of 10 is correct when only 3 vectors were allowed."""
    retrieved = np.array([[1, 2, 3] + [MISSING] * 7])
    truth = np.array([[1, 2, 3] + [MISSING] * 7])

    result = metrics.evaluate(retrieved, truth, k=10, n_allowed=3)

    assert result.mean_shortfall == pytest.approx(0.0)
    assert result.shortfall_rate == pytest.approx(0.0)
    assert result.attainable_recall_at_k == pytest.approx(1.0)


def test_attainable_recall_separates_narrow_filter_from_index_failure() -> None:
    """A narrow filter caps recall@k; attainable recall shows the index was fine."""
    retrieved = np.array([[1, 2, 3] + [MISSING] * 7])
    truth = np.array([[1, 2, 3] + [MISSING] * 7])

    result = metrics.evaluate(retrieved, truth, k=10, n_allowed=3)

    assert result.recall_at_k == pytest.approx(1.0)
    assert result.attainable_recall_at_k == pytest.approx(1.0)


def test_avoidable_shortfall_is_detected_within_an_allowlist() -> None:
    """20 vectors allowed and k=10, but only 4 returned: a real index failure."""
    retrieved = np.array([[1, 2, 3, 4] + [MISSING] * 6])
    truth = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    result = metrics.evaluate(retrieved, truth, k=10, n_allowed=20)

    assert result.mean_shortfall == pytest.approx(6.0)
    assert result.shortfall_rate == pytest.approx(1.0)


def test_shortfall_rate_is_fraction_of_affected_queries() -> None:
    retrieved = np.array([[1, 2, 3], [1, 2, MISSING], [1, MISSING, MISSING], [1, 2, 3]])
    truth = np.array([[1, 2, 3]] * 4)

    result = metrics.evaluate(retrieved, truth, k=3)

    assert result.shortfall_rate == pytest.approx(0.5)
    assert result.worst_shortfall == 2


def test_queries_without_ground_truth_are_excluded_not_zeroed() -> None:
    """An undefined query must not drag the mean down as if it scored zero."""
    retrieved = np.array([[1, 2], [MISSING, MISSING]])
    truth = np.array([[1, 2], [MISSING, MISSING]])

    result = metrics.evaluate(retrieved, truth, k=2)

    assert result.recall_at_k == pytest.approx(1.0)


def test_returning_more_than_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="more than requested"):
        metrics.evaluate(np.array([[1, 2, 3]]), np.array([[1, 2]]), k=2)


def test_query_count_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="query count mismatch"):
        metrics.evaluate(np.array([[1]]), np.array([[1], [2]]), k=1)


def test_to_dict_excludes_per_query_arrays() -> None:
    result = metrics.evaluate(np.array([[1]]), np.array([[1]]), k=1)
    data = result.to_dict()
    assert "per_query_recall" not in data
    assert data["recall_at_k"] == pytest.approx(1.0)


def test_pad_to_k_marks_absent_slots_as_missing() -> None:
    padded = metrics.pad_to_k([[1, 2], [3], []], k=3)
    expected = np.array([[1, 2, MISSING], [3, MISSING, MISSING], [MISSING] * 3])
    np.testing.assert_array_equal(padded, expected)


def test_pad_to_k_rejects_overlong_rows() -> None:
    with pytest.raises(ValueError, match="more than k"):
        metrics.pad_to_k([[1, 2, 3]], k=2)
