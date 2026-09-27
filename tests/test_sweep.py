"""Tests for the selectivity sweep and its serialised output."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vecverdict import backends, datasets, sweep

INSTALLED = backends.available()
pytestmark = pytest.mark.skipif(not INSTALLED, reason="no backends installed")

DIM = 16
N = 600
K = 5


@pytest.fixture(scope="module")
def dataset() -> datasets.Dataset:
    return datasets.synthetic(n=N, dim=DIM, n_queries=6, n_clusters=20, seed=3)


@pytest.fixture
def result(dataset: datasets.Dataset) -> sweep.SweepResult:
    instances = [backends.create(name, dim=DIM) for name in INSTALLED[:2]]
    try:
        return sweep.run(
            instances,
            dataset.vectors,
            dataset.ids,
            dataset.queries,
            k=K,
            selectivities=(1.0, 0.1, 0.01),
            dataset=dataset.name,
        )
    finally:
        for instance in instances:
            instance.close()


def test_allowlist_respects_requested_fraction() -> None:
    ids = np.arange(1000, dtype=np.int64)
    allowed = sweep.make_allowlist(ids, 0.1, k=5, rng=np.random.default_rng(0))
    assert allowed.size == 100


def test_allowlist_never_smaller_than_k() -> None:
    """A filter narrower than k would make shortfall unavoidable, not diagnostic."""
    ids = np.arange(1000, dtype=np.int64)
    allowed = sweep.make_allowlist(ids, 0.0001, k=10, rng=np.random.default_rng(0))
    assert allowed.size == 10


def test_allowlist_is_deterministic_for_a_seed() -> None:
    ids = np.arange(500, dtype=np.int64)
    first = sweep.make_allowlist(ids, 0.2, 5, np.random.default_rng(42))
    second = sweep.make_allowlist(ids, 0.2, 5, np.random.default_rng(42))
    np.testing.assert_array_equal(first, second)


def test_allowlist_entries_are_unique_and_sorted() -> None:
    ids = np.arange(300, dtype=np.int64)
    allowed = sweep.make_allowlist(ids, 0.5, 5, np.random.default_rng(1))
    assert np.unique(allowed).size == allowed.size
    assert np.all(np.diff(allowed) > 0)


@pytest.mark.parametrize("selectivity", [0.0, -0.1, 1.5])
def test_invalid_selectivity_is_rejected(selectivity: float) -> None:
    ids = np.arange(100, dtype=np.int64)
    with pytest.raises(ValueError, match="selectivity must be"):
        sweep.make_allowlist(ids, selectivity, 5, np.random.default_rng(0))


def test_sweep_covers_every_backend_and_selectivity(result: sweep.SweepResult) -> None:
    assert len(result.points) == len(INSTALLED[:2]) * 3


def test_sweep_records_dataset_shape(result: sweep.SweepResult, dataset: datasets.Dataset) -> None:
    assert result.n_vectors == dataset.size
    assert result.dim == DIM
    assert result.k == K


def test_points_are_ordered_from_wide_to_narrow(result: sweep.SweepResult) -> None:
    for name in result.backends():
        selectivities = [point.selectivity for point in result.for_backend(name)]
        assert selectivities == sorted(selectivities, reverse=True)


def test_unfiltered_backends_return_full_result_sets(result: sweep.SweepResult) -> None:
    for point in result.points:
        if point.selectivity == 1.0:
            assert point.quality.mean_returned == pytest.approx(K)


def test_allowed_count_shrinks_with_selectivity(result: sweep.SweepResult) -> None:
    for name in result.backends():
        counts = [point.n_allowed for point in result.for_backend(name)]
        assert counts == sorted(counts, reverse=True)


def test_every_backend_sees_the_same_allowlist(result: sweep.SweepResult) -> None:
    """Backends must be compared on identical filters, not merely similar ones."""
    by_selectivity: dict[float, set[int]] = {}
    for point in result.points:
        by_selectivity.setdefault(point.selectivity, set()).add(point.n_allowed)
    for counts in by_selectivity.values():
        assert len(counts) == 1


def test_empty_backend_list_is_rejected(dataset: datasets.Dataset) -> None:
    with pytest.raises(ValueError, match="no backends"):
        sweep.run([], dataset.vectors, dataset.ids, dataset.queries)


def test_serialised_result_is_valid_json(result: sweep.SweepResult, tmp_path) -> None:
    path = result.save(tmp_path / "sweep.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["schema"] == "vecverdict/sweep/1"
    assert len(loaded["points"]) == len(result.points)


def test_serialised_result_contains_no_identifying_data(
    result: sweep.SweepResult, tmp_path
) -> None:
    """Results are published, so they must carry no user or path information."""
    path = result.save(tmp_path / "sweep.json")
    blob = path.read_text(encoding="utf-8").lower()

    for forbidden in ("users\\", "/home/", "c:\\", "hostname", "username"):
        assert forbidden not in blob


def test_serialised_machine_info_keeps_hardware_context(
    result: sweep.SweepResult, tmp_path
) -> None:
    loaded = json.loads(result.save(tmp_path / "s.json").read_text(encoding="utf-8"))
    assert loaded["machine"]["cpu_count"] > 0
    assert "python" in loaded["machine"]
