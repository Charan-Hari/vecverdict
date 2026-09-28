"""Tests for the per-query probe exporter."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vecverdict import backends, datasets, probe, sweep
from vecverdict.metrics import MISSING


@pytest.fixture(scope="module")
def corpus():
    """A small clustered corpus with ids and one query."""
    data = datasets.synthetic(n=2000, dim=32, n_queries=3, seed=0)
    return data


def _installed(dim: int):
    return [backends.create(name, dim=dim) for name in backends.available()]


def test_probe_records_one_slot_per_k(corpus):
    """Every backend reports exactly k slots, filled or not."""
    built = _installed(corpus.dim)
    if not built:
        pytest.skip("no backends installed")

    rng = np.random.default_rng(0)
    allowed = sweep.make_allowlist(corpus.ids, 0.05, 10, rng)

    result = probe.probe(
        built, corpus.vectors, corpus.ids, corpus.queries[0], allowed, k=10
    )

    assert result.probes
    for entry in result.probes:
        assert len(entry.returned) == 10
        assert len(entry.correct) == 10


def test_probe_never_marks_a_missing_slot_correct(corpus):
    """An empty slot is never counted as a hit."""
    built = _installed(corpus.dim)
    if not built:
        pytest.skip("no backends installed")

    rng = np.random.default_rng(1)
    allowed = sweep.make_allowlist(corpus.ids, 0.02, 10, rng)

    result = probe.probe(
        built, corpus.vectors, corpus.ids, corpus.queries[0], allowed, k=10
    )

    for entry in result.probes:
        for value, hit in zip(entry.returned, entry.correct, strict=True):
            if value == MISSING:
                assert not hit


def test_probe_counts_match_the_slots(corpus):
    """Reported totals agree with the slot lists they summarise."""
    built = _installed(corpus.dim)
    if not built:
        pytest.skip("no backends installed")

    rng = np.random.default_rng(2)
    allowed = sweep.make_allowlist(corpus.ids, 0.05, 10, rng)

    result = probe.probe(
        built, corpus.vectors, corpus.ids, corpus.queries[0], allowed, k=10
    )

    for entry in result.probes:
        assert entry.n_returned == sum(v != MISSING for v in entry.returned)
        assert entry.n_correct == sum(entry.correct)
        assert entry.n_correct <= entry.n_returned


def test_every_returned_id_is_permitted_by_the_filter(corpus):
    """No backend may return an id the filter excluded."""
    built = _installed(corpus.dim)
    if not built:
        pytest.skip("no backends installed")

    rng = np.random.default_rng(3)
    allowed = sweep.make_allowlist(corpus.ids, 0.05, 10, rng)
    permitted = set(int(v) for v in allowed)

    result = probe.probe(
        built, corpus.vectors, corpus.ids, corpus.queries[0], allowed, k=10
    )

    for entry in result.probes:
        for value in entry.returned:
            if value != MISSING:
                assert value in permitted, f"{entry.backend} returned filtered-out id"


def test_truth_is_shared_by_every_backend(corpus):
    """Ground truth depends on the query and filter, not the backend."""
    built = _installed(corpus.dim)
    if len(built) < 2:
        pytest.skip("needs two backends")

    rng = np.random.default_rng(4)
    allowed = sweep.make_allowlist(corpus.ids, 0.05, 10, rng)

    result = probe.probe(
        built, corpus.vectors, corpus.ids, corpus.queries[0], allowed, k=10
    )

    truths = {tuple(entry.truth) for entry in result.probes}
    assert len(truths) == 1


def test_exhaustive_backend_finds_every_true_neighbour(corpus):
    """A flat index has no excuse: it must fill all k slots correctly."""
    if "faiss-flat" not in backends.available():
        pytest.skip("faiss not installed")

    flat = backends.create("faiss-flat", dim=corpus.dim)
    rng = np.random.default_rng(5)
    allowed = sweep.make_allowlist(corpus.ids, 0.05, 10, rng)

    result = probe.probe(
        [flat], corpus.vectors, corpus.ids, corpus.queries[0], allowed, k=10
    )

    entry = result.probes[0]
    assert entry.n_returned == 10
    assert entry.n_correct == 10


def test_allowlist_smaller_than_k_is_refused(corpus):
    """A filter narrower than k would force a shortfall and prove nothing."""
    built = _installed(corpus.dim)
    if not built:
        pytest.skip("no backends installed")

    allowed = corpus.ids[:4]
    with pytest.raises(ValueError, match="smaller than k"):
        probe.probe(
            built, corpus.vectors, corpus.ids, corpus.queries[0], allowed, k=10
        )


def test_saved_probe_is_json_and_carries_no_identifying_data(corpus, tmp_path):
    """Probe files are published, so they must stay anonymous."""
    built = _installed(corpus.dim)
    if not built:
        pytest.skip("no backends installed")

    rng = np.random.default_rng(6)
    allowed = sweep.make_allowlist(corpus.ids, 0.05, 10, rng)
    result = probe.probe(
        built, corpus.vectors, corpus.ids, corpus.queries[0], allowed, k=10
    )

    path = result.save(tmp_path / "probe.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == probe.SCHEMA
    assert payload["probes"]

    allowed_machine = {"os", "arch", "cpu", "cpu_count", "ram_gb", "python"}
    assert set(payload["machine"]) <= allowed_machine

    blob = json.dumps(payload).lower()
    for leak in ("users", "home", "c:\\", "/home/", "@", "token", "password"):
        assert leak not in blob, f"probe file leaked {leak!r}"


def test_probe_accepts_a_flat_query_vector(corpus):
    """A 1-D query is reshaped rather than rejected."""
    built = _installed(corpus.dim)
    if not built:
        pytest.skip("no backends installed")

    rng = np.random.default_rng(7)
    allowed = sweep.make_allowlist(corpus.ids, 0.05, 10, rng)

    flat_query = corpus.queries[0].reshape(-1)
    result = probe.probe(
        built, corpus.vectors, corpus.ids, flat_query, allowed, k=10
    )

    assert result.probes
    assert result.n_allowed == allowed.size
