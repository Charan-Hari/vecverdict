"""Tests for dataset loading, the switch verdict, and the CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vecverdict import backends, cli, datasets, switch

INSTALLED = backends.available()


class TestDatasets:
    def test_synthetic_has_requested_shape(self) -> None:
        dataset = datasets.synthetic(n=500, dim=16, n_queries=7, n_clusters=10)
        assert dataset.vectors.shape == (500, 16)
        assert dataset.queries.shape == (7, 16)
        assert dataset.ids.size == 500

    def test_synthetic_is_reproducible_for_a_seed(self) -> None:
        first = datasets.synthetic(n=100, dim=8, seed=5)
        second = datasets.synthetic(n=100, dim=8, seed=5)
        np.testing.assert_array_equal(first.vectors, second.vectors)

    def test_synthetic_is_clustered_not_uniform(self) -> None:
        """Clustering is required for graph reachability effects to appear."""
        dataset = datasets.synthetic(n=2000, dim=16, n_clusters=20, spread=0.3, seed=1)
        vectors = dataset.vectors
        distances = np.linalg.norm(vectors[:200, None, :] - vectors[None, :200, :], axis=-1)
        off_diagonal = distances[~np.eye(200, dtype=bool)]
        # Clustered data has a wide spread of pairwise distances: near neighbours
        # inside a cluster, far ones across clusters.
        assert off_diagonal.std() > 0.5

    def test_vectors_are_float32(self) -> None:
        assert datasets.synthetic(n=50, dim=8).vectors.dtype == np.float32

    def test_from_numpy_round_trips(self, tmp_path) -> None:
        vectors = np.random.default_rng(0).standard_normal((200, 12)).astype(np.float32)
        path = tmp_path / "vecs.npy"
        np.save(path, vectors)

        dataset = datasets.from_numpy(path, n_queries=5)

        assert dataset.size == 200
        assert dataset.dim == 12
        assert dataset.queries.shape == (5, 12)

    def test_from_numpy_reads_npz(self, tmp_path) -> None:
        vectors = np.random.default_rng(0).standard_normal((100, 8)).astype(np.float32)
        path = tmp_path / "vecs.npz"
        np.savez(path, vectors=vectors)
        assert datasets.from_numpy(path, n_queries=3).size == 100

    def test_from_numpy_rejects_corpus_smaller_than_query_count(self, tmp_path) -> None:
        path = tmp_path / "tiny.npy"
        np.save(path, np.zeros((3, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="too small"):
            datasets.from_numpy(path, n_queries=10)

    def test_from_numpy_rejects_one_dimensional_input(self, tmp_path) -> None:
        path = tmp_path / "flat.npy"
        np.save(path, np.zeros(10, dtype=np.float32))
        with pytest.raises(ValueError, match="2-D"):
            datasets.from_numpy(path)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            datasets.from_numpy("does-not-exist.npy")

    def test_load_rejects_unknown_spec(self) -> None:
        with pytest.raises(ValueError, match="unknown dataset"):
            datasets.load("not-a-dataset")


@pytest.mark.skipif("turbovec" not in INSTALLED, reason="turbovec required")
class TestSwitch:
    @pytest.fixture(scope="class")
    def vectors(self) -> np.ndarray:
        rng = np.random.default_rng(11)
        centers = rng.standard_normal((40, 64)).astype(np.float32) * 3
        data = np.repeat(centers, 50, axis=0)
        data += rng.standard_normal(data.shape).astype(np.float32) * 0.3
        return np.ascontiguousarray(data, dtype=np.float32)

    def test_reports_memory_saving(self, vectors: np.ndarray) -> None:
        report = switch.evaluate(vectors, k=10, bit_width=4, n_queries=20)
        assert report.savings_ratio > 2.0
        assert report.target_bytes < report.source_bytes

    def test_reports_recall_between_zero_and_one(self, vectors: np.ndarray) -> None:
        report = switch.evaluate(vectors, k=10, n_queries=20)
        assert 0.0 <= report.recall <= 1.0

    def test_verdict_is_one_of_the_supported_values(self, vectors: np.ndarray) -> None:
        report = switch.evaluate(vectors, k=10, n_queries=20)
        assert report.verdict in {"switch", "stay"}
        assert report.reasons

    def test_two_bit_saves_more_than_four_bit(self, vectors: np.ndarray) -> None:
        low = switch.evaluate(vectors, k=10, bit_width=2, n_queries=20)
        high = switch.evaluate(vectors, k=10, bit_width=4, n_queries=20)
        assert low.savings_ratio > high.savings_ratio

    def test_poor_recall_produces_a_stay_verdict(self) -> None:
        """The tool must be able to recommend against switching."""
        report = switch.SwitchReport(
            source="test",
            n_vectors=1000,
            dim=128,
            source_bytes=1000 * 128 * 4,
            target_bytes=1000 * 128 // 2,
            bit_width=4,
            k=10,
            recall=0.55,
        )
        switch._decide(report)
        assert report.verdict == "stay"
        assert any("below" in reason for reason in report.reasons)

    def test_small_saving_produces_a_stay_verdict(self) -> None:
        report = switch.SwitchReport(
            source="test",
            n_vectors=100,
            dim=8,
            source_bytes=1000,
            target_bytes=900,
            bit_width=4,
            k=10,
            recall=0.99,
        )
        switch._decide(report)
        assert report.verdict == "stay"

    def test_report_serialises(self, vectors: np.ndarray) -> None:
        report = switch.evaluate(vectors, k=5, n_queries=10)
        data = json.loads(json.dumps(report.to_dict()))
        assert data["verdict"] in {"switch", "stay"}

    def test_rejects_non_two_dimensional_input(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            switch.evaluate(np.zeros(10, dtype=np.float32))


@pytest.mark.skipif("faiss-flat" not in INSTALLED, reason="faiss required")
class TestLossyRefusal:
    def test_refuses_product_quantised_index(self, tmp_path) -> None:
        """Re-quantising PQ output stacks two codecs, so it is declined."""
        import faiss

        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((2000, 32)).astype(np.float32)
        index = faiss.IndexPQ(32, 8, 8)
        index.train(vectors)
        index.add(vectors)
        path = tmp_path / "pq.faiss"
        faiss.write_index(index, str(path))

        with pytest.raises(ValueError, match="quantised codes"):
            switch.inspect_faiss_index(path)

    def test_accepts_flat_float32_index(self, tmp_path) -> None:
        import faiss

        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((500, 16)).astype(np.float32)
        index = faiss.IndexFlatIP(16)
        index.add(vectors)
        path = tmp_path / "flat.faiss"
        faiss.write_index(index, str(path))

        loaded, name = switch.inspect_faiss_index(path)

        assert loaded.shape == (500, 16)
        assert "Flat" in name
        np.testing.assert_allclose(loaded, vectors, rtol=1e-5)


class TestCli:
    def test_no_command_prints_help(self, capsys) -> None:
        assert cli.main([]) == 0
        assert "vecverdict" in capsys.readouterr().out

    def test_unknown_backend_is_reported(self, capsys) -> None:
        code = cli.main(["filter", "--backends", "nonexistent"])
        assert code == 1
        assert "not installed" in capsys.readouterr().err

    def test_switch_requires_a_source(self) -> None:
        with pytest.raises(SystemExit):
            cli.main(["switch"])

    @pytest.mark.skipif(not INSTALLED, reason="no backends installed")
    def test_demo_runs_and_reports(self, capsys) -> None:
        code = cli.main(["demo", "--n", "2000", "--dim", "16", "--k", "5"])
        output = capsys.readouterr().out
        assert code == 0
        assert "backend" in output
        assert "recall" in output

    @pytest.mark.skipif("turbovec" not in INSTALLED, reason="turbovec required")
    def test_switch_on_user_vectors_writes_json(self, tmp_path, capsys) -> None:
        vectors = np.random.default_rng(2).standard_normal((800, 32)).astype(np.float32)
        source = tmp_path / "v.npy"
        np.save(source, vectors)
        out = tmp_path / "report.json"

        code = cli.main(
            ["switch", "--vectors", str(source), "--k", "5", "--queries", "20", "--json", str(out)]
        )

        assert code == 0
        assert "VERDICT" in capsys.readouterr().out
        assert json.loads(out.read_text(encoding="utf-8"))["verdict"] in {"switch", "stay"}
