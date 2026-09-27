"""Command line interface.

Four commands, each answering one question: demo ("show me the problem"),
filter ("does my stack have it?"), switch ("should I move these vectors?"),
and site ("regenerate the published demo data").
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from pathlib import Path

import numpy as np

from . import __version__, backends, datasets, probe, sweep, switch
from .sweep import DEFAULT_SELECTIVITIES, SweepResult

# Backends measured when the user does not name any.
_DEFAULT_BACKENDS = ("turbovec", "faiss-hnsw", "faiss-flat")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
        stream=sys.stderr,
    )
    _use_utf8_output()

    if args.command is None:
        parser.print_help()
        return 0

    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\ninterrupted", file=sys.stderr)
        return 130


def _use_utf8_output() -> None:
    """Allow non-ASCII output on consoles that default to a legacy code page.

    Windows terminals often default to cp1252, which cannot encode the
    characters used in report text and would raise mid-print.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # A console that refuses the change keeps its default encoding.
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vecverdict",
        description="Measure what your vector index actually returns, not what it claims.",
    )
    parser.add_argument("--version", action="version", version=f"vecverdict {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress to stderr")
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser(
        "demo", help="reproduce the filtered-search shortfall on generated data"
    )
    demo.add_argument("--n", type=int, default=20_000, help="corpus size (default: 20000)")
    demo.add_argument("--dim", type=int, default=128, help="dimension (default: 128)")
    demo.add_argument("--k", type=int, default=10, help="results per query (default: 10)")
    demo.add_argument("--charts", type=Path, help="directory to write SVG charts into")
    demo.set_defaults(handler=_run_demo)

    filter_cmd = subparsers.add_parser(
        "filter", help="sweep filter selectivity across backends"
    )
    filter_cmd.add_argument(
        "--dataset",
        default="synthetic",
        help="'synthetic', 'dbpedia', or a path to a .npy/.npz file (default: synthetic)",
    )
    filter_cmd.add_argument("--limit", type=int, default=50_000, help="max vectors to load")
    filter_cmd.add_argument("--k", type=int, default=10, help="results per query (default: 10)")
    filter_cmd.add_argument("--queries", type=int, default=50, help="number of queries")
    filter_cmd.add_argument(
        "--backends",
        nargs="+",
        default=list(_DEFAULT_BACKENDS),
        help=f"backends to measure (available: {', '.join(backends.names())})",
    )
    filter_cmd.add_argument("--metric", default="ip", choices=("ip", "l2", "cosine"))
    filter_cmd.add_argument("--seed", type=int, default=0)
    filter_cmd.add_argument("--json", type=Path, help="write the full result JSON here")
    filter_cmd.add_argument("--charts", type=Path, help="directory to write SVG charts into")
    filter_cmd.set_defaults(handler=_run_filter)

    switch_cmd = subparsers.add_parser(
        "switch", help="report whether vectors should move to a compressed index"
    )
    source = switch_cmd.add_mutually_exclusive_group(required=True)
    source.add_argument("--index", type=Path, help="path to a FAISS index file")
    source.add_argument("--vectors", type=Path, help="path to a .npy/.npz of float32 vectors")
    switch_cmd.add_argument("--k", type=int, default=10, help="results per query (default: 10)")
    switch_cmd.add_argument(
        "--bit-width", type=int, default=4, choices=(2, 4), help="target width (default: 4)"
    )
    switch_cmd.add_argument("--metric", default="ip", choices=("ip", "l2", "cosine"))
    switch_cmd.add_argument("--queries", type=int, default=100, help="queries to sample")
    switch_cmd.add_argument(
        "--allow-lossy",
        action="store_true",
        help="permit reading a quantised index despite stacked-codec distortion",
    )
    switch_cmd.add_argument("--json", type=Path, help="write the report JSON here")
    switch_cmd.set_defaults(handler=_run_switch)

    site_cmd = subparsers.add_parser(
        "site", help="regenerate the demo site's measured data"
    )
    site_cmd.add_argument("--n", type=int, default=30_000, help="corpus size (default: 30000)")
    site_cmd.add_argument("--dim", type=int, default=128, help="dimension (default: 128)")
    site_cmd.add_argument("--k", type=int, default=10, help="results per query (default: 10)")
    site_cmd.add_argument("--queries", type=int, default=50, help="number of queries")
    site_cmd.add_argument(
        "--selectivity",
        type=float,
        default=0.001,
        help="filter width for the per-query probe (default: 0.001)",
    )
    site_cmd.add_argument("--seed", type=int, default=0)
    site_cmd.add_argument(
        "--docs", type=Path, default=Path("docs"), help="site directory (default: docs)"
    )
    site_cmd.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="directory for the canonical result files (default: results)",
    )
    site_cmd.set_defaults(handler=_run_site)

    return parser


def _run_demo(args: argparse.Namespace) -> int:
    installed = backends.available()
    chosen = [name for name in _DEFAULT_BACKENDS if name in installed]
    if not chosen:
        print(
            "error: no backends installed. Try: pip install 'vecverdict[turbovec,faiss]'",
            file=sys.stderr,
        )
        return 1

    print(f"Generating {args.n:,} clustered vectors (dim {args.dim})...")
    dataset = datasets.synthetic(n=args.n, dim=args.dim, n_queries=50)
    result = _sweep(dataset, chosen, args.k, "ip", seed=0)

    _print_table(result)
    _print_summary(result)
    if args.charts:
        _write_charts(result, args.charts)
    return 0


def _run_filter(args: argparse.Namespace) -> int:
    installed = backends.available()
    missing = [name for name in args.backends if name not in installed]
    if missing:
        print(f"error: backends not installed: {', '.join(missing)}", file=sys.stderr)
        print(f"installed: {', '.join(installed) or 'none'}", file=sys.stderr)
        return 1

    dataset = datasets.load(
        args.dataset, limit=args.limit, n_queries=args.queries, seed=args.seed
    )
    print(f"{dataset.name}: {dataset.size:,} vectors, dim {dataset.dim}")
    result = _sweep(dataset, args.backends, args.k, args.metric, seed=args.seed)

    _print_table(result)
    _print_summary(result)
    if args.json:
        print(f"\nwrote {result.save(args.json)}")
    if args.charts:
        _write_charts(result, args.charts)
    return 0


def _run_switch(args: argparse.Namespace) -> int:
    if args.index:
        try:
            vectors, source = switch.inspect_faiss_index(args.index)
        except ValueError as exc:
            if not args.allow_lossy:
                raise
            print(f"warning: {exc}\n", file=sys.stderr)
            import faiss

            index = faiss.read_index(str(args.index))
            vectors = np.empty((index.ntotal, index.d), dtype=np.float32)
            index.reconstruct_n(0, index.ntotal, vectors)
            source = f"{type(index).__name__} (lossy, forced)"
    else:
        dataset = datasets.from_numpy(args.vectors, n_queries=args.queries)
        vectors, source = dataset.vectors, f"{args.vectors.name}"

    print(f"Evaluating {vectors.shape[0]:,} vectors (dim {vectors.shape[1]})...\n")
    report = switch.evaluate(
        vectors,
        k=args.k,
        bit_width=args.bit_width,
        metric=args.metric,
        source_name=source,
        n_queries=args.queries,
    )

    print(f"  source           {report.source}")
    print(f"  memory now       {report.source_bytes / 1024**2:,.1f} MB")
    print(
        f"  memory after     {report.target_bytes / 1024**2:,.1f} MB "
        f"({report.savings_ratio:.1f}x smaller)"
    )
    print(f"  recall@{report.k:<10} {report.recall:.3f}")
    print(f"  recall@1         {report.recall_at_1:.3f}")
    print(f"\n  VERDICT: {report.verdict.upper()}")
    for reason in report.reasons:
        print(f"    - {reason}")

    if args.json:
        import json

        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


def _run_site(args: argparse.Namespace) -> int:
    """Regenerate both JSON files the demo site reads.

    The site is only credible while its data is measured output, so this
    writes the canonical copies under results/ and mirrors them into the site
    directory rather than letting the two be edited apart.
    """
    installed = backends.available()
    if not installed:
        print(
            "error: no backends installed. Try: pip install 'vecverdict[turbovec,faiss,chroma]'",
            file=sys.stderr,
        )
        return 1

    print(f"Generating {args.n:,} clustered vectors (dim {args.dim})...")
    dataset = datasets.synthetic(
        n=args.n, dim=args.dim, n_queries=args.queries, seed=args.seed
    )

    print(f"Sweeping {len(installed)} backends: {', '.join(installed)}")
    result = _sweep(dataset, installed, args.k, "ip", seed=args.seed)

    instances = [backends.create(name, dim=dataset.dim) for name in installed]
    try:
        for instance in instances:
            instance.build(dataset.vectors, dataset.ids)

        rng = np.random.default_rng([args.seed, int(args.selectivity * 1e9)])
        allowed = sweep.make_allowlist(dataset.ids, args.selectivity, args.k, rng)

        # Show the query that most separates the backends: a run where they all
        # agree would hide the very behaviour the page exists to demonstrate.
        chosen, widest = 0, -1
        for index in range(min(10, dataset.queries.shape[0])):
            candidate = probe.probe(
                instances,
                dataset.vectors,
                dataset.ids,
                dataset.queries[index],
                allowed,
                k=args.k,
                dataset=dataset.name,
                query_index=index,
                build=False,
            )
            counts = [entry.n_correct for entry in candidate.probes]
            spread = max(counts) - min(counts)
            if spread > widest:
                chosen, widest = index, spread

        probe_result = probe.probe(
            instances,
            dataset.vectors,
            dataset.ids,
            dataset.queries[chosen],
            allowed,
            k=args.k,
            dataset=dataset.name,
            query_index=chosen,
            build=False,
        )
    finally:
        for instance in instances:
            instance.close()

    sweep_path = result.save(args.results / "demo_synthetic_30k.json")
    probe_path = probe_result.save(args.results / "probe_0.1pct.json")

    data_dir = args.docs / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "sweep.json").write_text(
        sweep_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (data_dir / "probe.json").write_text(
        probe_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    _write_charts(result, args.docs)

    print(f"\nProbe query {chosen} at {args.selectivity:.3%} ({allowed.size} allowed):")
    for entry in probe_result.probes:
        print(
            f"  {entry.backend:<24} returned {entry.n_returned:>2}/{args.k}"
            f"   correct {entry.n_correct:>2}/{args.k}"
        )
    print(f"\nWrote {sweep_path}, {probe_path}, and {data_dir}")
    return 0


def _sweep(
    dataset: datasets.Dataset, names: list[str], k: int, metric: str, seed: int
) -> SweepResult:
    instances = [backends.create(name, dim=dataset.dim, metric=metric) for name in names]
    try:
        return sweep.run(
            instances,
            dataset.vectors,
            dataset.ids,
            dataset.queries,
            k=k,
            selectivities=DEFAULT_SELECTIVITIES,
            metric=metric,
            dataset=dataset.name,
            seed=seed,
        )
    finally:
        for instance in instances:
            instance.close()


def _print_table(result: SweepResult) -> None:
    print(
        f"\n{'backend':<26}{'allowed':>10}{'sel%':>9}"
        f"{'returned':>11}{'recall':>9}{'missing':>9}"
    )
    print("-" * 74)
    for name in result.backends():
        for point in result.for_backend(name):
            quality = point.quality
            print(
                f"{name:<26}{point.n_allowed:>10,}{point.selectivity * 100:>8.3f}%"
                f"{quality.mean_returned:>7.1f}/{point.k:<3}{quality.recall_at_k:>9.3f}"
                f"{quality.mean_shortfall:>9.1f}"
            )


def _print_summary(result: SweepResult) -> None:
    """Name the backends that returned fewer results than requested."""
    failures = [
        (point.backend, point)
        for point in result.points
        if point.quality.mean_shortfall > 0.5
    ]
    print()
    if not failures:
        print("All backends returned the full result set at every selectivity.")
        return

    worst: dict[str, sweep.SweepPoint] = {}
    for name, point in failures:
        if name not in worst or point.quality.mean_shortfall > worst[name].quality.mean_shortfall:
            worst[name] = point

    print("Shortfall detected — these backends returned fewer results than requested:")
    for name, point in worst.items():
        quality = point.quality
        print(
            f"  {name}: at {point.selectivity * 100:.3f}% selectivity "
            f"({point.n_allowed:,} vectors allowed), returned "
            f"{quality.mean_returned:.1f} of {point.k} on average, "
            f"recall {quality.recall_at_k:.3f}"
        )
    print("\nThe allowed vectors existed and exact search found them.")


def _write_charts(result: SweepResult, directory: Path) -> None:
    from . import viz

    for path in viz.render_all(result, directory):
        print(f"wrote {path}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
