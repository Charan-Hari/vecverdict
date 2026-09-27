"""Charts that make the measurement legible at a glance.

The central chart plots results actually returned against filter selectivity.
A correct index is a flat line at k; an index that loses reachability under a
selective filter falls away from it. Presenting the failure as a distance from a
horizontal reference is what makes it readable without first explaining recall.

Output is SVG so charts render inline in a README with no hosting, and stay
diffable in version control.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .sweep import SweepResult

logger = logging.getLogger(__name__)

# Colour-blind safe; the control backend is deliberately the calm colour and
# failures land on warm ones.
_PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
_GRID = {"alpha": 0.25, "linewidth": 0.6}


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # No display needed; charts are written to disk.
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("charts require: pip install 'vecverdict[viz]'") from exc


def _percent_axis(axis) -> None:
    axis.set_xscale("log")
    axis.invert_xaxis()
    axis.set_xlabel("filter selectivity (% of corpus allowed) — narrower to the right")


def shortfall_cliff(result: SweepResult, path: str | Path) -> Path:
    """Plot results returned versus selectivity, against the k requested.

    Args:
        result: A completed sweep.
        path: Destination .svg path.

    Returns:
        The written path.
    """
    plt = _require_matplotlib()
    figure, axis = plt.subplots(figsize=(9, 5.5))

    for index, name in enumerate(result.backends()):
        points = result.for_backend(name)
        x = [point.selectivity * 100 for point in points]
        y = [point.quality.mean_returned for point in points]
        axis.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=_PALETTE[index % len(_PALETTE)],
            label=name,
        )

    axis.axhline(
        result.k,
        color="#444444",
        linestyle="--",
        linewidth=1.3,
        label=f"requested (k={result.k})",
    )

    _percent_axis(axis)
    axis.set_ylabel(f"results actually returned (of {result.k} requested)")
    axis.set_ylim(0, result.k * 1.15)
    axis.set_title(
        "Filtered search: what you asked for vs. what you got",
        fontsize=13,
        fontweight="bold",
    )
    axis.grid(True, **_GRID)
    axis.legend(frameon=False, fontsize=9)
    figure.text(
        0.5,
        -0.02,
        f"{result.dataset} · {result.n_vectors:,} vectors · dim {result.dim} · "
        f"{result.n_queries} queries",
        ha="center",
        fontsize=8,
        color="#666666",
    )

    return _save(figure, path)


def recall_curve(result: SweepResult, path: str | Path) -> Path:
    """Plot recall against selectivity, with the attainable ceiling marked.

    Recall alone is ambiguous under a narrow filter: it can fall because the
    filter left few correct answers to find. Plotting attainable recall
    alongside separates that from an index failing to find answers that existed.
    """
    plt = _require_matplotlib()
    figure, axis = plt.subplots(figsize=(9, 5.5))

    for index, name in enumerate(result.backends()):
        points = result.for_backend(name)
        colour = _PALETTE[index % len(_PALETTE)]
        x = [point.selectivity * 100 for point in points]
        axis.plot(
            x,
            [point.quality.recall_at_k for point in points],
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=colour,
            label=name,
        )
        axis.plot(
            x,
            [point.quality.attainable_recall_at_k for point in points],
            linestyle=":",
            linewidth=1.4,
            color=colour,
            alpha=0.7,
        )

    _percent_axis(axis)
    axis.set_ylabel(f"recall@{result.k}")
    axis.set_ylim(0, 1.05)
    axis.set_title(
        "Recall under filtering (dotted = attainable ceiling)",
        fontsize=13,
        fontweight="bold",
    )
    axis.grid(True, **_GRID)
    axis.legend(frameon=False, fontsize=9)

    return _save(figure, path)


def memory_comparison(result: SweepResult, path: str | Path) -> Path:
    """Plot index memory per backend, which is why compression is considered."""
    plt = _require_matplotlib()

    names = result.backends()
    sizes = [result.for_backend(name)[0].memory_bytes / 1024**2 for name in names]
    order = sorted(range(len(names)), key=lambda i: sizes[i])
    names = [names[i] for i in order]
    sizes = [sizes[i] for i in order]

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.barh(names, sizes, color=[_PALETTE[i % len(_PALETTE)] for i in range(len(names))])
    baseline = max(sizes) if sizes else 0
    for bar, size in zip(bars, sizes, strict=True):
        ratio = f"  {baseline / size:.1f}x smaller" if size and size < baseline else ""
        axis.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {size:,.0f} MB{ratio}",
            va="center",
            fontsize=9,
        )

    axis.set_xlabel("index memory (MB)")
    axis.set_xlim(0, baseline * 1.45 if baseline else 1)
    axis.set_title(
        f"Index memory - {result.n_vectors:,} vectors x {result.dim} dims",
        fontsize=12,
        fontweight="bold",
    )
    axis.grid(True, axis="x", **_GRID)

    return _save(figure, path)


def render_all(result: SweepResult, directory: str | Path) -> list[Path]:
    """Write every chart for a sweep into `directory`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return [
        shortfall_cliff(result, directory / "shortfall_cliff.svg"),
        recall_curve(result, directory / "recall_curve.svg"),
        memory_comparison(result, directory / "memory.svg"),
    ]


def _save(figure, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)
    logger.info("wrote %s", path.name)
    return path
