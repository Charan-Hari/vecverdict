"""Backend adapters for the vector indexes under measurement.

Backends are imported lazily so that `pip install vecverdict` pulls no vector
database at all, and a missing optional dependency produces an actionable
message rather than an ImportError from deep inside a module.
"""

from __future__ import annotations

from .base import Backend, BackendInfo, empty_results

# Import path per backend name, resolved on first use.
_REGISTRY: dict[str, tuple[str, str, str]] = {
    "turbovec": (".turbovec_backend", "TurbovecBackend", "turbovec"),
    "faiss-flat": (".faiss_backend", "FaissFlatBackend", "faiss"),
    "faiss-hnsw": (".faiss_backend", "FaissHnswBackend", "faiss"),
    "chroma": (".chroma_backend", "ChromaBackend", "chroma"),
}

__all__ = ["Backend", "BackendInfo", "available", "create", "empty_results", "names"]


def names() -> list[str]:
    """All registered backend names, installed or not."""
    return sorted(_REGISTRY)


def create(name: str, dim: int, metric: str = "ip", **kwargs: object) -> Backend:
    """Construct a backend by name.

    Args:
        name: A name from names(), e.g. "faiss-hnsw".
        dim: Vector dimension.
        metric: "ip", "l2", or "cosine".
        **kwargs: Backend-specific options, e.g. bit_width or ef_search.

    Raises:
        ValueError: If the name is not registered.
        ImportError: If the backend's optional dependency is not installed,
            naming the extra that provides it.
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown backend {name!r}; available: {', '.join(names())}")

    module_path, class_name, extra = _REGISTRY[name]
    try:
        from importlib import import_module

        module = import_module(module_path, package=__name__)
    except ImportError as exc:
        raise ImportError(
            f"backend {name!r} requires an optional dependency. "
            f"Install it with: pip install 'vecverdict[{extra}]'"
        ) from exc

    return getattr(module, class_name)(dim=dim, metric=metric, **kwargs)


def available() -> list[str]:
    """Names of backends whose dependencies are importable in this environment."""
    from importlib.util import find_spec

    modules = {
        "turbovec": "turbovec",
        "faiss-flat": "faiss",
        "faiss-hnsw": "faiss",
        "chroma": "chromadb",
    }
    found = []
    for name, module in modules.items():
        try:
            if find_spec(module) is not None:
                found.append(name)
        except (ImportError, ValueError):
            continue
    return sorted(found)
