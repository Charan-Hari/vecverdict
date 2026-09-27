"""Machine metadata for reproducible, non-identifying benchmark results.

Benchmark JSONs are meant to be published. Anything that identifies the machine
or its owner must never reach them, while enough hardware context must survive
that the numbers remain interpretable: CPU model and core count explain SIMD
throughput, RAM explains what fits in memory.

This module is the single place environment data is collected, so the
allowlist below is the complete set of fields that can ever be published.
"""

from __future__ import annotations

import platform
import re
import sys
from dataclasses import asdict, dataclass

# Substrings that may appear inside a CPU brand string on some platforms and
# are stripped defensively before publication.
_SCRUB_PATTERNS = (
    re.compile(r"\b[A-Za-z]:\\[^\s]*"),  # Windows absolute paths
    re.compile(r"/(?:home|Users)/[^\s/]+"),  # POSIX home directories
)


@dataclass(frozen=True)
class MachineInfo:
    """Non-identifying description of the machine a benchmark ran on.

    Deliberately excludes hostname, username, MAC address, local IP, and any
    filesystem path. Those are never collected, so they cannot leak.
    """

    os: str
    arch: str
    cpu: str
    cpu_count: int
    ram_gb: float | None
    python: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def scrub(text: str) -> str:
    """Remove path-like and user-identifying fragments from a free-text field."""
    for pattern in _SCRUB_PATTERNS:
        text = pattern.sub("<redacted>", text)
    return " ".join(text.split())


def _cpu_count() -> int:
    try:
        import os

        # Respects container CPU limits where available, unlike cpu_count().
        if hasattr(os, "sched_getaffinity"):
            return len(os.sched_getaffinity(0))
        return os.cpu_count() or 0
    except Exception:  # pragma: no cover - platform dependent
        return 0


def _ram_gb() -> float | None:
    try:
        import os

        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round(pages * page_size / 1024**3, 1)
    except Exception:  # pragma: no cover - platform dependent
        pass
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / 1024**3, 1)
    except Exception:  # pragma: no cover - platform dependent
        pass
    return None


def _cpu_model() -> str:
    # platform.processor() is empty on many Linux builds and verbose on Windows.
    model = platform.processor() or platform.machine() or "unknown"
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        model = line.split(":", 1)[1]
                        break
        except OSError:  # pragma: no cover - platform dependent
            pass
    return scrub(model)


def collect() -> MachineInfo:
    """Collect publishable machine metadata.

    Returns:
        MachineInfo containing only hardware and runtime facts. No identifier
        that could be traced to a person or host is gathered.
    """
    return MachineInfo(
        os=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
        cpu=_cpu_model(),
        cpu_count=_cpu_count(),
        ram_gb=_ram_gb(),
        python=platform.python_version(),
    )
