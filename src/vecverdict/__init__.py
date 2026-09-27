"""Vector Verdict — measure what your vector index actually returns.

Most vector-search evaluations report latency and average recall. Neither
reveals that a filtered query asked for ten results and received four. This
package measures retrieval against exact ground truth and reports that
shortfall explicitly.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["environment", "groundtruth", "metrics"]
