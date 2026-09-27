"""Tests for the published demo site.

The site's whole claim is that its numbers are measured rather than written.
These tests enforce that: the JSON it loads must be byte-identical to the
committed results, and the page must not contain hard-coded figures that could
drift away from them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"

SITE_FILES = ["index.html", "style.css", "app.js", "data/probe.json", "data/sweep.json"]


@pytest.mark.parametrize("name", SITE_FILES)
def test_site_file_exists(name):
    """Every asset the page references is committed."""
    assert (DOCS / name).is_file(), f"docs/{name} is missing"


def test_site_data_matches_committed_results():
    """The site serves the same bytes as results/, not a stale copy."""
    pairs = [
        ("data/probe.json", "probe_0.1pct.json"),
        ("data/sweep.json", "demo_synthetic_30k.json"),
    ]
    for site_name, result_name in pairs:
        site = json.loads((DOCS / site_name).read_text(encoding="utf-8"))
        source = json.loads((RESULTS / result_name).read_text(encoding="utf-8"))
        assert site == source, (
            f"docs/{site_name} has drifted from results/{result_name}; "
            "regenerate it rather than editing the site copy"
        )


def test_probe_data_shows_a_real_shortfall():
    """The demo must rest on a measured failure, not a flattering run."""
    probe = json.loads((DOCS / "data/probe.json").read_text(encoding="utf-8"))
    k = probe["k"]

    assert probe["n_allowed"] >= k, "allowlist must permit at least k results"

    worst = min(p["n_returned"] for p in probe["probes"])
    best = max(p["n_correct"] for p in probe["probes"])

    assert worst < k, "no backend under-returned; the page would have no subject"
    assert best == k, "no backend found all k; the shortfall would be unavoidable"


def test_page_hard_codes_no_measurements():
    """Numbers belong in the JSON, so the page cannot contradict it."""
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    body = re.sub(r"<pre>.*?</pre>", "", html, flags=re.DOTALL)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)

    # Any decimal, percentage, or thousands-separated figure is a measurement
    # that should have come from the data instead.
    for pattern in (r"\d+\.\d+", r"\d+\s*%", r"\d{1,3},\d{3}"):
        found = re.findall(pattern, body)
        assert not found, f"hard-coded measurement in index.html: {found[:5]}"


def test_page_references_only_committed_assets():
    """No link points at a file that was never committed."""
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="([^"#]+)"', html)
    for ref in refs:
        if ref.startswith(("http://", "https://", "mailto:")):
            continue
        assert (DOCS / ref).exists(), f"index.html references missing {ref}"


def test_site_data_carries_no_identifying_information():
    """The site is public; its data must stay anonymous."""
    allowed_machine = {"os", "arch", "cpu", "cpu_count", "ram_gb", "python"}
    for name in ("data/probe.json", "data/sweep.json"):
        payload = json.loads((DOCS / name).read_text(encoding="utf-8"))
        assert set(payload.get("machine", {})) <= allowed_machine

        blob = json.dumps(payload).lower()
        for leak in ("users", "/home/", "c:\\", "@", "token", "password", "haric"):
            assert leak not in blob, f"docs/{name} leaked {leak!r}"
