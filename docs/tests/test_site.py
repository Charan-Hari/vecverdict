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

# These tests guard the repository's published site, which is deliberately not
# shipped in the sdist. Skip rather than fail when running from a source
# distribution, where docs/ was never present to begin with.
pytestmark = pytest.mark.skipif(
    not DOCS.is_dir(), reason="docs/ is not part of the source distribution"
)


@pytest.mark.parametrize("name", SITE_FILES)
def test_site_file_exists(name):
    """Every asset the page references is committed."""
    assert (DOCS / name).is_file(), f"docs/{name} is missing"


def _site_data(name):
    return json.loads((DOCS / name).read_text(encoding="utf-8"))


def test_site_data_matches_a_committed_result():
    """The site serves measured bytes, not a hand-edited copy.

    Result files are named after the dataset that produced them, so this locates
    the matching file by content rather than assuming a fixed name.
    """
    for site_name in ("data/probe.json", "data/sweep.json"):
        site = _site_data(site_name)
        candidates = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in RESULTS.glob("*.json")
            if not path.name.startswith("_")
        ]
        assert any(site == candidate for candidate in candidates), (
            f"docs/{site_name} matches no file in results/; "
            "regenerate with `vecverdict site` rather than editing the site copy"
        )


def test_site_sweep_and_probe_describe_the_same_dataset():
    """A probe from one dataset beside a sweep from another would mislead."""
    probe = _site_data("data/probe.json")
    sweep = _site_data("data/sweep.json")
    assert probe["dataset"] == sweep["dataset"], (
        f"probe is {probe['dataset']!r} but sweep is {sweep['dataset']!r}; "
        "the page would present them as one run"
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
    # Attribute values are markup, not prose: a DOI in a citation URL is not a
    # measurement a visitor reads. Strip them so only visible text is checked.
    body = re.sub(r"<[^>]+>", " ", body)

    # Any decimal, percentage, or thousands-separated figure is a measurement
    # that should have come from the data instead.
    for pattern in (r"\d+\.\d+", r"\d+\s*%", r"\d{1,3},\d{3}"):
        found = re.findall(pattern, body)
        assert not found, f"hard-coded measurement in index.html: {found[:5]}"


def test_every_placeholder_is_populated_by_script():
    """Guard the failure where a new element keeps its em-dash placeholder.

    The page ships em-dashes as placeholders. If app.js never writes to an id,
    a visitor sees "—" where a measurement belongs, which reads as a broken
    page. Every placeholder id must therefore be referenced by the script.
    """
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    script = (DOCS / "app.js").read_text(encoding="utf-8")

    placeholder_ids = re.findall(r'id="([^"]+)"[^>]*>\s*—\s*<', html)
    assert placeholder_ids, "expected placeholder elements in index.html"

    unwired = [name for name in placeholder_ids if f'"{name}"' not in script]
    assert not unwired, f"placeholders never filled in by app.js: {unwired}"


def test_page_references_only_committed_assets():
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
