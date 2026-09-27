<h1 align="center">Vector Verdict</h1>

<p align="center">
  <strong>Measure what your vector index actually returns — not what it claims.</strong>
</p>

<p align="center">
  <a href="https://charan-hari.github.io/vecverdict/"><strong>▶ Live demo</strong></a> ·
  <a href="#the-problem">Problem</a> ·
  <a href="#the-approach">Approach</a> ·
  <a href="#results">Results</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#how-it-works">How it works</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/tests-140%20passing-brightgreen.svg" alt="140 tests passing">
  <img src="https://img.shields.io/badge/status-alpha-orange.svg" alt="alpha">
</p>

---

## The problem

You ask your vector database for 10 results with a metadata filter applied.

It returns 1.

No exception. No warning. No degraded-mode flag. Just a shorter list than you asked for —
and if you only logged latency and average recall, you would never know.

```python
# 30 of 30,000 vectors pass the filter. All 30 are in the index.
results = index.search(query, k=10, filter=allowed_ids)
len(results)   # 1
```

Exhaustive search over those same 30 vectors returns all 10 correct neighbours instantly.
The data is there. The index simply cannot reach it.

**Why it happens.** HNSW — the index behind FAISS HNSW, pgvector, Qdrant, Weaviate,
Milvus and Chroma — searches by walking a proximity graph. A selective filter removes most
nodes from consideration, the walk runs out of permitted neighbours, and the traversal
terminates early in a disconnected region. The index reports success because, from its own
point of view, the search completed.

**Why nobody notices.** Standard benchmarks report mean recall. A query that returned 1 of
10 results and a query that returned 10 mediocre ones produce a similar average. The
failure dissolves into the mean, and the shape of it — *how many results came back at
all* — is not measured.

This matters most in exactly the cases teams care about: per-tenant isolation, ACL
enforcement, time-window queries, category filters. Every one of them is a selective
filter over a shared index.

## The approach

`vecverdict` measures retrieval against **exhaustive, unquantised ground truth** and
reports two things that are normally collapsed into one:

| | question it answers |
|---|---|
| **Shortfall** | Did you get back as many results as you asked for? |
| **Recall** | Were the results you got the correct ones? |

Separating these matters, because indexes fail in both directions — and, as the results
below show, an index can score perfectly on one while failing badly on the other.

Three design commitments make the comparison fair:

1. **Ground truth is always exact.** Never another approximate index, which would cap the
   recall any backend could be shown to reach.
2. **Allowlists are never smaller than `k`.** A shortfall is therefore always the index's
   doing, never an arithmetic consequence of a narrow filter.
3. **Every backend sees byte-identical filters**, drawn from a seeded generator, so results
   are reproducible and no backend gets an easier question.

## Results

> **[▶ Explore these results interactively](https://charan-hari.github.io/vecverdict/)** — see a single
> query's results slot by slot, and drag the filter from 100% down to 0.01%.

<p align="center">
  <img src="docs/shortfall_cliff.svg" alt="Results returned vs filter selectivity" width="100%">
</p>

30,000 clustered vectors · dim 128 · k=10 · 50 queries · measured against exhaustive search.
Full data: [`results/demo_synthetic_30k.json`](results/demo_synthetic_30k.json)

### Two different failures

**Failure 1 — the index returns fewer results than requested.**

Results returned, of 10 requested:

| selectivity | allowed | turbovec | faiss-flat | chroma | faiss-hnsw |
|---|---|---|---|---|---|
| 100% | 30,000 | 10.0 | 10.0 | 10.0 | 10.0 |
| 10% | 3,000 | 10.0 | 10.0 | 10.0 | 10.0 |
| 1% | 300 | 10.0 | 10.0 | 10.0 | **5.7** |
| 0.1% | 30 | 10.0 | 10.0 | 10.0 | **0.3** |
| 0.01% | 10 | 10.0 | 10.0 | **9.0** | **0.1** |

At 0.1% selectivity FAISS HNSW returns **0.26 results out of 10**, recall **0.024**. Thirty
vectors were permitted, all thirty were indexed, and exhaustive search retrieved the
correct ten.

**Failure 2 — the index returns a full result set that is quietly wrong.**

This one is more dangerous, because every naive check passes. Chroma returns all 10
results at every selectivity but the narrowest — and the contents degrade anyway:

| selectivity | returned | recall@10 | results that are wrong |
|---|---|---|---|
| 100% | 10.0 / 10 | 0.926 | 7% |
| 10% | 10.0 / 10 | 0.726 | **27%** |
| 1% | 10.0 / 10 | 0.712 | **29%** |
| 0.1% | 10.0 / 10 | 0.654 | **35%** |

A shortfall-only metric would rate Chroma healthy here. A recall-only metric would miss how
completely FAISS HNSW collapses. **Both measurements are necessary**, which is the central
design argument of this tool.

### What this is not

**HNSW is not inherently broken.** Chroma is HNSW-backed and stays within one result of a
full set until the filter reaches 0.01% — nowhere near FAISS HNSW's collapse to 0.3. It
evidently applies filters differently. The honest conclusion is narrower and more useful:
*filtered-search correctness is an implementation choice, and implementations differ
enormously.* You cannot infer it from the algorithm name on the box.

Flat indexes (turbovec, faiss-flat) cannot exhibit failure 1 at all — there is no graph to
disconnect. They scan everything permitted, so `min(k, n_allowed)` results is guaranteed.

### The cost of correctness

Flat indexes are immune to this failure, but scan linearly. That trade-off is real and
worth stating plainly:

| backend | memory | vs flat | p50 latency @1% | filtered behaviour |
|---|---|---|---|---|
| turbovec-4bit | 2.2 MB | **7.2× smaller** | 0.025 ms | full sets, ~10% wrong |
| faiss-flat | 15.6 MB | 1.0× | 1.793 ms | exact |
| chroma | 19.2 MB | 0.8× | 5.857 ms | full sets, 29% wrong |
| faiss-hnsw | 23.3 MB | 0.7× | 0.058 ms | severe shortfall |

At 30k vectors a quantised flat scan is both smaller and faster than the graph. That
advantage narrows as corpora grow — latency is linear in corpus size — which is precisely
why measuring on *your* data at *your* scale is the point of this tool.

## Quick start

```bash
pip install 'vecverdict[turbovec,faiss,viz]'
```

<details>
<summary>Running from a clone or a GitHub Codespace</summary>

Opening this repo in a Codespace builds the devcontainer and installs everything
automatically. From a plain terminal, or any Linux machine:

```bash
bash setup.sh     # installs, lints, runs the test suite
```

</details>

**See the failure reproduce**, on generated data, in under a minute:

```bash
vecverdict demo --charts docs/
```

**Test your own embeddings** — the question that actually matters:

```bash
vecverdict filter --dataset my_vectors.npy --k 10 --charts out/ --json out/run.json
```

**Ask whether compressing your index is worth it:**

```bash
vecverdict switch --vectors my_vectors.npy
```

```
  source           my_vectors.npy
  memory now       5.9 MB
  memory after     0.8 MB (7.5x smaller)
  recall@10        0.878
  recall@1         1.000

  VERDICT: STAY
    - recall@10 of 0.878 is below 0.90: compression loses 12.2% of correct results
    - embeddings whose information is spread evenly across dimensions quantise
      poorly; a higher bit width may recover this
```

That is a real run. It declined a 7.5× memory saving.

## How it works

```
dataset ──► exact ground truth (exhaustive, per allowlist)
                      │
       ┌──────────────┼──────────────┐
   turbovec        faiss          chroma        ◄── native filtering, never post-filtered
       └──────────────┼──────────────┘
                      ▼
         normalise to MISSING-padded (n_queries, k)
                      ▼
          recall · shortfall · attainable ceiling
                      ▼
              JSON results + SVG charts
```

**Result normalisation is the load-bearing part.** Each library signals "no result"
differently: FAISS pads with `-1`, Chroma returns a shorter list, turbovec returns a
narrower array. Every adapter maps onto a single representation, so a missing result stays
visible in the metrics instead of being absorbed by an array shape.

Backends are always queried through their **own native filtering path** — never by
retrieving unfiltered results and discarding them afterwards, which would measure this
tool rather than the index.

### Metrics

| metric | meaning |
|---|---|
| `recall_at_k` | fraction of exact top-k neighbours retrieved |
| `mean_returned` | results actually returned, of `k` requested |
| `mean_shortfall` | **avoidable** missing results — excludes those a narrow filter made impossible |
| `shortfall_rate` | fraction of queries affected |
| `attainable_recall_at_k` | recall against `min(k, n_allowed)` — separates "filter was narrow" from "index failed" |

## Honesty commitments

Constraints on the tool, not marketing copy. Each is enforced by a test.

- **Shortfall is separated from narrowness.** If a filter allows 3 vectors and you asked
  for 10, returning 3 is correct — and is reported as correct.
- **Lossy migrations are refused by default.** Re-quantising vectors reconstructed from a
  FAISS PQ index stacks two lossy codecs and understates the target. `switch` declines
  rather than publishing a number it cannot stand behind. Override with `--allow-lossy`.
- **"Stay" is a supported verdict.** A tool that always recommends migration is an
  advertisement.
- **Published results carry no identifying data.** CPU model, core count and RAM are
  recorded because they make numbers interpretable. Usernames, hostnames and filesystem
  paths are never collected, so they cannot leak.

## Scope and limitations

Stated plainly, because a benchmark that hides its limits is not worth trusting.

- Results above are **synthetic clustered data at 30k vectors** — directional, not yet
  publishable. Real embeddings at scale are the next milestone.
- Only **FAISS, Chroma and turbovec** are measured so far. Qdrant, Weaviate and pgvector
  are the most valuable additions, since they are the most widely deployed.
- HNSW behaviour depends on `M` and `efSearch`; only one configuration is swept today.
  A higher `efSearch` will recover some of the shortfall, at a latency cost — quantifying
  that trade-off is open work.
- Latency figures are single-threaded and indicative, not tuned benchmarks.

## Development

```bash
git clone https://github.com/Charan-Hari/vecverdict
cd vecverdict
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -e '.[all,dev]'
pytest -q            # 140 passing, 3 skipped
ruff check src tests
```

Regenerate the published results and the demo site's data:

```bash
vecverdict site      # writes results/*.json, docs/data/*.json and docs/*.svg
```

`tests/test_site.py` fails if the site's JSON drifts from `results/`, so the page cannot
show numbers the tool did not produce.

## Contributing

The most useful contributions, in order:

1. **A backend adapter** — Qdrant, Weaviate or pgvector, implementing
   [`Backend`](src/vecverdict/backends/base.py) and passing the shared conformance suite.
2. **A counterexample** — a configuration where these findings do not reproduce. That is
   more valuable than a confirmation.
3. **Results from real corpora**, with the dataset named so others can reproduce them.

## Acknowledgements

Built to evaluate [turbovec](https://github.com/RyanCodrai/turbovec)'s exact allowlist
filtering, which prompted the broader question this tool answers. The filtered-search
discussion in
[turbovec#112](https://github.com/RyanCodrai/turbovec/issues/112) measured latency and
memory under filtering; this project measures the correctness side of the same question.

## Licence

MIT
