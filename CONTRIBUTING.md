# Contributing

The useful contributions, roughly in order of value:

1. **A backend adapter.** Qdrant, Weaviate and pgvector are the most valuable, because
   they are the most widely deployed and therefore the most consequential if they exhibit
   either failure mode.
2. **A counterexample.** A configuration where the published findings do *not* reproduce
   is worth more than another confirmation. If HNSW holds up somewhere it shouldn't, that
   is a finding.
3. **Results from real corpora.** Name the dataset and embedding model so anyone can
   reproduce them.

## Adding a backend

Implement [`Backend`](src/vecverdict/backends/base.py) and register it in
`src/vecverdict/backends/__init__.py`. The shared conformance suite in
`tests/test_backends.py` will pick it up automatically and hold it to the same contract as
every other adapter.

Two rules matter more than the rest:

- **Use the backend's own filtering path.** Never fetch unfiltered results and discard the
  disallowed ones afterwards — that measures this tool rather than the index, and would
  make every backend look correct.
- **Never invent a result to fill a slot.** If the backend finds four neighbours, report
  four and leave the rest `MISSING`. That gap is the measurement.

## Measurement principles

Changes that weaken any of these need a strong argument:

- Ground truth is exhaustive and unquantised. An approximate reference caps the recall any
  backend can be shown to reach.
- Allowlists are never smaller than `k`, so a shortfall is always attributable to the
  index rather than to arithmetic.
- Every backend sees byte-identical, seeded allowlists.
- Avoidable shortfall is distinguished from shortfall a narrow filter made unavoidable.

## Running the suite

```bash
pip install -e '.[all,dev]'
pytest -q
ruff check src tests
```

## Results and privacy

Result JSONs are published, so they must stay free of identifying data. Machine metadata
is collected in exactly one place — `src/vecverdict/environment.py` — and the fields listed
there are the complete set that can ever be written. Hardware facts make numbers
interpretable and are kept; usernames, hostnames and filesystem paths are never collected.

If you add a field to a result file, add a corresponding assertion to
`test_serialised_result_contains_no_identifying_data`.

## Reporting a finding

Include the dataset, dimension, corpus size, `k`, backend versions and configuration
(`M`/`efSearch` for HNSW). Without those, a result cannot be reproduced or argued with.
