#!/usr/bin/env bash
# One-shot setup for a fresh Codespace / Linux box.
#
#   bash setup.sh
#
# The devcontainer runs the same install automatically; this script exists for
# plain terminals where no devcontainer was built.

set -euo pipefail

PY="${PYTHON:-python3}"

echo "==> Python: $($PY --version)"

# Editable install so src/ edits take effect without reinstalling.
echo "==> Installing vecverdict and backends"
$PY -m pip install --upgrade pip --quiet
$PY -m pip install -e '.[turbovec,faiss,chroma,viz,dev]'

echo "==> Linting"
$PY -m ruff check src tests

echo "==> Running tests"
$PY -m pytest -q

cat <<'EOF'

Setup complete. Try:

  vecverdict demo                       # reproduce the published finding
  vecverdict filter --n 30000 --dim 128 # full selectivity sweep
  vecverdict switch --vectors mine.npy  # should you switch to turbovec?

Charts are written to docs/ and results to results/.
EOF
