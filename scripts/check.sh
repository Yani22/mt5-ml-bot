#!/usr/bin/env bash
# Local stand-in for CI: the same two checks .github/workflows/ci.yml runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
echo "== flake8 =="; "$PY" -m flake8 .
echo "== pytest =="; "$PY" -m pytest -q
echo "All checks passed."
