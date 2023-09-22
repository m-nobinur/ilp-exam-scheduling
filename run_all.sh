#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
rm -f outputs/runs/e1-*.json
$PY -I -B src/data.py
$PY -I -B src/evaluate.py
$PY -I -B src/experiments.py
$PY -I -B src/figures.py
if [ -f report/build_report.py ]; then $PY -I -B report/build_report.py; fi
if [ -f src/checks.py ]; then $PY -I -B src/checks.py; fi
