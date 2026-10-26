#!/usr/bin/env bash
# Gera notebooks/*.ipynb a partir de src_notebooks/*.py (jupytext, formato py:percent)
set -euo pipefail
cd "$(dirname "$0")/.."
for f in src_notebooks/*.py; do
  jupytext --quiet --to ipynb --output "notebooks/$(basename "${f%.py}").ipynb" "$f"
done
echo "ok: $(ls notebooks/*.ipynb | wc -l) notebooks"
