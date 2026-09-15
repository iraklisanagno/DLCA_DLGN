#!/usr/bin/env bash
# Rebuild the paper from its frozen local evidence, without running experiments.
set -euo pipefail
PAPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PAPER_DIR"
PAPER_PYTHON="$PAPER_DIR/../../../venv/bin/python"
export XDG_CACHE_HOME="$PAPER_DIR/.cache"
export XDG_CONFIG_HOME="$PAPER_DIR/.cache/config"
export TMPDIR="$PAPER_DIR/.cache/tmp"
"$PAPER_PYTHON" -B scripts/render_effects.py
"$PAPER_PYTHON" -B scripts/render_paper.py > output/render.log
.tools/tectonic --only-cached --keep-logs --keep-intermediates output/figures/factorial_effects.tex > output/figures/build.log 2>&1
.tools/tectonic --only-cached --keep-logs --keep-intermediates output/main.tex > output/build.stdout.log 2>&1
pdfinfo output/main.pdf > output/pdfinfo.txt
pdffonts output/main.pdf > output/pdffonts.txt
pdftotext -layout output/main.pdf output/main.txt
printf 'Built %s/output/main.pdf\n' "$PAPER_DIR"
