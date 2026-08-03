#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$PROJECT_ROOT"
export MPLBACKEND=Agg
STEPS="${STEPS:-1000}"
HOVER_SECONDS="${HOVER_SECONDS:-2.0}"
python3 -m evaluation.run_lqr_circular \
  --steps "$STEPS" \
  --hover-seconds "$HOVER_SECONDS" \
  --output-dir "$PROJECT_ROOT/evaluation/LQR Evaluation"
