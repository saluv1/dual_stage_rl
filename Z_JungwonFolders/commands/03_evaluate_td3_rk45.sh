#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$PROJECT_ROOT"
RUN_INDEX="${RUN_INDEX:-$(latest_run_index)}"
CHECKPOINT="${CHECKPOINT:-best}"
python3 -m evaluation.evaluate_td3_rk45 \
  --model-root "Trained Models" \
  --run-index "$RUN_INDEX" \
  --checkpoint "$CHECKPOINT" \
  --episodes-per-region "${EPISODES_PER_REGION:-64}" \
  --seed "${SEED:-1000}" \
  --output-root "evaluation/TD3 Evaluation RK45"
