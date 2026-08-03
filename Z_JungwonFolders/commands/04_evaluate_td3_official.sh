#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$PROJECT_ROOT"
RUN_INDEX="${RUN_INDEX:-$(latest_run_index)}"
CHECKPOINT="${CHECKPOINT:-best}"
RUN_TAG="$(printf '%03d' "$((10#$RUN_INDEX))")"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/Trained Models/$RUN_TAG/checkpoints/$CHECKPOINT.pt}"
RESET_LIBRARY="${RESET_LIBRARY:-$PROJECT_ROOT/official_phase1_evaluation/assets/reset_library.pkl}"
[[ -f "$MODEL_PATH" ]] || { echo "ERROR: checkpoint not found: $MODEL_PATH" >&2; exit 2; }
[[ -f "$RESET_LIBRARY" ]] || { echo "ERROR: reset library not found: $RESET_LIBRARY" >&2; exit 2; }
ARGS=(
  --checkpoint "$MODEL_PATH"
  --reset-library "$RESET_LIBRARY"
  --output-root "$PROJECT_ROOT/evaluation/TD3 Evaluation Official"
  --run-name "run_${RUN_TAG}_${CHECKPOINT}"
  --seed "${SEED:-0}"
  --general-count "${GENERAL_COUNT:-1024}"
  --near-ceiling-count "${NEAR_CEILING_COUNT:-1024}"
  --bridge-count "${BRIDGE_COUNT:-1024}"
  --base-shell-count "${BASE_SHELL_COUNT:-512}"
  --max-per-region "${MAX_PER_REGION:-0}"
  --curriculum-scale "${CURRICULUM_SCALE:-1.0}"
  --horizon-steps "${HORIZON_STEPS:-100}"
  --beta "${BETA:-0.99}"
  --training-gravity "${TRAINING_GRAVITY:-9.81}"
  --overwrite
)
[[ "${REGENERATE_BENCHMARK:-0}" == "1" ]] && ARGS+=(--regenerate-benchmark)
python3 -m official_phase1_evaluation.evaluate_td3 "${ARGS[@]}"
