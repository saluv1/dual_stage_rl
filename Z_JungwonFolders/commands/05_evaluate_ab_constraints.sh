#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$PROJECT_ROOT"
ALLOW_SMOKE_ACTOR="${ALLOW_SMOKE_ACTOR:-0}"
if [[ "$ALLOW_SMOKE_ACTOR" == "1" ]]; then
  RUN_INDEX="${RUN_INDEX:-0}"
  RUN_TAG="000"
else
  RUN_INDEX="${RUN_INDEX:-$(latest_run_index)}"
  RUN_TAG="$(printf '%03d' "$((10#$RUN_INDEX))")"
fi
CHECKPOINT="${CHECKPOINT:-best}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/Trained Models/$RUN_TAG/checkpoints/$CHECKPOINT.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/evaluation/A-B Constraint Evaluation/run_${RUN_TAG}_${CHECKPOINT}}"
ARGS=(
  --output-dir "$OUTPUT_DIR"
  --base-pair-mode "official"
  --gravity "9.81"
  --dt "${DT:-0.02}"
  --num-steps "${NUM_STEPS:-100}"
  --seed "${SEED:-0}"
  --states-per-region "${STATES_PER_REGION:-3}"
)
[[ -n "${STATE:-}" ]] && ARGS+=(--state "$STATE")
if [[ -f "$MODEL_PATH" ]]; then
  ARGS+=(--checkpoint "$MODEL_PATH")
elif [[ "$ALLOW_SMOKE_ACTOR" == "1" ]]; then
  echo "WARNING: using zero actor for matrix-only smoke validation."
  ARGS+=(--allow-smoke-actor)
else
  echo "ERROR: checkpoint not found: $MODEL_PATH" >&2
  exit 2
fi
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python3 -m evaluation.evaluate_ab_constraints "${ARGS[@]}"
