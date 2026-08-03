#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$PROJECT_ROOT"
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python3 -m compileall -q backup_policy bcbf desired_trajectory env evaluation official_phase1_evaluation tests
python3 -m unittest discover -s tests -p 'test_*.py' -v
ALLOW_SMOKE_ACTOR=1 STATES_PER_REGION=1 NUM_STEPS=4 \
OUTPUT_DIR="$PROJECT_ROOT/evaluation/A-B Constraint Evaluation/smoke_validation" \
  ./commands/05_evaluate_ab_constraints.sh
printf '\nVALIDATION COMPLETE\n'
