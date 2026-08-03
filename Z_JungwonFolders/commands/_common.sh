#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
latest_run_index() {
  local root="${1:-$PROJECT_ROOT/Trained Models}"
  local latest
  latest="$(find "$root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | grep -E '^[0-9]{3}$' | sort -n | tail -1 || true)"
  if [[ -z "$latest" ]]; then
    echo "ERROR: no indexed trained-model folder exists under: $root" >&2
    return 1
  fi
  echo "$((10#$latest))"
}
