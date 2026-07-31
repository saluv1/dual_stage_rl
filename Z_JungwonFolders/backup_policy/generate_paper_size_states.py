#!/usr/bin/env python3
"""Generate and freeze the paper-sized PS2-RL Phase-I evaluation states."""
from __future__ import annotations

import argparse
from pathlib import Path

from backup_policy.phase1.official_reset_library import (
    PAPER_SIZE_COUNTS,
    generate_paper_size_evaluation_set,
    load_official_reset_library,
    save_evaluation_state_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 3,584 official-region initial states: 1,024 general, "
            "1,024 near-ceiling, 1,024 bridge, and 512 base-shell."
        )
    )
    parser.add_argument("--reset-library", required=True)
    parser.add_argument(
        "--output",
        default="evaluation_assets/official_ps2rl/paper_size_seed1234.npz",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--curriculum-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_official_reset_library(args.reset_library)
    evaluation_set = generate_paper_size_evaluation_set(
        payload,
        seed=args.seed,
        curriculum_scale=args.curriculum_scale,
        counts=PAPER_SIZE_COUNTS,
    )
    output = save_evaluation_state_file(
        args.output,
        evaluation_set,
        source_reset_library=payload.path,
        seed=args.seed,
        curriculum_scale=args.curriculum_scale,
    )

    print(f"Saved paper-sized benchmark: {output}")
    print(f"Total states: {evaluation_set.states.shape[0]}")
    for region in PAPER_SIZE_COUNTS:
        count = int((evaluation_set.regions == region).sum())
        print(f"  {region}: {count}")


if __name__ == "__main__":
    main()
