#!/usr/bin/env python3
"""Run one PS2-RL SITL trial and keep all artifacts in one directory.

This wrapper does not start PX4 SITL or MicroXRCEAgent.  Start those in their
usual two terminals, then run this script in the bridge terminal.  After the
vehicle lands, press Ctrl-C once; the wrapper stops ``ros2 launch`` and runs the
existing analysis and plotting scripts against the saved flight CSV.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
from typing import Sequence


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_.-")
    if not label:
        raise ValueError("--label must contain at least one filename-safe character")
    return label


def _new_output_dir(root: Path, label: str) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"{stamp}-{label}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}-{label}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _copy_if_present(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=10.0)
    except (OSError, subprocess.TimeoutExpired):
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)


def _run_launch(command: Sequence[str], log_path: Path) -> tuple[int, bool]:
    interrupted = False
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_file.write("COMMAND: " + " ".join(command) + "\n\n")
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
            return_code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            print("\nStopping bridge; post-processing saved flight data...", flush=True)
            _stop_process_group(process)
            return_code = process.returncode if process.returncode is not None else 130
    return int(return_code), interrupted


def _run_postprocess(script: Path, args: list[str], output_log: Path) -> int:
    result = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
    )
    output_log.write_text(
        result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode != 0:
        print(f"Warning: {script.name} failed; see {output_log}", file=sys.stderr)
    return int(result.returncode)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Short run label, e.g. nominal or delay_d5_j2_h5.")
    parser.add_argument("--config", required=True, help="Bridge YAML for this vehicle/policy condition.")
    parser.add_argument("--run-dir", required=True, help="PS2-RL run directory containing configs and weights.")
    parser.add_argument("--checkpoint", default="best", choices=["best", "final"])
    parser.add_argument(
        "--output-root",
        default="/home/dev/ws_shared/experiments/ps2rl_sitl",
        help="Persistent parent directory for timestamped run folders.",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Save launch log and CSV only; do not run analysis/plot scripts.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Create nothing and print the ros2 launch command that would run.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    label = _safe_label(args.label)
    source_root = Path(__file__).resolve().parent
    config = Path(args.config).expanduser().resolve()
    policy_run = Path(args.run_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not config.is_file():
        raise FileNotFoundError(f"Bridge config not found: {config}")
    if not (policy_run / "configs.json").is_file():
        raise FileNotFoundError(f"Policy configs.json not found under: {policy_run}")
    weights = policy_run / f"{args.checkpoint}_weights.pkl"
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    preview_command = [
        "ros2",
        "launch",
        "ps2rl_px4_bridge",
        "ps2rl_sitl.launch.py",
        f"config:={config}",
        f"run_dir:={policy_run}",
        f"checkpoint:={args.checkpoint}",
        "log_csv:=<timestamped-output>/flight.csv",
    ]
    if args.print_only:
        print(" ".join(preview_command))
        return 0

    output_dir = _new_output_dir(output_root, label)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir()
    config_snapshot = output_dir / "bridge_config.yaml"
    shutil.copy2(config, config_snapshot)
    flight_csv = output_dir / "flight.csv"
    bridge_log = output_dir / "bridge.log"

    copied_policy_files = []
    for name in ("configs.json", "summary.json"):
        if _copy_if_present(policy_run / name, output_dir / "policy" / name):
            copied_policy_files.append(name)

    command = [
        "ros2",
        "launch",
        "ps2rl_px4_bridge",
        "ps2rl_sitl.launch.py",
        f"config:={config_snapshot}",
        f"run_dir:={policy_run}",
        f"checkpoint:={args.checkpoint}",
        f"log_csv:={flight_csv}",
    ]
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "label": label,
        "output_dir": str(output_dir),
        "policy_run_dir": str(policy_run),
        "checkpoint": args.checkpoint,
        "checkpoint_file": str(weights),
        "checkpoint_sha256": _sha256(weights),
        "source_config": str(config),
        "config_snapshot": str(config_snapshot),
        "policy_files_copied": copied_policy_files,
        "bridge_source_commit": _git_head(source_root),
        "policy_repo_commit": _git_head(policy_run),
        "launch_command": command,
        "launch_return_code": None,
        "launch_interrupted_by_user": False,
        "postprocess": {},
    }
    metadata_path = output_dir / "run_metadata.json"
    _write_json(metadata_path, metadata)

    print(f"Experiment directory: {output_dir}")
    print("After LAND, press Ctrl-C once to stop the bridge and generate analysis artifacts.")
    return_code, interrupted = _run_launch(command, bridge_log)
    metadata["launch_return_code"] = return_code
    metadata["launch_interrupted_by_user"] = interrupted

    if flight_csv.is_file() and flight_csv.stat().st_size > 0 and not args.skip_postprocess:
        analyze_script = source_root / "analyze_flight.py"
        plot_script = source_root / "plot_trajectory.py"
        metadata["postprocess"]["analyze_return_code"] = _run_postprocess(
            analyze_script,
            [str(flight_csv)],
            output_dir / "analysis.txt",
        )
        metadata["postprocess"]["plot_return_code"] = _run_postprocess(
            plot_script,
            [str(flight_csv), "--out", str(plots_dir / "trajectory.png")],
            output_dir / "plot.log",
        )
    elif not flight_csv.is_file():
        metadata["postprocess"]["skipped_reason"] = "flight.csv was not created"
        print("Warning: flight.csv was not created; inspect bridge.log.", file=sys.stderr)
    else:
        metadata["postprocess"]["skipped_reason"] = "--skip-postprocess"

    _write_json(metadata_path, metadata)
    print(f"Artifacts saved under: {output_dir}")
    # Ctrl-C after a normal landing is the documented way to end ros2 launch,
    # so do not turn that deliberate action into a failed experiment command.
    return 0 if interrupted else return_code


if __name__ == "__main__":
    raise SystemExit(main())
