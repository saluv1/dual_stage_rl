"""Load a trained PS2-RL Phase-2 checkpoint and run it (through the CIL) online.

This mirrors what ``ps2rl/evaluation/quadrotor_p2_eval.py`` does to rebuild a
run, but for a single state at a time driven by wall-clock time instead of a
batched ``lax.scan`` rollout.

Two things are deliberately kept identical to training, because getting either
wrong silently degrades the policy:

1. **The observation layout.**  ``[x(10), ref_state(10), ref_omega(3), t,
   sin(phase), cos(phase)]`` — 26D with ``include_time_features``.  The
   reference is interpolated at ``t / reference_dt`` and clipped at both ends,
   exactly as ``QuadrotorEnvConfig`` does.

2. **The CIL projection.**  ``eval_action`` runs the actor *and* the BCBF-QP.
   Skipping the projection at deploy time throws away the entire safety
   argument of the method, so there is no option to disable it here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np


def _ensure_ps2rl_importable(ps2rl_path: str | None) -> None:
    if ps2rl_path:
        path = str(Path(ps2rl_path).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


@dataclass
class PolicyOutput:
    u_safe: np.ndarray        # [a_cmd, wx, wy, wz] in ENU/FLU convention
    u_raw: np.ndarray         # actor output before CIL projection
    slack: float              # QP slack: > 0 means the safety row was relaxed
    projected_norm: float     # ||u_safe - u_raw||, how hard the CIL intervened
    latency_s: float


class PS2RLPolicy:
    """Wraps one Phase-2 run directory into a single-state ``act()`` call."""

    def __init__(
        self,
        run_dir: str,
        checkpoint: str = "best",
        ps2rl_path: str | None = None,
        learned_backup_policy_path: str | None = None,
        reference_path: str | None = None,
        jax_platform: str = "cpu",
        single_thread: bool = True,
        logger=None,
    ) -> None:
        self._log = logger
        os.environ.setdefault("JAX_PLATFORMS", jax_platform)

        # The backup CBF rolls its policy forward 100 sequential steps and
        # differentiates through them. Those are many tiny dependent ops, so
        # thread synchronisation costs more than parallelism buys — measured
        # 7.5 ms multi-threaded vs 4.2 ms single-threaded. Threads also fight
        # PX4 and Gazebo for cores on the same machine, which is what turns a
        # 13 ms warm latency into 37 ms once the simulator is busy.
        # Must be set before JAX is first imported.
        if single_thread and jax_platform == "cpu":
            os.environ.setdefault(
                "XLA_FLAGS",
                "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
            )
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ.setdefault(var, "1")

        # Python hands the GIL over only every 5 ms by default. With the
        # executor, the DDS threads and the 125 Hz odometry callback all live
        # in this interpreter, the policy thread can lose tens of milliseconds
        # per control step just waiting its turn. A shorter interval trades a
        # little switching overhead for far less jitter.
        sys.setswitchinterval(0.002)

        _ensure_ps2rl_importable(ps2rl_path)

        import jax
        import jax.numpy as jnp

        from ps2rl.cil.cil_policy import ActorConfig
        from ps2rl.cil.quadrotor_backup_cbf import (
            QuadrotorBCBFConfig,
            QuadrotorBackupCBFProjector,
        )
        from ps2rl.envs.assets.utils import _load_reference_bundle
        from ps2rl.envs.quadrotor_env import QuadrotorEnvConfig
        from ps2rl.phase2_ps2.quadrotor_ps2_trainer import SACConfig, _build_action_fns

        self._jax = jax
        self._jnp = jnp

        run = Path(run_dir).expanduser().resolve()
        cfg_path = self._resolve_config_path(run)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_json = json.load(f)

        self.sac_cfg = self._dataclass_from_dict(SACConfig, cfg_json.get("sac", {}))
        env_overrides: dict[str, Any] = {}
        if reference_path:
            env_overrides["reference_path"] = str(Path(reference_path).expanduser().resolve())
        self.env_cfg = self._dataclass_from_dict(
            QuadrotorEnvConfig, cfg_json.get("env", {}), overrides=env_overrides
        )

        cbf_overrides: dict[str, Any] = {}
        if learned_backup_policy_path:
            cbf_overrides["learned_backup_policy_path"] = str(
                Path(learned_backup_policy_path).expanduser().resolve()
            )
        self.cbf_cfg = self._dataclass_from_dict(
            QuadrotorBCBFConfig, cfg_json.get("cbf", {}), overrides=cbf_overrides
        )

        weights_path = self._resolve_weights_path(run, checkpoint)
        actor_params = self._load_actor_params(weights_path)

        projector = QuadrotorBackupCBFProjector(self.cbf_cfg)
        action_scale = np.array(
            [
                self.cbf_cfg.a_cmd_max,
                self.cbf_cfg.omega_max,
                self.cbf_cfg.omega_max,
                self.cbf_cfg.omega_max,
            ],
            dtype=np.float64,
        )

        # --- observation / reference tables -------------------------------
        bundle = _load_reference_bundle(self.env_cfg.reference_path)
        self._ref_states = np.asarray(bundle["states"], dtype=np.float64)
        self._ref_omega = np.asarray(bundle["omega_cmd"], dtype=np.float64)
        self._ref_dt = float(
            self.env_cfg.reference_dt if self.env_cfg.reference_dt is not None else self.env_cfg.dt
        )
        self._ref_last = max(int(self._ref_states.shape[0]) - 1, 1)
        self.reference_duration = self._ref_last * self._ref_dt
        self.include_time_features = bool(self.env_cfg.include_time_features)

        self.dt = float(self.env_cfg.dt)
        self.z_max = float(self.cbf_cfg.z_max)
        self.a_cmd_max = float(self.cbf_cfg.a_cmd_max)
        self.omega_max = float(self.cbf_cfg.omega_max)

        obs_dim = 23 + (3 if self.include_time_features else 0)
        actor_obs_dim = self._infer_actor_obs_dim(actor_params) or obs_dim
        if actor_obs_dim > obs_dim:
            raise ValueError(
                f"Actor expects obs_dim={actor_obs_dim} but this reference bundle "
                f"only produces {obs_dim}. Wrong checkpoint or wrong reference?"
            )
        self._actor_obs_dim = int(actor_obs_dim)

        actor_cfg = ActorConfig(
            obs_dim=self._actor_obs_dim,
            action_dim=4,
            hidden_sizes=(self.sac_cfg.hidden_size, self.sac_cfg.hidden_size),
        )
        _, eval_action = _build_action_fns(
            self.sac_cfg,
            actor_cfg,
            self.cbf_cfg,
            jnp.asarray(action_scale, dtype=jnp.float32),
            backup_runtime=projector.runtime,
        )
        self._eval_action = eval_action
        self._params = actor_params

        self.entry_state = self._ref_states[0].copy()
        self.exit_state = self._ref_states[-1].copy()

        self._warmup()

    # ------------------------------------------------------------------ #
    # Loading helpers (mirrors quadrotor_vanilla_eval)                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_config_path(run_dir: Path) -> Path:
        for name in ("configs.json", "config.json"):
            candidate = run_dir / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No configs.json/config.json under {run_dir}")

    @staticmethod
    def _resolve_weights_path(run_dir: Path, checkpoint: str) -> Path:
        mapping = {"best": "best_weights.pkl", "final": "final_weights.pkl"}
        name = mapping.get(checkpoint, f"{checkpoint}_weights.pkl")
        path = run_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint {path}")
        return path

    @staticmethod
    def _dataclass_from_dict(cls, payload: dict, overrides: dict | None = None):
        from dataclasses import fields

        valid = {f.name for f in fields(cls) if f.init}
        kwargs = {k: v for k, v in payload.items() if k in valid}
        if overrides:
            kwargs.update({k: v for k, v in overrides.items() if k in valid})
        return cls(**kwargs)

    @staticmethod
    def _load_actor_params(weights_path: Path):
        with open(weights_path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "actor_params" in payload:
            return payload["actor_params"]
        raise KeyError(f"'actor_params' not found in {weights_path}")

    @staticmethod
    def _infer_actor_obs_dim(actor_params: Any) -> int | None:
        if isinstance(actor_params, dict):
            layers = actor_params.get("layers")
            if isinstance(layers, (list, tuple)) and layers:
                first = layers[0]
                if isinstance(first, dict) and "w" in first:
                    w = np.asarray(first["w"])
                    if w.ndim == 2:
                        return int(w.shape[0])
        return None

    # ------------------------------------------------------------------ #
    # Reference + observation                                              #
    # ------------------------------------------------------------------ #

    def reference_at(self, t: float) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Interpolated ``(ref_state, ref_omega, phase_sin, phase_cos)`` at time t."""
        idx = float(t) / self._ref_dt
        idx = min(max(idx, 0.0), float(self._ref_last))
        i0 = int(math.floor(idx))
        i1 = min(i0 + 1, self._ref_last)
        w = idx - i0

        ref = (1.0 - w) * self._ref_states[i0] + w * self._ref_states[i1]
        q = ref[6:10]
        nrm = float(np.linalg.norm(q))
        ref[6:10] = q / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])

        omega = (1.0 - w) * self._ref_omega[i0] + w * self._ref_omega[i1]

        phase = 2.0 * math.pi * (idx / float(self._ref_last))
        return ref, omega, math.sin(phase), math.cos(phase)

    def build_obs(self, x: np.ndarray, t: float) -> np.ndarray:
        ref, omega, s, c = self.reference_at(t)
        parts = [np.asarray(x, dtype=np.float64), ref, omega]
        if self.include_time_features:
            parts.append(np.array([t, s, c], dtype=np.float64))
        obs = np.concatenate(parts)
        return obs[: self._actor_obs_dim].astype(np.float32)

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def act(self, x: np.ndarray, t: float) -> PolicyOutput:
        obs = self.build_obs(x, t)
        started = time.perf_counter()
        safe, raw, slack = self._eval_action(self._params, self._jnp.asarray(obs))
        safe = np.asarray(self._jax.device_get(safe), dtype=np.float64)
        raw = np.asarray(self._jax.device_get(raw), dtype=np.float64)
        slack_f = float(np.asarray(self._jax.device_get(slack)).reshape(-1)[0])
        latency = time.perf_counter() - started

        return PolicyOutput(
            u_safe=safe,
            u_raw=raw,
            slack=slack_f,
            projected_norm=float(np.linalg.norm(safe - raw)),
            latency_s=latency,
        )

    def _warmup(self, iterations: int = 5) -> None:
        """Trigger JIT compilation before the vehicle is in the air."""
        x0 = self.entry_state.copy()
        timings = []
        for _ in range(iterations):
            out = self.act(x0, 0.0)
            timings.append(out.latency_s)
        steady = min(timings)
        msg = (
            f"PS2-RL policy ready | obs_dim={self._actor_obs_dim} dt={self.dt:.4f}s "
            f"z_max={self.z_max:.2f}m ref_duration={self.reference_duration:.3f}s "
            f"backup={self.cbf_cfg.backup_policy_mode} "
            f"warm_latency={steady * 1e3:.2f}ms"
        )
        if self._log is not None:
            self._log.info(msg)
        else:
            print(msg)

        if steady > 0.8 * self.dt:
            warn = (
                f"Policy latency {steady * 1e3:.1f}ms is close to the {self.dt * 1e3:.0f}ms "
                f"control period. Consider JAX_PLATFORMS=cuda or a lower policy rate."
            )
            if self._log is not None:
                self._log.warn(warn)
            else:
                print("WARNING: " + warn)
