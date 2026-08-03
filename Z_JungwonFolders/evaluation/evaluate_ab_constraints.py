"""Validate the PS2-RL quadrotor backup-CBF A(x), b(x) implementation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize

from backup_policy.phase1.sampling import load_official_sampler, sample_initial_state
from backup_policy.phase1.state_action import normalize_quat
from backup_policy.td3 import Actor, infer_actor_obs_dim, safe_arrival_obs
from bcbf.lqrgain import LQRGain
from bcbf.official_ab import ABConfig, ABConstraintBuilder
from bcbf.set_indicator import SetIndicator

REGIONS = ("general_trace", "near_ceiling", "bridge", "base_shell")


def parse_state(text: str | None) -> np.ndarray:
    if text is None:
        state = np.array([0.0, 0.0, 2.60, 1.5, 0.0, -0.5, 0.9659258, 0.0, 0.2588190, 0.0])
    else:
        values = [float(v.strip()) for v in text.replace("[", "").replace("]", "").split(",") if v.strip()]
        if len(values) != 10:
            raise ValueError(f"--state requires 10 values, got {len(values)}")
        state = np.asarray(values, dtype=float)
    state[6:10] = normalize_quat(state[6:10])
    return state


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_torch_actor(checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = payload["actor"] if isinstance(payload, dict) and "actor" in payload else payload
    obs_dim = infer_actor_obs_dim(state_dict)
    max_action = float(payload.get("max_action", 1.0)) if isinstance(payload, dict) else 1.0
    actor = Actor(obs_dim, 4, max_action)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor, obs_dim


def actor_parity_error(builder, checkpoint: Path, states: np.ndarray) -> float:
    actor, obs_dim = load_torch_actor(checkpoint)
    obs = safe_arrival_obs(states, obs_dim=obs_dim)
    with torch.no_grad():
        torch_action = actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()
    jax_action = np.stack([np.asarray(builder.actor_action_norm_jax(x.astype(np.float32))) for x in states])
    return float(np.max(np.abs(torch_action - jax_action)))


def finite_difference_jacobian(builder, state: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    jac = np.zeros((10, 10))
    for j in range(10):
        xp, xm = state.copy(), state.copy()
        xp[j] += eps
        xm[j] -= eps
        jac[:, j] = (
            np.asarray(builder.closed_loop_jax(xp.astype(np.float32)))
            - np.asarray(builder.closed_loop_jax(xm.astype(np.float32)))
        ) / (2.0 * eps)
    return jac


def terminal_flow_sensitivity_error(builder, state: np.ndarray, eps: float = 1e-3) -> tuple[float, float]:
    _, psi, _ = builder.rollout(state)
    expected = psi[-1]
    fd = np.zeros((10, 10))
    for j in range(10):
        xp, xm = state.copy(), state.copy()
        xp[j] += eps
        xm[j] -= eps
        yp = builder.rollout(xp)[0][-1]
        ym = builder.rollout(xm)[0][-1]
        fd[:, j] = (yp - ym) / (2.0 * eps)
    # Initial quaternions are normalized before rollout, so the radial qw
    # perturbation at the identity is intentionally projected out. Compare the
    # physically meaningful tangent columns instead.
    columns = [0, 1, 2, 3, 4, 5, 7, 8, 9]
    expected_tangent = expected[:, columns]
    fd_tangent = fd[:, columns]
    abs_error = float(np.max(np.abs(expected_tangent - fd_tangent)))
    rel_error = float(np.linalg.norm(expected_tangent - fd_tangent) / max(1.0, np.linalg.norm(fd_tangent)))
    return abs_error, rel_error


def build_regional_states(builder, cfg: ABConfig, seed: int, states_per_region: int):
    sets = SetIndicator(P=np.asarray(builder.P), c_b=cfg.base_set_c, zceil=cfg.z_max)
    sampler = load_official_sampler(split="test")
    rng = np.random.default_rng(seed)
    output = []
    for region in REGIONS:
        if region not in sampler.region_names:
            continue
        for index in range(states_per_region):
            state, actual = sample_initial_state(
                sets=sets,
                regions=sampler,
                s=1.0,
                rng=rng,
                return_region=True,
                force_region=region,
            )
            output.append((actual, index, state))
    return output


def solve_slack_qp(q_matrix, q_vector, g, h, initial_u, n_bcbf_rows):
    q_matrix = np.asarray(q_matrix, dtype=np.float64)
    q_vector = np.asarray(q_vector, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    u0 = np.asarray(initial_u, dtype=np.float64)

    # Rows may be normalized, so the slack coefficient is not always -1.
    slack_coeff = np.maximum(-g[:n_bcbf_rows, 4], 1e-12)
    needed = max(
        0.0,
        float(np.max((g[:n_bcbf_rows, :4] @ u0 - h[:n_bcbf_rows]) / slack_coeff)),
    )
    z0 = np.concatenate([u0, [needed + 1e-4]]).astype(np.float64)

    # Positive rescaling preserves the QP minimizer and improves SLSQP
    # conditioning when the shared-slack weight is 1e6.
    objective_scale = max(1.0, float(np.max(np.abs(q_matrix))))
    def objective(z):
        return float((0.5 * z @ q_matrix @ z + q_vector @ z) / objective_scale)
    def gradient(z):
        return np.asarray((q_matrix @ z + q_vector) / objective_scale, dtype=np.float64)

    result = minimize(
        objective, z0, jac=gradient, method="SLSQP",
        constraints={
            "type": "ineq",
            "fun": lambda z: np.asarray(h - g @ z, dtype=np.float64),
            "jac": lambda z: np.asarray(-g, dtype=np.float64),
        },
        options={"maxiter": 2000, "ftol": 1e-12, "disp": False},
    )
    max_residual = float(np.max(g @ result.x - h))
    return result, max_residual

def save_plots(plot_dir: Path, margins: np.ndarray, h_s: np.ndarray, h_b: np.ndarray, dt: float):
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(np.arange(len(margins)), margins, linewidth=1.3)
    ax.axhline(0.0, linestyle="--")
    ax.axvline(len(margins) - 1, linestyle=":")
    ax.set_xlabel("Constraint row")
    ax.set_ylabel(r"$b_i-A_i\pi_b(x)$")
    ax.set_title("BCBF affine-row margins")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "constraint_margins.png", dpi=180)
    plt.close(fig)

    time = np.arange(len(h_s)) * dt
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(time, h_s, label=r"$h_S$")
    ax.plot(time, h_b, label=r"$h_B$")
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Backup rollout time [s]")
    ax.set_ylabel("Barrier value")
    ax.set_title("Barriers along composed-backup rollout")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "backup_rollout_barriers.png", dpi=180)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--allow-smoke-actor", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("evaluation/A-B Constraint Evaluation/smoke"))
    p.add_argument("--state")
    p.add_argument("--base-pair-mode", choices=("phase1", "official"), default="official")
    p.add_argument("--gravity", type=float, default=9.81)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--states-per-region", type=int, default=3)
    args = p.parse_args()

    if args.checkpoint is None and not args.allow_smoke_actor:
        p.error("Provide --checkpoint or --allow-smoke-actor")
    if args.checkpoint is not None and not args.checkpoint.exists():
        p.error(f"Checkpoint does not exist: {args.checkpoint}")

    output = args.output_dir.resolve()
    data_dir, plot_dir = output / "data", output / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    cfg = ABConfig(
        dt=args.dt, num_steps=args.num_steps, gravity=args.gravity,
        base_pair_mode=args.base_pair_mode,
    )
    builder = (
        ABConstraintBuilder.from_checkpoint(args.checkpoint, cfg)
        if args.checkpoint is not None else ABConstraintBuilder.for_smoke_test(cfg)
    )

    state = parse_state(args.state)
    equilibrium = np.array([0.0, 0.0, cfg.z_des, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    a, b, info = builder.compute_bcbf_rows(state)
    ah, bh, _ = builder.compute_hard_constraints(state)
    u_backup = np.asarray(info["backup_action"])
    qmat, qvec, aeq, beq, gqp, hqp, _ = builder.compute_qp_matrices(state, u_backup)

    margins = b - a @ u_backup
    direct = builder.direct_derivative_margins(state, u_backup)
    affine_abs = float(np.max(np.abs(margins - direct)))
    affine_rel = float(affine_abs / max(1.0, np.max(np.abs(margins)), np.max(np.abs(direct))))

    test_u = np.clip(np.array([1.3 * cfg.gravity, 0.7, -0.4, 0.2]), cfg.action_low, cfg.action_high)
    f0, g0 = builder.control_affine_terms_jax(state.astype(np.float32))
    affine_dyn = np.asarray(f0) + np.asarray(g0) @ test_u
    direct_dyn = np.asarray(builder.dynamics_jax(state.astype(np.float32), test_u.astype(np.float32)))
    dynamics_error = float(np.max(np.abs(affine_dyn - direct_dyn)))

    _, eq_psi, _ = builder.rollout(equilibrium)
    jac0 = np.asarray(builder.closed_loop_jacobian_jax(equilibrium.astype(np.float32)))
    psi1 = np.asarray(jsp.linalg.expm(jnp.asarray(cfg.dt * jac0, dtype=jnp.float32)))
    sensitivity_first_error = float(np.max(np.abs(eq_psi[1] - psi1)))

    fd_jac = finite_difference_jacobian(builder, equilibrium)
    jac_rel = float(np.linalg.norm(jac0 - fd_jac) / max(1.0, np.linalg.norm(fd_jac)))
    flow_abs, flow_rel = terminal_flow_sensitivity_error(builder, equilibrium)

    a_eq_bcbf, b_eq_bcbf, info_eq = builder.compute_bcbf_rows(equilibrium)
    u_eq = np.asarray(info_eq["backup_action"])
    equilibrium_max_residual = float(np.max(a_eq_bcbf @ u_eq - b_eq_bcbf))

    regional_rows = []
    regional_finite = True
    regional_identity_pass = True
    ct_feasible_pass = True
    ct_count = 0
    for region, index, x in build_regional_states(builder, cfg, args.seed, args.states_per_region):
        ai, bi, ii = builder.compute_bcbf_rows(x)
        ui = np.asarray(ii["backup_action"])
        finite = bool(np.all(np.isfinite(ai)) and np.all(np.isfinite(bi)))
        direct_i = builder.direct_derivative_margins(x, ui)
        margin_i = bi - ai @ ui
        identity_error = float(np.max(np.abs(direct_i - margin_i)))
        identity_scale = max(1.0, float(np.max(np.abs(direct_i))), float(np.max(np.abs(margin_i))))
        identity_relative = identity_error / identity_scale
        min_hs = float(np.min(ii["h_s_rollout"]))
        terminal_hb = float(ii["h_b_terminal"])
        in_ct = min_hs >= -1e-5 and terminal_hb >= -1e-5
        max_residual = float(np.max(ai @ ui - bi))
        if in_ct:
            ct_count += 1
            ct_feasible_pass &= max_residual <= 2e-3
        regional_finite &= finite
        regional_identity_pass &= (identity_error <= 3e-3 or identity_relative <= 3e-4)
        regional_rows.append({
            "region": region, "index": index, "finite": finite,
            "affine_identity_max_abs": identity_error,
            "affine_identity_relative": identity_relative,
            "minimum_hs": min_hs, "terminal_hb": terminal_hb,
            "in_backup_induced_set": in_ct,
            "maximum_backup_Au_minus_b": max_residual,
            "max_abs_sensitivity": float(ii["max_abs_psi"]),
            "sensitivity_clipped": bool(ii["psi_saturated"]),
        })

    qp_result, qp_residual = solve_slack_qp(qmat, qvec, gqp, hqp, u_backup, cfg.num_bcbf_rows)

    parity_error = None
    if args.checkpoint is not None:
        parity_states = np.stack([state, equilibrium] + [x for _, _, x in build_regional_states(builder, cfg, args.seed + 1, 1)])
        parity_error = actor_parity_error(builder, args.checkpoint, parity_states)

    k_ref, p_ref = LQRGain(dt=cfg.dt, g=cfg.gravity).gain()
    k_error = float(np.max(np.abs(np.asarray(builder.K) - k_ref)))
    p_error = float(np.max(np.abs(np.asarray(builder.P) - p_ref)))

    checks = {
        "bcbf_shape": {"passed": a.shape == (cfg.num_bcbf_rows, 4)},
        "hard_shape": {"passed": ah.shape == (cfg.num_hard_rows, 4)},
        "slack_qp_shape": {"passed": gqp.shape == (cfg.num_slack_qp_rows, 5)},
        "main_values_finite": {"passed": bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(gqp)) and np.all(np.isfinite(hqp)))},
        "affine_row_identity": {"passed": affine_abs <= 3e-3 or affine_rel <= 3e-4, "max_abs": affine_abs, "relative": affine_rel},
        "control_affine_identity": {"passed": dynamics_error <= 2e-6, "max_abs": dynamics_error},
        "sensitivity_first_step": {"passed": sensitivity_first_error <= 2.5e-4, "max_abs": sensitivity_first_error},
        "closed_loop_jacobian": {"passed": jac_rel <= 1e-3, "relative": jac_rel},
        "terminal_flow_sensitivity": {"passed": flow_rel <= 1e-1, "max_abs": flow_abs, "relative": flow_rel, "note": "diagnostic comparison: matrix-exponential sensitivity versus finite-difference forward-Euler rollout; quaternion radial column excluded"},
        "equilibrium_backup_feasible": {"passed": equilibrium_max_residual <= 1e-4, "maximum_Au_minus_b": equilibrium_max_residual},
        "regional_values_finite": {"passed": regional_finite, "count": len(regional_rows)},
        "regional_affine_identity": {"passed": regional_identity_pass, "count": len(regional_rows)},
        "backup_feasible_on_sampled_C_T": {"passed": ct_feasible_pass, "count_in_C_T": ct_count},
        "slack_qp_solved": {"passed": bool(qp_result.success and qp_residual <= 1e-5), "solver_success": bool(qp_result.success), "maximum_Gz_minus_h": qp_residual, "slack": float(qp_result.x[-1])},
    }
    if parity_error is not None:
        checks["pytorch_jax_actor_parity"] = {"passed": parity_error <= 2e-6, "max_abs": parity_error}
    checks["official_lqr_reconstruction"] = {"passed": k_error <= 2e-5 and p_error <= 2e-4, "K_max_abs": k_error, "P_max_abs": p_error}

    overall = all(bool(v["passed"]) for v in checks.values())

    rollout_states = np.asarray(info["states"])
    h_s = np.asarray(info["h_s_rollout"])
    h_b = np.asarray([builder.h_b_jax(x.astype(np.float32)) for x in rollout_states], dtype=float)
    np.savez_compressed(
        data_dir / "constraint_matrices.npz",
        state=state, A_bcbf=a, b_bcbf=b, A_hard=ah, b_hard=bh,
        Q_qp=qmat, q_qp=qvec, Aeq_qp=aeq, beq_qp=beq,
        G_qp=gqp, h_qp=hqp, qp_solution=qp_result.x,
        backup_action=u_backup, backup_rollout_states=rollout_states,
        sensitivities=np.asarray(info["sensitivities"]),
        h_s_rollout=h_s, h_b_rollout=h_b,
        affine_margins=margins, direct_margins=direct,
    )
    with (data_dir / "regional_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(regional_rows[0].keys()))
        writer.writeheader()
        writer.writerows(regional_rows)
    save_plots(plot_dir, margins, h_s, h_b, cfg.dt)

    summary = {
        "overall_passed": overall,
        "actor_source": "checkpoint" if args.checkpoint else "zero_actor_smoke_test_only",
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "configuration": cfg.to_dict(),
        "checks": checks,
        "regional_checks": regional_rows,
        "matrix_shapes": {"A_bcbf": list(a.shape), "A_hard": list(ah.shape), "G_qp": list(gqp.shape)},
        "outputs": {
            "matrices": str((data_dir / "constraint_matrices.npz").resolve()),
            "regional_checks": str((data_dir / "regional_checks.csv").resolve()),
            "plots": str(plot_dir.resolve()),
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_value(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("=" * 72)
    print("PS2-RL A,b constraint validation")
    for name, result in checks.items():
        print(f"  {'PASS' if result['passed'] else 'FAIL'}  {name}")
    print(f"Output: {output}")
    print("OVERALL:", "PASS" if overall else "FAIL")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
