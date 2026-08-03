"""Official-style PS2-RL backup-CBF A,b construction for the quadrotor.

The public entry point is :class:`ABConstraintBuilder`.

For a state x, the builder rolls out the fixed composed backup policy

    pi_b(x) = pi_B(x),  x in B
              pi_SA(x), otherwise

and constructs the finite-mesh affine constraints

    A_BCBF(x) u <= b_BCBF(x).

The implementation follows the released PS2-RL code path:

* forward-Euler backup state rollout with quaternion renormalization;
* sensitivity propagation Psi_{k+1} = exp(dt J_k) Psi_k;
* the relative-time correction -f_pi_b(x_i) in every safe-set row;
* one terminal LQR-base-set row;
* hard action-box rows and the optional slack-QP matrices.

The default ``base_pair_mode='official'`` uses the released paper code's
forward-Euler DLQR convention. ``'phase1'`` is retained only for old checkpoints.  A trained backup actor
must always be paired with the same base pair used during its Phase-I training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from scipy.linalg import solve_discrete_are


STATE_DIM = 10
ACTION_DIM = 4


class ActorParams(NamedTuple):
    W1: jax.Array
    b1: jax.Array
    W2: jax.Array
    b2: jax.Array
    W3: jax.Array
    b3: jax.Array
    max_action: float


@dataclass(frozen=True)
class ABConfig:
    """Official quadrotor Phase-I and BCBF configuration.

    The default uses the released forward-Euler DLQR, gravity 9.81, and the
    same base-set/action conventions as the packaged Phase-I trainer.
    ``base_pair_mode="phase1"`` is retained only for old local checkpoints.
    """

    dt: float = 0.02
    num_steps: int = 100
    gravity: float = 9.81

    z_max: float = 3.0
    z_des: float = 2.0

    a_cmd_min: float = 0.0
    a_cmd_max: float | None = None
    omega_max: float = 18.0

    alpha_s: float = 4.0
    alpha_b: float = 2.0
    base_set_c: float = 8.0

    lqr_q_z: float = 1.0
    lqr_q_vx: float = 0.16
    lqr_q_vy: float = 0.16
    lqr_q_vz: float = 0.4
    lqr_q_thetax: float = 0.8
    lqr_q_thetay: float = 0.8
    lqr_q_thetaz: float = 0.16

    lqr_r_a_cmd: float = 0.02
    lqr_r_omega_x: float = 0.012
    lqr_r_omega_y: float = 0.012
    lqr_r_omega_z: float = 0.004

    control_weight: float = 1.0
    slack_weight: float = 1e6

    include_relative_time_term: bool = True
    sensitivity_clip: float = 1e6
    constraint_row_normalize: bool = True
    constraint_row_scale_floor: float = 1.0
    max_abs_constraint_value: float = 1e6

    # ``official`` is the packaged default. ``phase1`` supports legacy local
    # checkpoints trained before the official DLQR alignment.
    base_pair_mode: str = "official"

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.gravity <= 0.0:
            raise ValueError("gravity must be positive")
        if self.z_des >= self.z_max:
            raise ValueError("z_des must be lower than z_max")
        if self.omega_max <= 0.0:
            raise ValueError("omega_max must be positive")
        if self.resolved_a_cmd_max <= self.a_cmd_min:
            raise ValueError("a_cmd_max must be greater than a_cmd_min")
        if self.base_pair_mode not in {"phase1", "official"}:
            raise ValueError("base_pair_mode must be 'phase1' or 'official'")

    @property
    def horizon(self) -> float:
        return self.dt * self.num_steps

    @property
    def resolved_a_cmd_max(self) -> float:
        return 4.0 * self.gravity if self.a_cmd_max is None else float(self.a_cmd_max)

    @property
    def action_low(self) -> np.ndarray:
        return np.array(
            [self.a_cmd_min, -self.omega_max, -self.omega_max, -self.omega_max],
            dtype=float,
        )

    @property
    def action_high(self) -> np.ndarray:
        return np.array(
            [self.resolved_a_cmd_max, self.omega_max, self.omega_max, self.omega_max],
            dtype=float,
        )

    @property
    def num_bcbf_rows(self) -> int:
        # One safe-set condition at every node, plus one terminal base row.
        return self.num_steps + 2

    @property
    def num_hard_rows(self) -> int:
        return self.num_bcbf_rows + 2 * ACTION_DIM

    @property
    def num_slack_qp_rows(self) -> int:
        return self.num_bcbf_rows + 2 * ACTION_DIM + 1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["a_cmd_max"] = self.resolved_a_cmd_max
        payload["horizon"] = self.horizon
        payload["num_bcbf_rows"] = self.num_bcbf_rows
        payload["num_hard_rows"] = self.num_hard_rows
        payload["num_slack_qp_rows"] = self.num_slack_qp_rows
        return payload


def _normalize_quaternion(q: jax.Array) -> jax.Array:
    q = jnp.asarray(q)
    norm = jnp.linalg.norm(q)
    identity = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=q.dtype)
    return jnp.where(norm > jnp.asarray(1e-12, dtype=q.dtype), q / norm, identity)


def _quaternion_rate_matrix(q: jax.Array) -> jax.Array:
    """Xi(q) such that q_dot = 0.5 Xi(q) omega."""
    qw, qx, qy, qz = q
    return jnp.array(
        [
            [-qx, -qy, -qz],
            [qw, -qz, qy],
            [qz, qw, -qx],
            [-qy, qx, qw],
        ],
        dtype=q.dtype,
    )


def _thrust_axis_world(q: jax.Array) -> jax.Array:
    """Third column of the body-to-world rotation matrix."""
    qw, qx, qy, qz = q
    return jnp.array(
        [
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
        dtype=q.dtype,
    )


def _safe_arrival_observation(x: jax.Array, z_des: float, obs_dim: int) -> jax.Array:
    """Encode state for either the new official 10-D or legacy 8-D actor."""
    q = _normalize_quaternion(x[6:10])
    if obs_dim == 10:
        return jnp.concatenate([x[:6], q])
    if obs_dim == 8:
        sign = jnp.where(q[0] < 0.0, -1.0, 1.0).astype(x.dtype)
        q = sign * q
        return jnp.concatenate([x[2:3] - jnp.asarray(z_des, dtype=x.dtype), x[3:6], q])
    raise ValueError(f"Unsupported actor observation width: {obs_dim}")

def load_actor_params(checkpoint: str | Path, *, max_action: float = 1.0) -> ActorParams:
    """Load the compact ``best.pt``/``last.pt`` format or a raw actor state dict."""
    import torch

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Safe-arrival checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "actor" in payload:
        state_dict = payload["actor"]
        max_action = float(payload.get("max_action", max_action))
    elif isinstance(payload, dict) and "l1.weight" in payload:
        state_dict = payload
    else:
        raise KeyError(
            "Expected a compact TD3 checkpoint containing key 'actor', or a raw "
            "actor state_dict with key 'l1.weight'."
        )

    def array(name: str) -> jax.Array:
        value = state_dict[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return jnp.asarray(np.asarray(value), dtype=jnp.float32)

    # PyTorch Linear stores (out, in); JAX forward below uses x @ W.
    params = ActorParams(
        W1=array("l1.weight").T,
        b1=array("l1.bias"),
        W2=array("l2.weight").T,
        b2=array("l2.bias"),
        W3=array("l3.weight").T,
        b3=array("l3.bias"),
        max_action=max_action,
    )

    if params.W1.shape[0] not in (8, 10) or params.W3.shape[1] != ACTION_DIM:
        raise ValueError(
            "The Phase-I actor must map an 8-D legacy or 10-D official observation "
            f"to 4 actions; got W1 {params.W1.shape}, W3 {params.W3.shape}."
        )
    return params


def zero_actor_params(width: int = 128, obs_dim: int = 10) -> ActorParams:
    """Deterministic zero-output actor for smoke tests only."""
    return ActorParams(
        W1=jnp.zeros((obs_dim, width), dtype=jnp.float32),
        b1=jnp.zeros((width,), dtype=jnp.float32),
        W2=jnp.zeros((width, width), dtype=jnp.float32),
        b2=jnp.zeros((width,), dtype=jnp.float32),
        W3=jnp.zeros((width, ACTION_DIM), dtype=jnp.float32),
        b3=jnp.zeros((ACTION_DIM,), dtype=jnp.float32),
        max_action=1.0,
    )


def _actor_forward(params: ActorParams, obs: jax.Array) -> jax.Array:
    hidden = jax.nn.relu(obs @ params.W1 + params.b1)
    hidden = jax.nn.relu(hidden @ params.W2 + params.b2)
    return params.max_action * jnp.tanh(hidden @ params.W3 + params.b3)


def _phase1_lqr(cfg: ABConfig) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct ``bcbf.lqrgain.LQRGain`` used by the local Phase-I trainer."""
    dt = float(cfg.dt)
    g = float(cfg.gravity)
    ad = np.array(
        [
            [1, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, g * dt, 0],
            [0, 0, 1, 0, -g * dt, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    bd = np.array(
        [
            [0.5 * dt**2, 0, 0, 0],
            [0, 0, 0.5 * g * dt**2, 0],
            [0, -0.5 * g * dt**2, 0, 0],
            [dt, 0, 0, 0],
            [0, dt, 0, 0],
            [0, 0, dt, 0],
            [0, 0, 0, dt],
        ],
        dtype=float,
    )
    return _solve_lqr(ad, bd, cfg)


def _official_lqr(cfg: ABConfig) -> tuple[np.ndarray, np.ndarray]:
    """Released PS2-RL forward-Euler hover DLQR convention."""
    g = float(cfg.gravity)
    a_cont = np.zeros((7, 7), dtype=float)
    a_cont[0, 3] = 1.0
    a_cont[1, 5] = -g
    a_cont[2, 4] = g

    b_cont = np.zeros((7, 4), dtype=float)
    b_cont[3, 0] = 1.0
    b_cont[4, 1] = -1.0
    b_cont[5, 2] = -1.0
    b_cont[6, 3] = -1.0

    ad = np.eye(7) + cfg.dt * a_cont
    bd = cfg.dt * b_cont
    return _solve_lqr(ad, bd, cfg)


def _solve_lqr(ad: np.ndarray, bd: np.ndarray, cfg: ABConfig) -> tuple[np.ndarray, np.ndarray]:
    q = np.diag(
        [
            cfg.lqr_q_z,
            cfg.lqr_q_vx,
            cfg.lqr_q_vy,
            cfg.lqr_q_vz,
            cfg.lqr_q_thetax,
            cfg.lqr_q_thetay,
            cfg.lqr_q_thetaz,
        ]
    )
    r = np.diag(
        [
            cfg.lqr_r_a_cmd,
            cfg.lqr_r_omega_x,
            cfg.lqr_r_omega_y,
            cfg.lqr_r_omega_z,
        ]
    )
    p = solve_discrete_are(ad, bd, q, r)
    p = 0.5 * (p.real + p.real.T)
    k = np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad)
    return k, p


class ABConstraintBuilder:
    """Construct BCBF, actuator, and slack-QP matrices for one state."""

    def __init__(
        self,
        actor_params: ActorParams,
        cfg: ABConfig = ABConfig(),
    ) -> None:
        self.cfg = cfg
        self.actor_params = actor_params

        if cfg.base_pair_mode == "phase1":
            k_matrix, p_matrix = _phase1_lqr(cfg)
        else:
            k_matrix, p_matrix = _official_lqr(cfg)

        self.K = jnp.asarray(k_matrix, dtype=jnp.float32)
        self.P = jnp.asarray(p_matrix, dtype=jnp.float32)
        self.u_star = jnp.asarray([cfg.gravity, 0.0, 0.0, 0.0], dtype=jnp.float32)
        self.u_low = jnp.asarray(cfg.action_low, dtype=jnp.float32)
        self.u_high = jnp.asarray(cfg.action_high, dtype=jnp.float32)

        self._make_functions()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        cfg: ABConfig = ABConfig(),
    ) -> "ABConstraintBuilder":
        return cls(load_actor_params(checkpoint), cfg)

    @classmethod
    def for_smoke_test(cls, cfg: ABConfig = ABConfig()) -> "ABConstraintBuilder":
        return cls(zero_actor_params(), cfg)

    def _make_functions(self) -> None:
        cfg = self.cfg
        actor = self.actor_params
        k_matrix = self.K
        p_matrix = self.P
        u_star = self.u_star
        u_low = self.u_low
        u_high = self.u_high

        def reduced_state(x: jax.Array) -> jax.Array:
            x = jnp.asarray(x)
            q = _normalize_quaternion(x[6:10])
            if cfg.base_pair_mode == "phase1":
                # Local Phase-I convention: 2 * sign(qw) * q_vector.
                sign = jnp.where(q[0] < 0.0, -1.0, 1.0).astype(x.dtype)
                theta = 2.0 * sign * q[1:4]
            else:
                # Official convention: q_err = conjugate(q), then shortest sign.
                q_err = jnp.concatenate([q[0:1], -q[1:4]])
                sign = jnp.where(q_err[0] >= 0.0, 1.0, -1.0).astype(x.dtype)
                theta = 2.0 * sign * q_err[1:4]
            return jnp.concatenate(
                [
                    x[2:3] - jnp.asarray(cfg.z_des, dtype=x.dtype),
                    x[3:6],
                    theta,
                ]
            )

        def h_s(x: jax.Array) -> jax.Array:
            return jnp.asarray(cfg.z_max, dtype=x.dtype) - x[2]

        def h_b(x: jax.Array) -> jax.Array:
            error = reduced_state(x)
            p = jnp.asarray(p_matrix, dtype=error.dtype)
            return jnp.asarray(cfg.base_set_c, dtype=error.dtype) - error @ p @ error

        grad_h_s = jax.grad(h_s)
        grad_h_b = jax.grad(h_b)

        actor_obs_dim = int(actor.W1.shape[0])

        def actor_action_norm(x: jax.Array) -> jax.Array:
            obs = _safe_arrival_observation(x, cfg.z_des, actor_obs_dim)
            return _actor_forward(actor, obs)

        def safe_arrival_action(x: jax.Array) -> jax.Array:
            action_norm = jnp.clip(actor_action_norm(x), -1.0, 1.0)
            a_cmd = 2.0 * cfg.gravity * (action_norm[0] + 1.0)
            omega = cfg.omega_max * action_norm[1:4]
            return jnp.clip(jnp.concatenate([a_cmd[None], omega]), u_low, u_high)

        def lqr_action(x: jax.Array) -> jax.Array:
            error = reduced_state(x)
            u = jnp.asarray(u_star, dtype=error.dtype) - jnp.asarray(k_matrix, dtype=error.dtype) @ error
            return jnp.clip(u, u_low, u_high)

        def backup_action(x: jax.Array) -> jax.Array:
            # Literal Eq. (8) / released code hard handoff.
            return jax.lax.select(h_b(x) >= 0.0, lqr_action(x), safe_arrival_action(x))

        def control_affine_terms(x: jax.Array) -> tuple[jax.Array, jax.Array]:
            x = jnp.asarray(x)
            q = _normalize_quaternion(x[6:10])
            f = jnp.zeros((STATE_DIM,), dtype=x.dtype)
            f = f.at[0:3].set(x[3:6])
            f = f.at[5].set(-jnp.asarray(cfg.gravity, dtype=x.dtype))
            g = jnp.zeros((STATE_DIM, ACTION_DIM), dtype=x.dtype)
            g = g.at[3:6, 0].set(_thrust_axis_world(q))
            g = g.at[6:10, 1:4].set(0.5 * _quaternion_rate_matrix(q))
            return f, g

        def dynamics(x: jax.Array, u: jax.Array) -> jax.Array:
            u = jnp.clip(u, u_low, u_high)
            f, g = control_affine_terms(x)
            return f + g @ u

        def closed_loop(x: jax.Array) -> jax.Array:
            return dynamics(x, backup_action(x))

        jac_closed_loop = jax.jacfwd(closed_loop)

        def rollout(x0: jax.Array) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
            x0 = jnp.asarray(x0)
            x0 = x0.at[6:10].set(_normalize_quaternion(x0[6:10]))
            dt = jnp.asarray(cfg.dt, dtype=x0.dtype)
            identity = jnp.eye(STATE_DIM, dtype=x0.dtype)
            clip_enabled = cfg.sensitivity_clip > 0.0
            clip_value = jnp.asarray(cfg.sensitivity_clip, dtype=x0.dtype)
            eps = jnp.asarray(1e-12, dtype=x0.dtype)

            def scan_step(carry, _):
                x, psi, max_abs_psi, saturated = carry
                f_pi = closed_loop(x)
                jac = jac_closed_loop(x)
                x_next = x + dt * f_pi
                x_next = x_next.at[6:10].set(_normalize_quaternion(x_next[6:10]))
                psi_pred = jsp.linalg.expm(dt * jac) @ psi
                step_max = jnp.max(jnp.abs(psi_pred))
                max_abs_next = jnp.maximum(max_abs_psi, step_max)
                if clip_enabled:
                    fro = jnp.linalg.norm(psi_pred, ord="fro")
                    scale = jnp.minimum(1.0, clip_value / (fro + eps))
                    psi_next = psi_pred * scale
                    saturated_next = saturated | (step_max >= clip_value)
                else:
                    psi_next = psi_pred
                    saturated_next = saturated
                return (x_next, psi_next, max_abs_next, saturated_next), (x_next, psi_next)

            init = (
                x0,
                identity,
                jnp.max(jnp.abs(identity)),
                jnp.asarray(False),
            )
            (_, _, max_abs, saturated), (xs_tail, psis_tail) = jax.lax.scan(
                scan_step, init, xs=None, length=cfg.num_steps
            )
            xs = jnp.concatenate([x0[None, :], xs_tail], axis=0)
            psis = jnp.concatenate([identity[None, :, :], psis_tail], axis=0)
            return xs, psis, {
                "max_abs_psi": max_abs,
                "psi_saturated": saturated,
            }

        def bcbf_rows(x: jax.Array):
            x = jnp.asarray(x)
            x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = x.at[6:10].set(_normalize_quaternion(x[6:10]))
            xs, psis, rollout_info = rollout(x)
            f0, g0 = control_affine_terms(x)

            def safety_row(xi: jax.Array, psi_i: jax.Array):
                grad = grad_h_s(xi)
                psi_g = psi_i @ g0
                psi_f = psi_i @ f0
                a = -(grad @ psi_g)
                flow_term = psi_f
                if cfg.include_relative_time_term:
                    flow_term = flow_term - closed_loop(xi)
                b = cfg.alpha_s * h_s(xi) + grad @ flow_term
                return a, b

            a_safe, b_safe = jax.vmap(safety_row)(xs, psis)

            x_terminal = xs[-1]
            psi_terminal = psis[-1]
            grad_terminal = grad_h_b(x_terminal)
            a_terminal = -(grad_terminal @ (psi_terminal @ g0))
            b_terminal = (
                cfg.alpha_b * h_b(x_terminal)
                + grad_terminal @ (psi_terminal @ f0)
            )

            a_rows = jnp.concatenate([a_safe, a_terminal[None, :]], axis=0)
            b_rows = jnp.concatenate([b_safe, b_terminal[None]], axis=0)
            info = {
                **rollout_info,
                "states": xs,
                "sensitivities": psis,
                "h_s_rollout": jax.vmap(h_s)(xs),
                "h_b_terminal": h_b(x_terminal),
                "backup_action": backup_action(x),
                "safe_arrival_action": safe_arrival_action(x),
                "lqr_action": lqr_action(x),
                "in_base_set": h_b(x) >= 0.0,
            }
            return a_rows, b_rows, info

        def hard_rows(x: jax.Array):
            a_bcbf, b_bcbf, info = bcbf_rows(x)
            # Released implementation uses interleaved upper/lower box rows.
            rows = []
            bounds = []
            for i in range(ACTION_DIM):
                high_row = jnp.zeros((ACTION_DIM,), dtype=a_bcbf.dtype).at[i].set(1.0)
                low_row = jnp.zeros((ACTION_DIM,), dtype=a_bcbf.dtype).at[i].set(-1.0)
                rows.extend([high_row, low_row])
                bounds.extend([u_high[i], -u_low[i]])
            a_box = jnp.stack(rows, axis=0)
            b_box = jnp.stack(bounds, axis=0)
            return (
                jnp.concatenate([a_bcbf, a_box], axis=0),
                jnp.concatenate([b_bcbf, b_box], axis=0),
                info,
            )

        def derivative_margins(x: jax.Array, u: jax.Array) -> jax.Array:
            """Direct Eq. (6) LHS; must equal b_BCBF - A_BCBF u."""
            x = jnp.asarray(x)
            x = x.at[6:10].set(_normalize_quaternion(x[6:10]))
            u = jnp.asarray(u, dtype=x.dtype)
            xs, psis, _ = rollout(x)
            f0, g0 = control_affine_terms(x)
            current_vector = f0 + g0 @ u

            def safety_margin(xi: jax.Array, psi_i: jax.Array):
                grad = grad_h_s(xi)
                flow = psi_i @ current_vector
                if cfg.include_relative_time_term:
                    flow = flow - closed_loop(xi)
                return grad @ flow + cfg.alpha_s * h_s(xi)

            safe_margins = jax.vmap(safety_margin)(xs, psis)
            x_terminal = xs[-1]
            psi_terminal = psis[-1]
            terminal_margin = (
                grad_h_b(x_terminal) @ (psi_terminal @ current_vector)
                + cfg.alpha_b * h_b(x_terminal)
            )
            return jnp.concatenate([safe_margins, terminal_margin[None]], axis=0)

        self.reduced_state_jax = jax.jit(reduced_state)
        self.h_s_jax = jax.jit(h_s)
        self.h_b_jax = jax.jit(h_b)
        self.grad_h_s_jax = jax.jit(grad_h_s)
        self.grad_h_b_jax = jax.jit(grad_h_b)
        self.actor_action_norm_jax = jax.jit(actor_action_norm)
        self.safe_arrival_action_jax = jax.jit(safe_arrival_action)
        self.lqr_action_jax = jax.jit(lqr_action)
        self.backup_action_jax = jax.jit(backup_action)
        self.control_affine_terms_jax = jax.jit(control_affine_terms)
        self.dynamics_jax = jax.jit(dynamics)
        self.closed_loop_jax = jax.jit(closed_loop)
        self.closed_loop_jacobian_jax = jax.jit(jac_closed_loop)
        self.rollout_jax = jax.jit(rollout)
        self.bcbf_rows_jax = jax.jit(bcbf_rows)
        self.hard_rows_jax = jax.jit(hard_rows)
        self.derivative_margins_jax = jax.jit(derivative_margins)

    def backup_action(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(self.backup_action_jax(jnp.asarray(state, dtype=jnp.float32)))

    def rollout(self, state: np.ndarray):
        xs, psis, info = self.rollout_jax(jnp.asarray(state, dtype=jnp.float32))
        return np.asarray(xs), np.asarray(psis), {key: np.asarray(value) for key, value in info.items()}

    def compute_bcbf_rows(self, state: np.ndarray):
        """Return the 102 BCBF rows for the paper defaults."""
        a, b, info = self.bcbf_rows_jax(jnp.asarray(state, dtype=jnp.float32))
        return np.asarray(a), np.asarray(b), {key: np.asarray(value) for key, value in info.items()}

    def compute_hard_constraints(self, state: np.ndarray):
        """Return ``A,b`` including the eight hard actuator rows."""
        a, b, info = self.hard_rows_jax(jnp.asarray(state, dtype=jnp.float32))
        return np.asarray(a), np.asarray(b), {key: np.asarray(value) for key, value in info.items()}

    def compute_qp_matrices(self, state: np.ndarray, u_reference: np.ndarray):
        """Return official-style slack-QP matrices for z=[u, delta].

        The inequalities are ``G z <= h``.  Only BCBF rows receive the shared
        slack ``-delta``; actuator bounds and ``delta >= 0`` remain hard.
        """
        a_rows, b_rows, info = self.compute_bcbf_rows(state)
        dtype = a_rows.dtype
        n_rows = a_rows.shape[0]

        g_cbf = np.concatenate([a_rows, -np.ones((n_rows, 1), dtype=dtype)], axis=1)
        h_cbf = b_rows.copy()
        if self.cfg.constraint_row_normalize:
            scales = np.maximum(
                self.cfg.constraint_row_scale_floor,
                np.max(np.abs(g_cbf), axis=1, keepdims=True),
            )
            g_cbf = g_cbf / scales
            h_cbf = h_cbf / scales[:, 0]

        if self.cfg.max_abs_constraint_value > 0.0:
            limit = self.cfg.max_abs_constraint_value
            g_cbf = np.clip(g_cbf, -limit, limit)
            h_cbf = np.clip(h_cbf, -limit, limit)

        box_rows = []
        box_bounds = []
        for i in range(ACTION_DIM):
            high = np.zeros((ACTION_DIM + 1,), dtype=dtype)
            high[i] = 1.0
            box_rows.append(high)
            box_bounds.append(self.cfg.action_high[i])

            low = np.zeros((ACTION_DIM + 1,), dtype=dtype)
            low[i] = -1.0
            box_rows.append(low)
            box_bounds.append(-self.cfg.action_low[i])

        slack_nonnegative = np.zeros((ACTION_DIM + 1,), dtype=dtype)
        slack_nonnegative[-1] = -1.0
        box_rows.append(slack_nonnegative)
        box_bounds.append(0.0)

        g = np.concatenate([g_cbf, np.asarray(box_rows, dtype=dtype)], axis=0)
        h = np.concatenate([h_cbf, np.asarray(box_bounds, dtype=dtype)], axis=0)

        q_matrix = np.diag(
            np.asarray(
                [self.cfg.control_weight] * ACTION_DIM + [self.cfg.slack_weight],
                dtype=dtype,
            )
        )
        u_reference = np.asarray(u_reference, dtype=dtype)
        u_reference = np.nan_to_num(u_reference, nan=0.0, posinf=0.0, neginf=0.0)
        u_reference = np.clip(u_reference, self.cfg.action_low, self.cfg.action_high)
        q_vector = -np.asarray(
            [self.cfg.control_weight * value for value in u_reference] + [0.0],
            dtype=dtype,
        )
        a_equal = np.zeros((0, ACTION_DIM + 1), dtype=dtype)
        b_equal = np.zeros((0,), dtype=dtype)
        return q_matrix, q_vector, a_equal, b_equal, g, h, info

    def compute_normalized_action_constraints(self, state: np.ndarray):
        """Convert physical-action rows to the local [-1,1]^4 convention."""
        a, b, info = self.compute_hard_constraints(state)
        scale = 0.5 * (self.cfg.action_high - self.cfg.action_low)
        offset = 0.5 * (self.cfg.action_high + self.cfg.action_low)
        a_norm = a * scale[None, :]
        b_norm = b - a @ offset
        return a_norm, b_norm, info

    def direct_derivative_margins(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.derivative_margins_jax(
                jnp.asarray(state, dtype=jnp.float32),
                jnp.asarray(action, dtype=jnp.float32),
            )
        )


__all__ = [
    "ABConfig",
    "ABConstraintBuilder",
    "ActorParams",
    "load_actor_params",
    "zero_actor_params",
]
