"""
BCBF constraint rows:  A_BCBF(x) u <= b_BCBF(x).

This is App. E.7.1 of the paper, Eqs. (110)-(119), implemented on the finite
backup mesh 0 = t'_0 < t'_1 < ... < t'_N = T.

For every safe-set component h_{S,j} and every node i:

    a^S_{i,j}(x)^T = - grad h_{S,j}(x^b_i)^T Psi_i(x) g(x)
    b^S_{i,j}(x)   =   alpha_S( h_{S,j}(x^b_i) )
                     + grad h_{S,j}(x^b_i)^T [ Psi_i(x) f(x) - f^b_i(x) ]

and at the terminal node only, for every base-set component h_{B,l}:

    a^B_l(x)^T = - grad h_{B,l}(x^b_N)^T Psi_N(x) g(x)
    b^B_l(x)   =   alpha_B( h_{B,l}(x^b_N) )
                 + grad h_{B,l}(x^b_N)^T Psi_N(x) f(x)

with x^b_i = Phi_pi_b(x, t'_i), Psi_i = dPhi/dx at t'_i, f^b_i = f_pi_b(x^b_i).

The `- f^b_i` term in the safety rows (absent in the terminal row) is the
"the constraint point itself is moving along the backup flow" correction; the
terminal condition is anchored at a fixed horizon T so it has no such term.

Row counts for the quadrotor task (n_S = n_B = 1, T = 2.0 s, dt = 0.02):
    (N + 1) * n_S + n_B = 101 + 1 = 102 BCBF rows
    + 2m = 8 actuator rows                  = 110
    + 1 slack nonnegativity row             = 111
which matches App. F.3.4.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .backup_policy import BackupContext, make_closed_loop, pi_b
from .barriers import grad_h_B, grad_h_S, h_B, h_S, make_linear_alpha
from .dynamics_jax import N_INPUT, f_x, g_x
from .flow import make_rollout


class CILConfig(NamedTuple):
    T: float = 2.0             # backup horizon [s]
    dt: float = 0.02           # mesh spacing [s] (= control period)
    n_sub: int = 16            # RK4 substeps per mesh interval.
    #   NOT a free parameter: ||J_cl|| ~ 100 for the learned pi_b, so
    #   h*||J_cl|| ~ 2 at n_sub=1, right at RK4's stability edge.  Phi
    #   survives that but dPhi/dx does not (68% error on dB-crossing
    #   trajectories).  See scripts/run_bcbf_rows.py check [6].
    alpha_S_gain: float = 4.0  # App. F.3, Table 9
    alpha_B_gain: float = 2.0
    mode: str = "blend"        # pi_b switching: "blend" or "hard"
    blend_eps: float = 0.5
    drop_t0_row: bool = False  # see note below

    @property
    def N(self):
        return int(round(self.T / self.dt))


# ----------------------------------------------------------------------
# static row counts (needed because jit shapes must be concrete)
# ----------------------------------------------------------------------

def n_bcbf_rows(cfg: CILConfig, n_S: int = 1, n_B: int = 1):
    """(N + 1) * n_S + n_B, minus the t'=0 row if it was dropped."""
    n_nodes = cfg.N + 1 - (1 if cfg.drop_t0_row else 0)
    return n_nodes * n_S + n_B


# ----------------------------------------------------------------------
# core row builder
# ----------------------------------------------------------------------

def make_bcbf_rows(ctx: BackupContext, cfg: CILConfig = CILConfig()):
    """
    Returns bcbf_rows(x) -> (A_BCBF, b_BCBF, info).

        A_BCBF : (n_rows, 4)
        b_BCBF : (n_rows,)
        info   : dict with the backup rollout, useful for debugging / plotting

    The returned function is jitted and differentiable in x.
    """
    f_cl = make_closed_loop(ctx, cfg.mode, cfg.blend_eps)
    rollout = make_rollout(f_cl, cfg.N, cfg.dt, cfg.n_sub)

    alpha_S = make_linear_alpha(cfg.alpha_S_gain)
    alpha_B = make_linear_alpha(cfg.alpha_B_gain)

    def bcbf_rows(x):
        X, Psi = rollout(x)                      # (N+1, n), (N+1, n, n)

        f0 = f_x(x, ctx.g_acc)                   # (n,)
        g0 = g_x(x)                              # (n, m)

        # ---- safety rows along the whole backup flow -------------------
        def safety_row(xi, Psi_i):
            gh = grad_h_S(xi, ctx.z_ceil)        # (n,)
            w = gh @ Psi_i                       # (n,)
            a = -(w @ g0)                        # (m,)
            b = alpha_S(h_S(xi, ctx.z_ceil)) + w @ f0 - gh @ f_cl(xi)
            return a, b

        A_S, b_S = jax.vmap(safety_row)(X, Psi)  # (N+1, m), (N+1,)

        # ---- terminal base-set row -------------------------------------
        xN, PsiN = X[-1], Psi[-1]
        ghb = grad_h_B(xN, ctx.P, ctx.c_b, ctx.z_des)
        wN = ghb @ PsiN
        a_B = -(wN @ g0)
        b_B = alpha_B(h_B(xN, ctx.P, ctx.c_b, ctx.z_des)) + wN @ f0

        if cfg.drop_t0_row:
            A_S, b_S = A_S[1:], b_S[1:]

        A = jnp.concatenate([A_S, a_B[None, :]], axis=0)
        b = jnp.concatenate([b_S, jnp.array([b_B])], axis=0)

        info = {
            "X": X,
            "Psi": Psi,
            "h_S_traj": jax.vmap(lambda xi: h_S(xi, ctx.z_ceil))(X),
            "h_B_T": h_B(xN, ctx.P, ctx.c_b, ctx.z_des),
            "u_backup": pi_b(x, ctx, cfg.mode, cfg.blend_eps),
        }
        return A, b, info

    return jax.jit(bcbf_rows)


# ----------------------------------------------------------------------
# actuator rows and full QP assembly
# ----------------------------------------------------------------------

def actuator_rows(ctx: BackupContext):
    """A_U u <= b_U with A_U = [I; -I], b_U = [u_max; -u_min].  Eq. (121)."""
    A_U = jnp.concatenate([jnp.eye(N_INPUT), -jnp.eye(N_INPUT)], axis=0)
    b_U = jnp.concatenate([ctx.u_max, -ctx.u_min])
    return A_U, b_U


def make_cil_matrices(ctx: BackupContext, cfg: CILConfig = CILConfig()):
    """
    Returns cil_matrices(x) -> (A, b, info) for the *hard* QP (Eq. 122):

        min_u ||u - pi_phi(x)||^2   s.t.   A u <= b

    where A stacks [A_BCBF; A_U] and b stacks [b_BCBF; b_U].
    """
    bcbf_rows = make_bcbf_rows(ctx, cfg)
    A_U, b_U = actuator_rows(ctx)

    def cil_matrices(x):
        A_bcbf, b_bcbf, info = bcbf_rows(x)
        A = jnp.concatenate([A_bcbf, A_U], axis=0)
        b = jnp.concatenate([b_bcbf, b_U])
        info["n_bcbf_rows"] = n_bcbf_rows(cfg)
        return A, b, info

    return jax.jit(cil_matrices)


def make_slack_qp_matrices(ctx: BackupContext, cfg: CILConfig = CILConfig(),
                           lambda_delta=1e6):
    """
    Slack-regularized form actually solved in the paper (Eq. 123), with
    decision variable z = [u; delta] in R^{m+1}:

        min_z  0.5 z^T Q z + q^T z
        s.t.   A_slack z <= b_slack

        Q = 2 * blockdiag(I_m, lambda_delta)
        q = -2 * [pi_phi(x); 0]

    Only the BCBF rows get the slack column; actuator limits stay hard, so
    every executed u is inside U regardless of delta.  Row layout:

        [ A_BCBF   -1 ]        [ b_BCBF ]
        [ A_U       0 ]  z <=  [ b_U    ]
        [ 0 ... 0  -1 ]        [ 0      ]

    Returns slack_qp(x, u_nom) -> (Q, q, A_slack, b_slack, info).
    """
    cil_matrices = make_cil_matrices(ctx, cfg)
    m = N_INPUT

    Q = 2.0 * jnp.diag(jnp.concatenate([jnp.ones(m), jnp.array([lambda_delta])]))

    def slack_qp(x, u_nom):
        A, b, info = cil_matrices(x)
        n_bcbf = n_bcbf_rows(cfg)          # static Python int

        slack_col = jnp.concatenate([
            -jnp.ones(n_bcbf),
            jnp.zeros(A.shape[0] - n_bcbf),
        ])[:, None]

        A_slack = jnp.concatenate([
            jnp.concatenate([A, slack_col], axis=1),
            jnp.concatenate([jnp.zeros((1, m)), -jnp.ones((1, 1))], axis=1),
        ], axis=0)

        b_slack = jnp.concatenate([b, jnp.zeros(1)])

        q = -2.0 * jnp.concatenate([u_nom, jnp.zeros(1)])

        return Q, q, A_slack, b_slack, info

    return jax.jit(slack_qp)


# ----------------------------------------------------------------------
# helper: rows in normalized action coordinates
# ----------------------------------------------------------------------

def to_normalized_rows(A, b, ctx: BackupContext):
    """
    If the Phase II actor outputs a_norm in [-1,1]^4 with
    u = scale * a_norm + offset, convert A u <= b into rows on a_norm:

        (A @ diag(scale)) a_norm <= b - A @ offset
    """
    scale = 0.5 * (ctx.u_max - ctx.u_min)
    offset = 0.5 * (ctx.u_max + ctx.u_min)
    return A * scale[None, :], b - A @ offset
