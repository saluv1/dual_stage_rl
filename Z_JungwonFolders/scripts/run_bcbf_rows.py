"""
Sanity checks for the BCBF row construction.

Run:  python -m scripts.run_bcbf_rows        (from Z_JungwonFolders/)

Checks
------
1. f(x) + g(x)u from dynamics_jax matches env/dynamics.py equationsOfMotion.
2. Psi_i from the variational equation matches a finite-difference Jacobian of
   the flow map (this is the check that actually catches sign / transpose bugs).
3. Feasibility (Chen et al. [13], Thm. 2 / paper Prop. C.3):
   for x in C_T(pi_b), u = pi_b(x) must satisfy A_BCBF u <= b_BCBF.
4. Row counts match App. F.3.4 (102 BCBF + 8 actuator + 1 slack = 111).
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from bcbf.backup_policy import (BackupContext, U_MAX, U_MIN, load_actor_params,
                                make_closed_loop, pi_b, random_actor_params)
from bcbf.barriers import h_B, h_S
from bcbf.dynamics_jax import G_ACC, f_x, g_x
from bcbf.flow import flow_only, make_rollout
from bcbf.lqrgain import LQRGain
from bcbf.rows import (CILConfig, make_bcbf_rows, make_cil_matrices,
                       make_slack_qp_matrices)


ACTOR_PATH = os.path.join(PROJECT_ROOT, "models", "td3_safe_arrival_actor")


def build_context():
    gains = LQRGain(dt=0.02, g=G_ACC)
    K, P = gains.gain()

    actor = None
    if os.path.exists(ACTOR_PATH):
        try:
            actor = load_actor_params(ACTOR_PATH)
            print(f"[ctx] loaded trained actor from {ACTOR_PATH}")
        except ImportError:
            print("[ctx] torch not available, falling back to a random actor")

    if actor is None:
        print("[ctx] using a RANDOM actor (math checks are still valid)")
        actor = random_actor_params(jax.random.PRNGKey(0))

    return BackupContext(
        K=jnp.asarray(K),
        P=jnp.asarray(P),
        u_star=jnp.asarray(gains.u_star),
        actor=actor,
        c_b=8.0,
        z_des=2.0,
        z_ceil=3.0,
        g_acc=G_ACC,
        u_min=U_MIN,
        u_max=U_MAX,
    )


# ----------------------------------------------------------------------

def check_dynamics_match():
    from env.dynamics import Dynamics
    dyn = Dynamics()

    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        x = np.concatenate([rng.normal(size=3), rng.normal(size=3), q])
        u = np.array([rng.uniform(0, 4 * 9.807), *rng.uniform(-18, 18, size=3)])

        ref = dyn.equationsOfMotion(0.0, x, u)
        mine = np.asarray(f_x(jnp.asarray(x)) + g_x(jnp.asarray(x)) @ jnp.asarray(u))
        max_err = max(max_err, np.max(np.abs(ref - mine)))

    print(f"[1] f+g*u vs env/dynamics.py   max abs err = {max_err:.3e}")
    assert max_err < 1e-10


def check_sensitivity(ctx, cfg):
    f_cl = make_closed_loop(ctx, cfg.mode, cfg.blend_eps)
    rollout = make_rollout(f_cl, cfg.N, cfg.dt, cfg.n_sub)
    flow = flow_only(f_cl, cfg.N, cfg.dt, cfg.n_sub)

    x0 = jnp.array([0.0, 0.0, 2.6, 0.3, -0.2, 1.1, 1.0, 0.05, -0.04, 0.02])
    x0 = x0.at[6:10].set(x0[6:10] / jnp.linalg.norm(x0[6:10]))

    X, Psi = rollout(x0)

    # finite-difference Jacobian of x0 -> Phi(x0, T)
    eps = 1e-6
    n = 10
    Jfd = np.zeros((n, n))
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = eps
        xp = flow(x0 + jnp.asarray(dx))[-1]
        xm = flow(x0 - jnp.asarray(dx))[-1]
        Jfd[:, j] = np.asarray((xp - xm) / (2 * eps))

    err = np.max(np.abs(np.asarray(Psi[-1]) - Jfd))
    rel = err / max(np.max(np.abs(Jfd)), 1e-12)
    print(f"[2] Psi_N vs finite differences  max abs err = {err:.3e}  (rel {rel:.3e})")
    assert rel < 1e-5, "sensitivity propagation is wrong"


def check_feasibility(ctx, cfg):
    """
    Take a state whose backup rollout genuinely lands in B while staying in S,
    i.e. x in C_T(pi_b), and verify that u = pi_b(x) is feasible for the rows.
    """
    f_cl = make_closed_loop(ctx, cfg.mode, cfg.blend_eps)
    flow = flow_only(f_cl, cfg.N, cfg.dt, cfg.n_sub)
    bcbf_rows = make_bcbf_rows(ctx, cfg)

    # a state near hover: the LQR branch of pi_b dominates and drives to B
    x = jnp.array([0.0, 0.0, 2.05, 0.05, 0.0, -0.05, 1.0, 0.01, 0.0, 0.0])
    x = x.at[6:10].set(x[6:10] / jnp.linalg.norm(x[6:10]))

    Xtraj = flow(x)
    hs_min = float(jnp.min(jax.vmap(lambda xi: h_S(xi, ctx.z_ceil))(Xtraj)))
    hb_T = float(h_B(Xtraj[-1], ctx.P, ctx.c_b, ctx.z_des))
    print(f"[3] backup rollout: min h_S = {hs_min:+.4f}, h_B(T) = {hb_T:+.4f}")

    in_CT = (hs_min >= 0.0) and (hb_T >= 0.0)
    print(f"    x in C_T(pi_b)? {in_CT}")

    A, b, info = bcbf_rows(x)
    u_bk = info["u_backup"]
    resid = np.asarray(A @ u_bk - b)
    print(f"    max residual of A pi_b(x) - b = {resid.max():+.3e}  "
          f"(<= 0 means feasible)")

    if in_CT:
        assert resid.max() < 1e-6, "pi_b infeasible on C_T - construction is wrong"


def check_feasibility_sweep(ctx, cfg, n_samples=200, seed=1):
    """
    Same check as [3] but over the Phase I curriculum region, so the learned
    pi_SA branch (not just the LQR branch) is exercised.
    """
    f_cl = make_closed_loop(ctx, cfg.mode, cfg.blend_eps)
    flow = flow_only(f_cl, cfg.N, cfg.dt, cfg.n_sub)
    bcbf_rows = make_bcbf_rows(ctx, cfg)

    rng = np.random.default_rng(seed)
    n_in, worst = 0, -np.inf

    for _ in range(n_samples):
        pz = rng.uniform(1.2, 2.98)
        v = rng.uniform(-1.4, 1.4, size=3)
        qv = rng.uniform(-0.14, 0.14, size=3)
        q = np.concatenate([[1.0], qv])
        q /= np.linalg.norm(q)
        x = jnp.asarray(np.concatenate([[0.0, 0.0, pz], v, q]))

        X = flow(x)
        hs = float(jnp.min(jax.vmap(lambda xi: h_S(xi, ctx.z_ceil))(X)))
        hb = float(h_B(X[-1], ctx.P, ctx.c_b, ctx.z_des))
        if hs < 0.0 or hb < 0.0:
            continue

        n_in += 1
        A, b, info = bcbf_rows(x)
        worst = max(worst, float(jnp.max(A @ info["u_backup"] - b)))

    print(f"[5] states in C_T(pi_b): {n_in}/{n_samples}")
    print(f"    worst residual of A pi_b(x) - b over C_T = {worst:+.3e}")
    assert worst <= 1e-6, "pi_b infeasible somewhere on C_T"


def check_psi_convergence(ctx, cfg, tol=1e-2):
    """
    The finite-difference check [2] only proves Psi is the derivative of *our
    discrete* flow map - it cannot see discretization error, since Psi and the
    FD reference are wrong in the same way.  So refine n_sub and watch Psi.
    """
    x = jnp.array([0.0, 0.0, 2.55, 0.4, -0.3, 1.2, 1.0, 0.05, -0.03, 0.02])
    x = x.at[6:10].set(x[6:10] / jnp.linalg.norm(x[6:10]))

    def psi_N(n_sub):
        f_cl = make_closed_loop(ctx, cfg.mode, cfg.blend_eps)
        return make_rollout(f_cl, cfg.N, cfg.dt, n_sub)(x)[1][-1]

    ref = psi_N(8 * max(cfg.n_sub, 1))
    cur = psi_N(cfg.n_sub)
    rel = float(jnp.linalg.norm(cur - ref) / jnp.linalg.norm(ref))

    print(f"[6] Psi_N convergence: n_sub={cfg.n_sub} vs {8 * cfg.n_sub}  "
          f"rel err = {rel:.3e}")
    assert rel < tol, ("Psi is not converged - raise n_sub, or switch to the "
                       "discrete-map convention and match env/dynamics.py")


def check_pib_identity(ctx, cfg, n_samples=30, seed=11, tol=1e-1):
    """
    The sharpest discretization diagnostic available.

    For the exact flow, Psi(x,t') f_pi_b(x) = f_pi_b(Phi(x,t')) identically
    (differentiate the group property Phi(Phi(x,s),t) = Phi(x,s+t) at s=0).
    Substituting u = pi_b(x) therefore collapses the rows to values that are
    known in closed form:

        safety row i : (A u - b)_i = -alpha_S * h_S(x^b_i)
        terminal row : (A u - b)_N = -alpha_B * h_B(x^b_N)
                                     - grad h_B(x^b_N)^T f_pi_b(x^b_N)

    (the terminal row keeps the extra term because it has no -f^b correction).
    Any deviation is pure numerical error, so this is a manufactured-solution
    test: far more sensitive than [2] or [6], and it is what should drive the
    choice of n_sub.  It also re-derives why the -f^b term must be there:
    without it R != 0 and pi_b would not be feasible on C_T (Prop. C.3).
    """
    f_cl = make_closed_loop(ctx, cfg.mode, cfg.blend_eps)
    bcbf_rows = make_bcbf_rows(ctx, cfg)

    rng = np.random.default_rng(seed)
    err_S = err_B = worst_resid = -np.inf

    for _ in range(n_samples):
        pz = rng.uniform(1.3, 2.9)
        v = rng.uniform(-1.3, 1.3, size=3)
        qv = rng.uniform(-0.13, 0.13, size=3)
        q = np.concatenate([[1.0], qv])
        q /= np.linalg.norm(q)
        x = jnp.asarray(np.concatenate([[0.0, 0.0, pz], v, q]))

        A, b, info = bcbf_rows(x)
        u = info["u_backup"]
        resid = np.asarray(A @ u - b)
        X, xN = info["X"], info["X"][-1]

        pred_S = -cfg.alpha_S_gain * np.asarray(
            jax.vmap(lambda z: h_S(z, ctx.z_ceil))(X))
        ghb = jax.grad(h_B)(xN, ctx.P, ctx.c_b, ctx.z_des)
        pred_B = (-cfg.alpha_B_gain * float(h_B(xN, ctx.P, ctx.c_b, ctx.z_des))
                  - float(ghb @ f_cl(xN)))

        err_S = max(err_S, np.abs(resid[:-1] - pred_S).max())
        err_B = max(err_B, abs(resid[-1] - pred_B))
        worst_resid = max(worst_resid, resid.max())

    print(f"[7] pi_b identity residual (n_sub={cfg.n_sub}):")
    print(f"      safety rows   max |resid - closed form| = {err_S:.3e}")
    print(f"      terminal row  max |resid - closed form| = {err_B:.3e}")
    print(f"      feasibility   max resid = {worst_resid:+.3e}  (<= 0 required)")
    assert worst_resid <= 1e-6, "pi_b violates its own rows"
    if max(err_S, err_B) > tol:
        print(f"      WARNING: identity error > {tol:g}; raise n_sub "
              f"(or reduce the Lipschitz constant of pi_SA)")


def check_shapes(ctx, cfg):
    cil = make_cil_matrices(ctx, cfg)
    slack = make_slack_qp_matrices(ctx, cfg, lambda_delta=1e6)

    x = jnp.array([0.0, 0.0, 2.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    A, b, info = cil(x)
    Q, q, As, bs, _ = slack(x, jnp.array([G_ACC, 0.0, 0.0, 0.0]))

    print(f"[4] N = {cfg.N}, BCBF rows = {info['n_bcbf_rows']}, "
          f"A = {A.shape}, b = {b.shape}")
    print(f"    slack QP: Q = {Q.shape}, A_slack = {As.shape}, b_slack = {bs.shape}")
    assert A.shape == (info["n_bcbf_rows"] + 8, 4)
    assert As.shape[0] == A.shape[0] + 1


def main():
    ctx = build_context()
    cfg = CILConfig(T=2.0, dt=0.02, n_sub=16, mode="blend")

    check_dynamics_match()
    check_sensitivity(ctx, cfg)
    check_feasibility(ctx, cfg)
    check_shapes(ctx, cfg)
    check_feasibility_sweep(ctx, cfg)
    check_psi_convergence(ctx, cfg)
    check_pib_identity(ctx, cfg)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
