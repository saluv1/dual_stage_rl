"""
Composed backup policy pi_b for the BCBF construction.

    pi_b(x) = pi_B(x)   if x in B      (certified LQR base controller)
              pi_SA(x)  otherwise      (learned safe-arrival policy, Phase I)

The Phase I actor is a PyTorch MLP (backup_policy/td3.py: Linear-ReLU-Linear-
ReLU-Linear-tanh, widths 10 -> 128 -> 128 -> 4).  We re-implement the forward
pass in JAX and load the trained weights, because the BCBF sensitivity needs
d(pi_b)/dx and Phase II wants a jittable, differentiable pipeline (qpax).

--------------------------------------------------------------------------
IMPORTANT - Lipschitz continuity of pi_b
--------------------------------------------------------------------------
Eq. (8) of the paper defines pi_b with a *hard* switch at the boundary of B.
That makes the closed-loop vector field f_pi_b discontinuous there, so the
flow sensitivity Psi = dPhi/dx is not classically defined for trajectories
that cross dB (you would need a saltation matrix).  Every BCBF result
(Chen et al. [13], Thm. 1-3) assumes a locally Lipschitz backup policy.

So `mode="blend"` is the default here: pi_B is used *exactly* on B (so base-set
invariance is untouched), and pi_SA is blended in over a thin shell
h_B in [-eps, 0) with a C^1 smoothstep.  `mode="hard"` reproduces Eq. (8)
literally if you want to compare.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .barriers import h_B, reduced_state
from .dynamics_jax import G_ACC


# ----------------------------------------------------------------------
# actuator limits (App. F.2)
# ----------------------------------------------------------------------

U_MIN = jnp.array([0.0, -18.0, -18.0, -18.0])
U_MAX = jnp.array([4.0 * G_ACC, 18.0, 18.0, 18.0])


class ActorParams(NamedTuple):
    """Weights of the Phase I TD3 actor, JAX arrays."""
    W1: jnp.ndarray
    b1: jnp.ndarray
    W2: jnp.ndarray
    b2: jnp.ndarray
    W3: jnp.ndarray
    b3: jnp.ndarray
    max_action: float = 1.0


class BackupContext(NamedTuple):
    """Everything pi_b needs.  A pytree, so it can be closed over / jitted."""
    K: jnp.ndarray            # (4, 7) LQR gain
    P: jnp.ndarray            # (7, 7) Riccati matrix
    u_star: jnp.ndarray       # (4,)   = [g, 0, 0, 0]
    actor: ActorParams
    c_b: float = 8.0
    z_des: float = 2.0
    z_ceil: float = 3.0
    g_acc: float = G_ACC
    u_min: jnp.ndarray = U_MIN
    u_max: jnp.ndarray = U_MAX


# ----------------------------------------------------------------------
# loading the Phase I checkpoint
# ----------------------------------------------------------------------

def load_actor_params(path, max_action=1.0):
    """
    Read models/td3_safe_arrival_actor (a torch state_dict) into ActorParams.

    torch stores nn.Linear weight as (out, in); we transpose to (in, out) so
    the JAX forward pass is a plain `x @ W + b`.
    """
    import torch  # local import: Phase II runtime does not need torch

    sd = torch.load(path, map_location="cpu")

    def get(name):
        return jnp.asarray(sd[name].detach().cpu().numpy())

    return ActorParams(
        W1=get("l1.weight").T, b1=get("l1.bias"),
        W2=get("l2.weight").T, b2=get("l2.bias"),
        W3=get("l3.weight").T, b3=get("l3.bias"),
        max_action=max_action,
    )


def random_actor_params(key, state_dim=10, action_dim=4, width=128, max_action=1.0):
    """Randomly initialized actor with the same shapes - for tests only."""
    k1, k2, k3 = jax.random.split(key, 3)
    scale = 0.1
    return ActorParams(
        W1=scale * jax.random.normal(k1, (state_dim, width)), b1=jnp.zeros(width),
        W2=scale * jax.random.normal(k2, (width, width)),     b2=jnp.zeros(width),
        W3=scale * jax.random.normal(k3, (width, action_dim)), b3=jnp.zeros(action_dim),
        max_action=max_action,
    )


# ----------------------------------------------------------------------
# policies
# ----------------------------------------------------------------------

def actor_forward(actor: ActorParams, x):
    """Normalized action in [-1, 1]^4."""
    a = jax.nn.relu(x @ actor.W1 + actor.b1)
    a = jax.nn.relu(a @ actor.W2 + actor.b2)
    return actor.max_action * jnp.tanh(a @ actor.W3 + actor.b3)


def scale_action(a_norm, g_acc=G_ACC):
    """
    JAX version of train.scale_action:
        a_cmd in [0, 4g],  omega in [-18, 18]^3.
    """
    a_cmd = 2.0 * g_acc * (a_norm[0] + 1.0)
    omega = 18.0 * a_norm[1:4]
    return jnp.concatenate([jnp.array([a_cmd]), omega])


def pi_SA(x, ctx: BackupContext):
    """Learned safe-arrival policy, in physical units."""
    return scale_action(actor_forward(ctx.actor, x), ctx.g_acc)


def pi_B(x, ctx: BackupContext):
    """Certified LQR base controller, clipped to U."""
    e = reduced_state(x, ctx.z_des)
    u = ctx.u_star - ctx.K @ e
    return jnp.clip(u, ctx.u_min, ctx.u_max)


def _smoothstep(s):
    s = jnp.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def pi_b(x, ctx: BackupContext, mode="blend", eps=0.5):
    """
    Composed backup policy.

    mode="blend" : sigma = 1 for h_B >= 0, 0 for h_B <= -eps, C^1 in between.
    mode="hard"  : literal Eq. (8) switch.
    """
    hb = h_B(x, ctx.P, ctx.c_b, ctx.z_des)

    u_base = pi_B(x, ctx)
    u_arr = pi_SA(x, ctx)

    if mode == "hard":
        sigma = jnp.where(hb >= 0.0, 1.0, 0.0)
    elif mode == "blend":
        sigma = _smoothstep((hb + eps) / eps)
    else:
        raise ValueError(f"unknown mode: {mode}")

    return sigma * u_base + (1.0 - sigma) * u_arr


def make_closed_loop(ctx: BackupContext, mode="blend", eps=0.5):
    """Return f_pi_b(x) = f(x) + g(x) pi_b(x)."""
    from .dynamics_jax import f_x, g_x

    def f_cl(x):
        return f_x(x, ctx.g_acc) + g_x(x) @ pi_b(x, ctx, mode, eps)

    return f_cl
