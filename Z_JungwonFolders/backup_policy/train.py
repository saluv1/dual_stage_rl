import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env.dynamics import Dynamics
from bcbf.lqrgain import LQRGain
from bcbf.set_indicator import SetIndicator
from backup_policy.td3 import TD3
from backup_policy.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Basic state utilities
# ---------------------------------------------------------------------------

def compute_reduced_state(full_state):

    z_des = 2.0

    pz = full_state[2]
    vx = full_state[3]
    vy = full_state[4]
    vz = full_state[5]

    qw = full_state[6]
    qx = full_state[7]
    qy = full_state[8]
    qz = full_state[9]

    if qw < 0.0:
        qx = -qx
        qy = -qy
        qz = -qz

    reduced_state = np.array([
        pz - z_des,
        vx,
        vy,
        vz,
        2.0 * qx,
        2.0 * qy,
        2.0 * qz
    ])

    return reduced_state


def compute_bfc(sets, full_state):

    reduced_state = compute_reduced_state(full_state)

    indicator = sets.compute_indicator(full_state, reduced_state)

    if indicator == 0:
        b = 1.0
        f = 0.0
        c = 0.0
    elif indicator == 2:
        b = 0.0
        f = 1.0
        c = 0.0
    else:
        b = 0.0
        f = 0.0
        c = 1.0

    return b, f, c


def scale_action(action_norm, g):

    action_norm = np.array(action_norm, dtype=float)
    action_norm = np.clip(action_norm, -1.0, 1.0)

    a_cmd = 2.0 * g * (action_norm[0] + 1.0)
    wx = 18.0 * action_norm[1]
    wy = 18.0 * action_norm[2]
    wz = 18.0 * action_norm[3]

    return np.array([a_cmd, wx, wy, wz])


def unscale_action(action_phys, g):
    """Inverse of `scale_action`: physical command -> normalized action."""

    a_cmd = action_phys[0]
    w = action_phys[1:4]

    a0 = a_cmd / (2.0 * g) - 1.0
    a1 = w[0] / 18.0
    a2 = w[1] / 18.0
    a3 = w[2] / 18.0

    return np.clip(np.array([a0, a1, a2, a3]), -1.0, 1.0)


# ---------------------------------------------------------------------------
# LQR base controller (FIX #6: replay-buffer warm start)
# ---------------------------------------------------------------------------

class LQRController:
    """
    Clipped LQR base controller around hover -- the SAME controller that
    certifies the base set B.

    Used only to PREFILL the replay buffer before training starts. The
    safe-arrival reward is an indicator b(x) in {0,1} with no shaping, so if
    the buffer contains almost no successes the critic regresses toward zero
    everywhere and the actor gradient is flat -- there is no signal to ascend,
    not merely a weak one. Measured random-policy arrival was ~35% from easy
    states but only ~3.5% from the hardest ones.

    A one-time prefill is deliberately preferred over mixing LQR episodes into
    ongoing collection: it keeps the behavior distribution stationary, adds no
    annealing schedule to tune, and -- since LQR data stops arriving once
    training begins -- creates no pull toward imitating the analytic backup,
    which the paper's whole contribution is about beating (35.5% -> 69.3%).
    """

    def __init__(self, K, g, z_des=2.0):
        self.K = K
        self.g = g
        self.z_des = z_des
        self.u_star = np.array([g, 0.0, 0.0, 0.0])

    def act_phys(self, state):

        xe = compute_reduced_state(state)
        u = self.u_star - self.K @ xe

        u[0] = np.clip(u[0], 0.0, 4.0 * self.g)
        u[1:4] = np.clip(u[1:4], -18.0, 18.0)

        return u

    def act_norm(self, state):

        return unscale_action(self.act_phys(state), self.g)


def warm_start_buffer(
        replay_buffer,
        lqr,
        sets,
        regions,
        rng,
        n_transitions=100000,
        max_episode_steps=300,
        s_values=(0.0, 0.25, 0.5, 0.75, 1.0),
        action_noise=0.05
):
    """
    Prefill the replay buffer with LQR rollouts spanning ALL curriculum regions.

    Sweeping s across its range (and forcing each region in turn) is the point:
    the cold-start problem is worst in near_ceiling / bridge, so prefilling only
    easy states would leave exactly the gap we are trying to close.

    A little action noise is added so the critic sees more than a single action
    per state -- otherwise Q is trained on a measure-zero slice of the action
    space and extrapolates freely everywhere else.

    Note the buffer is FIFO with capacity 4e5, so this data is evicted after
    roughly 4e5 steps. It is a starting signal, not a permanent one -- by then
    the policy should be generating its own successes.
    """

    dyn = Dynamics()

    eval_regions = [
        "synthetic_capture", "synthetic_mid",
        "trace_general", "near_ceiling", "bridge"
    ]
    eval_regions = [
        r for r in eval_regions
        if r in ["synthetic_capture", "synthetic_mid"]
        or (r in regions and len(regions[r]) > 0)
    ]

    added = 0
    n_success = 0
    n_failure = 0
    n_episodes = 0
    per_region_success = {r: [0, 0] for r in eval_regions}

    while added < n_transitions:

        region = eval_regions[n_episodes % len(eval_regions)]
        s_val = s_values[(n_episodes // len(eval_regions)) % len(s_values)]

        try:
            state, region_name = sample_initial_state(
                sets=sets,
                regions=regions,
                s=s_val,
                rng=rng,
                return_region=True,
                force_region=region
            )
        except RuntimeError:
            n_episodes += 1
            continue

        reset_dynamics_state(dyn, state)
        n_episodes += 1
        per_region_success[region_name][1] += 1

        for step in range(max_episode_steps):

            b_cur, f_cur, c_cur = compute_bfc(sets, state)

            action_norm = lqr.act_norm(state)
            action_norm = np.clip(
                action_norm + rng.normal(0.0, action_noise, size=4),
                -1.0, 1.0
            )

            next_state = dyn.step(scale_action(action_norm, dyn.g)).copy()

            if not np.all(np.isfinite(next_state)):
                break

            b_next, f_next, c_next = compute_bfc(sets, next_state)

            replay_buffer.add(state, action_norm, next_state, b_cur, c_cur)
            added += 1

            success = b_next == 1.0
            failure = f_next == 1.0

            # Same terminal-anchor logic as the training loop (FIX #1).
            if success or failure:
                b_term, f_term, c_term = compute_bfc(sets, next_state)
                replay_buffer.add(
                    next_state, action_norm, next_state, b_term, c_term
                )
                added += 1

            state = next_state.copy()

            if success:
                n_success += 1
                per_region_success[region_name][0] += 1
                break

            if failure:
                n_failure += 1
                break

            if added >= n_transitions:
                break

    print("---------------------------------------")
    print(f"Warm start: {added} transitions from {n_episodes} LQR episodes")
    print(f"  successes: {n_success}, failures: {n_failure}")
    print("  per-region LQR arrival rate:")
    for r, (succ, tot) in per_region_success.items():
        if tot > 0:
            print(f"    {r}: {succ}/{tot} = {succ / tot:.3f}")
    print("---------------------------------------")

    dyn.state = state.copy()

    if hasattr(dyn, "curr_step"):
        dyn.curr_step = 0

    if hasattr(dyn, "xlist"):
        dyn.xlist = []

    if hasattr(dyn, "vlist"):
        dyn.vlist = []

    if hasattr(dyn, "qlist"):
        dyn.qlist = []


def reset_dynamics_state(dyn, state):

    dyn.state = state.copy()

    if hasattr(dyn, "curr_step"):
        dyn.curr_step = 0

    if hasattr(dyn, "xlist"):
        dyn.xlist = []

    if hasattr(dyn, "vlist"):
        dyn.vlist = []

    if hasattr(dyn, "qlist"):
        dyn.qlist = []


def normalize_quat(q):

    q = np.array(q, dtype=float)
    q_norm = np.linalg.norm(q)

    if q_norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])

    return q / q_norm


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def quat_from_axis_angle(axis, angle):

    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-9)

    half = angle / 2.0

    w = np.cos(half)
    xyz = axis * np.sin(half)

    return np.array([w, xyz[0], xyz[1], xyz[2]])


def quat_mult(q1, q2):

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])


# ---------------------------------------------------------------------------
# Barrier / difficulty metrics
# ---------------------------------------------------------------------------

def h_b_value(state, P, c_b):

    xe = compute_reduced_state(state)
    return c_b - xe.T @ P @ xe


def h_s_value(state, zceil):

    return zceil - state[2]


def state_difficulty(state, P, c_b, zceil):

    xe = compute_reduced_state(state)

    hb = c_b - xe.T @ P @ xe
    hs = zceil - state[2]
    vnorm = np.linalg.norm(state[3:6])
    att_err = np.linalg.norm(xe[4:7])

    return hb, hs, vnorm, att_err


# ---------------------------------------------------------------------------
# Continuous curriculum (FIX #3)
# ---------------------------------------------------------------------------
# The previous design used 6 DISCRETE levels whose gates decided which regions
# were REACHABLE. Measured consequence: near_ceiling and bridge states have
# att_err in [1.65, 1.86], while att_max only reaches 1.69 at level 5. Their
# pass rate was EXACTLY 0.000 at levels 0-4 and 1.000 at level 5. The agent
# advanced 0->1->2->3->4 having literally never seen one of these states, then
# at level 5 they became ~50% of the mixture at full radius_scale = 1.00.
# That is not a hard final level -- it is a different task with no warning.
#
# The redesign separates three things that were conflated:
#
#   1. WHICH regions exist        -> all of them, always, via weight floors.
#   2. HOW FAR states are pushed  -> the continuous scalar s in [0,1].
#   3. WHEN to advance            -> weighted mu_SA estimate at T = 100 steps.
#
# s now controls perturbation RADIUS, not membership. Trace regions are fixed
# real states from the reference trajectory, so their difficulty is not
# something s controls -- only the synthetic shells are gated by s.


def gate_bounds(s):
    """
    Continuous gate bounds for the SYNTHETIC regions only.

    Interpolates between the old level-0 and level-5 bounds.
    """

    def lerp(a, b):
        return a + (b - a) * s

    hb_min = lerp(-1.0, -150.0)
    v_max = lerp(1.0, 5.5)
    att_max = lerp(0.30, 2.00)
    hs_min = lerp(0.50, 0.02)

    return hb_min, v_max, att_max, hs_min


def passes_level_gate(state, P, c_b, zceil, s):
    """
    Applied ONLY to synthetic_capture / synthetic_mid, whose difficulty really
    is procedurally controlled by s. Trace regions are curated by
    `classify_trace_states` plus the c == 1 check at the sampling site.
    """

    hb, hs, vnorm, att_err = state_difficulty(state, P, c_b, zceil)

    hb_min, v_max, att_max, hs_min = gate_bounds(s)

    if hb >= 0.0:
        return False

    if hb < hb_min:
        return False

    if hs < hs_min:
        return False

    if vnorm > v_max:
        return False

    if att_err > att_max:
        return False

    return True


# ---------------------------------------------------------------------------
# Analytic power-loop trace placeholder
# ---------------------------------------------------------------------------

def generate_reference_trace(n_points=200, n_variants=20, seed=0):
    """
    Placeholder for the paper's vanilla SAC power-loop tracker traces.

    The paper uses states from unsafe vanilla SAC power-loop rollouts.
    Here we generate analytic power-loop-like states because those traces
    are not available yet.

    Later, replace this with real saved vanilla SAC traces.
    """

    rng = np.random.default_rng(seed)

    center = np.array([0.0, 0.0, 2.0])
    radius = 1.5
    v0 = 4.5

    trace_states = []

    for _variant in range(n_variants):

        variant_center = center + rng.normal(0.0, 0.05, size=3)
        variant_radius = radius + rng.normal(0.0, 0.03)

        thetas = np.linspace(
            -np.pi / 2.0,
            -np.pi / 2.0 + 2.0 * np.pi,
            n_points
        )

        for theta in thetas:

            px = variant_center[0] + variant_radius * (
                np.cos(theta) - np.cos(-np.pi / 2.0)
            )

            py = variant_center[1] + rng.normal(0.0, 0.02)

            pz = variant_center[2] + variant_radius * np.sin(theta)

            speed = v0 + rng.normal(0.0, 0.1)

            vx = speed * (-np.sin(theta))
            vy = rng.normal(0.0, 0.05)
            vz = speed * np.cos(theta)

            flip_angle = theta + np.pi / 2.0
            q = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), flip_angle)

            jitter_axis = rng.normal(0.0, 1.0, size=3)
            jitter_axis = jitter_axis / (np.linalg.norm(jitter_axis) + 1e-9)
            jitter_angle = rng.normal(0.0, np.deg2rad(3.0))

            q_jitter = quat_from_axis_angle(jitter_axis, jitter_angle)

            q = quat_mult(q_jitter, q)
            q = normalize_quat(q)

            state = np.array([
                px,
                py,
                pz,
                vx,
                vy,
                vz,
                q[0],
                q[1],
                q[2],
                q[3]
            ])

            trace_states.append(state)

    return trace_states


def classify_trace_states(
        sets,
        trace_states,
        P,
        c_b,
        near_ceiling_margin=0.25
):
    """
    Classifies analytic or vanilla-SAC trace states into trace regions.

    We do NOT rely on trace capture_shell anymore because the analytic trace
    produced capture_shell = 0 in the failed run.
    """

    general = []
    near_ceiling = []

    for state in trace_states:

        b, f, c = compute_bfc(sets, state)

        if f == 1.0:
            continue

        if b == 1.0:
            continue

        hs = sets.hs

        if hs <= near_ceiling_margin:
            near_ceiling.append(state)
        else:
            general.append(state)

    bridge = []
    n_interp = 20

    for i in range(len(trace_states) - 1):

        s0 = trace_states[i]
        s1 = trace_states[i + 1]

        compute_bfc(sets, s0)
        hs0 = sets.hs

        compute_bfc(sets, s1)
        hs1 = sets.hs

        crosses_boundary = (hs0 >= 0.0) != (hs1 >= 0.0)
        near_apex = (hs0 < near_ceiling_margin) or (hs1 < near_ceiling_margin)

        if not (crosses_boundary or near_apex):
            continue

        for alpha in np.linspace(0.0, 1.0, n_interp):

            interp = (1.0 - alpha) * s0 + alpha * s1
            interp[6:10] = normalize_quat(interp[6:10])

            b, f, c = compute_bfc(sets, interp)

            if f == 1.0:
                continue

            if b == 1.0:
                continue

            bridge.append(interp)

    regions = {
        "trace_general": general,
        "near_ceiling": near_ceiling,
        "bridge": bridge,
    }

    return regions


# ---------------------------------------------------------------------------
# Hybrid curriculum regions
# ---------------------------------------------------------------------------

def get_radius_scale(s):
    """Trace perturbation radius grows smoothly with difficulty s."""

    return 0.10 + 0.90 * s


# Weights used to combine per-region success into the mu_SA estimate.
# These are FIXED and independent of s, so the number is comparable across
# evaluations and across runs. They mirror App. F.3.3, which weights the
# near-ceiling and bridge sub-regions more heavily because those are the
# states that limit downstream powerloop tracking.
MU_SA_WEIGHTS = {
    "synthetic_capture": 0.10,
    "synthetic_mid": 0.15,
    "trace_general": 0.25,
    "near_ceiling": 0.30,
    "bridge": 0.20,
}


def get_curriculum_weights(s):
    """
    Continuous region mixture: a per-region FLOOR plus a triangular bump that
    peaks at that region's own difficulty center.

    The floor is the important part. Under the old discrete table, near_ceiling
    had weight 0.00 for levels 0-2 and bridge 0.00 for levels 0-3, and the easy
    regions fell toward 0.05 at level 5 -- so regions were alternately starved
    and then abruptly dominant, which is the catastrophic-forgetting pattern.
    With a floor, every region is trained from step 0 and none is ever dropped;
    s only shifts the EMPHASIS.

    Hard regions get the largest floors because that is where the paper's whole
    Phase I gain comes from (near-ceiling recoverability 35.5% -> 69.3%,
    Sec. 5.2).
    """

    centers = {
        "synthetic_capture": 0.00,
        "synthetic_mid": 0.15,
        "trace_general": 0.30,
        "near_ceiling": 0.50,
        "bridge": 0.70,
    }

    floors = {
        "synthetic_capture": 0.10,
        "synthetic_mid": 0.10,
        "trace_general": 0.15,
        "near_ceiling": 0.35,
        "bridge": 0.30,
    }

    width = 0.35

    weights = {}

    for name, center in centers.items():
        bump = max(0.0, 1.0 - abs(s - center) / width)
        weights[name] = floors[name] + bump

    return weights


def sample_reduced_shell(P, c_b, delta_min, delta_max, rng):
    """
    Samples reduced state xe such that:

        xe^T P xe = c_b + delta

    Therefore:

        h_B = c_b - xe^T P xe = -delta

    This creates states just outside or moderately outside the base ellipsoid.
    """

    dim = P.shape[0]

    direction = rng.normal(0.0, 1.0, size=dim)
    direction = direction / (np.linalg.norm(direction) + 1e-9)

    denom = direction.T @ P @ direction

    if denom <= 1e-9:
        return None

    delta = rng.uniform(delta_min, delta_max)
    target = c_b + delta

    scale = np.sqrt(target / denom)

    xe = scale * direction

    return xe


def full_state_from_reduced_shell(xe, rng, pxy_mode="near_origin", trace_anchor=None):

    if pxy_mode == "trace" and trace_anchor is not None:
        px = trace_anchor[0] + rng.uniform(-0.25, 0.25)
        py = trace_anchor[1] + rng.uniform(-0.25, 0.25)
    else:
        px = rng.uniform(-0.50, 0.50)
        py = rng.uniform(-0.50, 0.50)

    pz = 2.0 + xe[0]

    vx = xe[1]
    vy = xe[2]
    vz = xe[3]

    qx = 0.5 * xe[4]
    qy = 0.5 * xe[5]
    qz = 0.5 * xe[6]

    qv_norm_sq = qx**2 + qy**2 + qz**2

    if qv_norm_sq >= 0.95:
        return None

    qw = np.sqrt(max(1.0 - qv_norm_sq, 1e-9))

    q = normalize_quat(np.array([qw, qx, qy, qz]))

    state = np.array([
        px,
        py,
        pz,
        vx,
        vy,
        vz,
        q[0],
        q[1],
        q[2],
        q[3]
    ])

    return state


def sample_synthetic_capture(P, c_b, rng):

    xe = sample_reduced_shell(
        P=P,
        c_b=c_b,
        delta_min=0.02,
        delta_max=1.0,
        rng=rng
    )

    if xe is None:
        return None

    return full_state_from_reduced_shell(
        xe=xe,
        rng=rng,
        pxy_mode="near_origin"
    )


def sample_synthetic_mid(P, c_b, rng):

    xe = sample_reduced_shell(
        P=P,
        c_b=c_b,
        delta_min=1.0,
        delta_max=8.0,
        rng=rng
    )

    if xe is None:
        return None

    return full_state_from_reduced_shell(
        xe=xe,
        rng=rng,
        pxy_mode="near_origin"
    )


def perturb_trace_state(base_state, rng, radius_scale):

    pos_radius = 0.4 * radius_scale
    vel_radius = 1.5 * radius_scale
    tilt_radius = np.deg2rad(30.0) * radius_scale
    yaw_radius = np.deg2rad(12.0) * radius_scale

    px, py, pz, vx, vy, vz, qw, qx, qy, qz = base_state

    px += rng.uniform(-pos_radius, pos_radius)
    py += rng.uniform(-pos_radius, pos_radius)
    pz += rng.uniform(-pos_radius, pos_radius)

    vx += rng.uniform(-vel_radius, vel_radius)
    vy += rng.uniform(-vel_radius, vel_radius)
    vz += rng.uniform(-vel_radius, vel_radius)

    tilt_axis_xy = rng.normal(0.0, 1.0, size=2)
    tilt_axis_xy = tilt_axis_xy / (np.linalg.norm(tilt_axis_xy) + 1e-9)

    tilt_angle = rng.uniform(-tilt_radius, tilt_radius)

    q_tilt = quat_from_axis_angle(
        np.array([tilt_axis_xy[0], tilt_axis_xy[1], 0.0]),
        tilt_angle
    )

    yaw_angle = rng.uniform(-yaw_radius, yaw_radius)

    q_yaw = quat_from_axis_angle(
        np.array([0.0, 0.0, 1.0]),
        yaw_angle
    )

    q_base = np.array([qw, qx, qy, qz])
    q_new = quat_mult(q_yaw, quat_mult(q_tilt, q_base))
    q_new = normalize_quat(q_new)

    state = np.array([
        px,
        py,
        pz,
        vx,
        vy,
        vz,
        q_new[0],
        q_new[1],
        q_new[2],
        q_new[3]
    ])

    return state


def sample_trace_region(region_name, regions, rng, radius_scale):

    if region_name not in regions:
        return None

    candidates = regions[region_name]

    if len(candidates) == 0:
        return None

    base_state = candidates[rng.integers(len(candidates))]

    return perturb_trace_state(
        base_state=base_state,
        rng=rng,
        radius_scale=radius_scale
    )


def sample_initial_state(
        sets,
        regions,
        s,
        rng,
        max_tries=10000,
        return_region=False,
        force_region=None
):
    """
    Continuous-curriculum sampler.

    `s` in [0,1] controls perturbation RADIUS (and the synthetic gates), not
    which regions are available -- every region with nonzero weight is
    reachable at every s.

    `force_region` restricts sampling to one region, used by the stratified
    evaluator so each region gets a fixed episode count regardless of mixture.
    """

    weights = get_curriculum_weights(s)

    region_names = []
    region_probs = []

    for name, weight in weights.items():

        if force_region is not None and name != force_region:
            continue

        if weight <= 0.0:
            continue

        if name in ["trace_general", "near_ceiling", "bridge"]:

            if name not in regions:
                continue

            if len(regions[name]) == 0:
                continue

        region_names.append(name)
        region_probs.append(weight)

    if len(region_names) == 0:
        raise RuntimeError("No valid curriculum regions available.")

    region_probs = np.array(region_probs, dtype=float)
    region_probs = region_probs / np.sum(region_probs)

    radius_scale = get_radius_scale(s)

    # When a region is forced (evaluation) use the widest synthetic gate, so
    # the forced region is actually representable instead of resampling
    # forever.
    gate_s = 1.0 if force_region is not None else s

    for _ in range(max_tries):

        region_name = rng.choice(region_names, p=region_probs)

        if region_name == "synthetic_capture":
            state = sample_synthetic_capture(
                P=sets.P,
                c_b=sets.c_b,
                rng=rng
            )

        elif region_name == "synthetic_mid":
            state = sample_synthetic_mid(
                P=sets.P,
                c_b=sets.c_b,
                rng=rng
            )

        else:
            state = sample_trace_region(
                region_name=region_name,
                regions=regions,
                rng=rng,
                radius_scale=radius_scale
            )

        if state is None:
            continue

        b, f, c = compute_bfc(sets, state)

        if c != 1.0:
            continue

        # Gate ONLY the synthetic shells. The trace regions are fixed real
        # states pulled from the reference trajectory -- their h_B / attitude
        # / speed are not something s controls. Applying the gate to them was
        # what made near_ceiling and bridge 0%-reachable below s ~ 0.82, i.e.
        # the same "hard region unreachable" failure the continuous curriculum
        # exists to remove, reintroduced through the gate.
        if region_name in ("synthetic_capture", "synthetic_mid"):
            if not passes_level_gate(
                state=state,
                P=sets.P,
                c_b=sets.c_b,
                zceil=sets.zceil,
                s=gate_s
            ):
                continue

        if return_region:
            return state, region_name

        return state

    raise RuntimeError(
        f"Failed to sample valid state at difficulty s={s:.3f}."
    )


def inspect_sampler(
        sets,
        regions,
        s,
        rng,
        n_samples=500
):

    hb_list = []
    hs_list = []
    vnorm_list = []
    att_list = []
    pz_list = []

    region_count = {
        "synthetic_capture": 0,
        "synthetic_mid": 0,
        "trace_general": 0,
        "near_ceiling": 0,
        "bridge": 0
    }

    for _ in range(n_samples):

        # Use the TRUE region label. The previous version guessed the region
        # from h_B thresholds, which cannot distinguish the three trace
        # regions and so silently misreported the mixture.
        state, region_name = sample_initial_state(
            sets=sets,
            regions=regions,
            s=s,
            rng=rng,
            return_region=True
        )

        hb, hs, vnorm, att_err = state_difficulty(
            state=state,
            P=sets.P,
            c_b=sets.c_b,
            zceil=sets.zceil
        )

        hb_list.append(hb)
        hs_list.append(hs)
        vnorm_list.append(vnorm)
        att_list.append(att_err)
        pz_list.append(state[2])

        if region_name in region_count:
            region_count[region_name] += 1

    print("---------------------------------------")
    print(f"Sampler inspection, s = {s:.2f}")
    print(f"Radius scale: {get_radius_scale(s):.2f}")
    print("Sampled regions:")

    for name, count in region_count.items():
        print(f"  {name}: {count}")

    print("")
    print(f"h_B mean/min/max: {np.mean(hb_list):.3f}, {np.min(hb_list):.3f}, {np.max(hb_list):.3f}")
    print(f"h_S mean/min/max: {np.mean(hs_list):.3f}, {np.min(hs_list):.3f}, {np.max(hs_list):.3f}")
    print(f"p_z mean/min/max: {np.mean(pz_list):.3f}, {np.min(pz_list):.3f}, {np.max(pz_list):.3f}")
    print(f"|v| mean/min/max: {np.mean(vnorm_list):.3f}, {np.min(vnorm_list):.3f}, {np.max(vnorm_list):.3f}")
    print(f"|att err| mean/min/max: {np.mean(att_list):.3f}, {np.min(att_list):.3f}, {np.max(att_list):.3f}")
    print("---------------------------------------")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_policy(
        policy,
        sets,
        regions,
        s,
        rng,
        eval_episodes=150,
        max_episode_steps=300,
        success_horizon_steps=100,
        stratified=True
):
    """
    Stratified evaluation producing a mu_SA estimate.

    mu_SA (paper Sec. 4.1.2, Eq. 9) is the fraction of the design region Omega
    from which the policy safely arrives at B within the backup horizon T.
    Here T = success_horizon_steps = 100 steps = 2.0 s, matching Table 8.

    FIX #3: the previous evaluator sampled from the CURRICULUM mixture, so the
    number of near_ceiling episodes was random and often tiny -- and the mixture
    itself changed as the curriculum advanced, making successive evaluations
    non-comparable. We now run a FIXED number of episodes PER REGION and combine
    them with FIXED weights (MU_SA_WEIGHTS), so the reported mu_SA is a stable,
    comparable estimate of the same quantity throughout training.
    """

    eval_dyn = Dynamics()

    if stratified:
        eval_regions = [
            "synthetic_capture", "synthetic_mid",
            "trace_general", "near_ceiling", "bridge"
        ]
        eval_regions = [
            r for r in eval_regions
            if r in ["synthetic_capture", "synthetic_mid"]
            or (r in regions and len(regions[r]) > 0)
        ]
        per_region = max(1, eval_episodes // len(eval_regions))
        plan = [(r, per_region) for r in eval_regions]
    else:
        plan = [(None, eval_episodes)]

    success_count = 0
    success_horizon_count = 0
    failure_count = 0
    timeout_count = 0
    total_episodes = 0

    steps_list = []
    min_hs_list = []
    final_hb_list = []

    region_stats = {}

    for forced_region, n_eps in plan:

        for _ in range(n_eps):

            state, region_name = sample_initial_state(
                sets=sets,
                regions=regions,
                s=s,
                rng=rng,
                return_region=True,
                force_region=forced_region
            )

            total_episodes += 1

            if region_name not in region_stats:
                region_stats[region_name] = {
                    "n": 0,
                    "success": 0,
                    "success_horizon": 0,
                    "failure": 0,
                    "timeout": 0,
                    "steps": [],
                    "final_hb": []
                }

            region_stats[region_name]["n"] += 1

            reset_dynamics_state(eval_dyn, state)

            min_hs = 1e9
            final_hb = None
            outcome = "timeout"
            terminal_step = max_episode_steps

            for step in range(max_episode_steps):

                action_norm = policy.select_action(np.array(state))
                action_norm = np.clip(action_norm, -1.0, 1.0)

                action = scale_action(action_norm, eval_dyn.g)
                next_state = eval_dyn.step(action).copy()

                b_next, f_next, c_next = compute_bfc(sets, next_state)

                min_hs = min(min_hs, sets.hs)

                reduced_next = compute_reduced_state(next_state)
                sets.compute_hb(reduced_next)
                final_hb = sets.hb

                state = next_state.copy()

                if b_next == 1.0:
                    outcome = "success"
                    terminal_step = step + 1
                    break

                if f_next == 1.0:
                    outcome = "failure"
                    terminal_step = step + 1
                    break

            if outcome == "success":

                success_count += 1
                region_stats[region_name]["success"] += 1

                # Only arrivals within the backup horizon T count toward mu_SA.
                if terminal_step <= success_horizon_steps:
                    success_horizon_count += 1
                    region_stats[region_name]["success_horizon"] += 1

            elif outcome == "failure":

                failure_count += 1
                region_stats[region_name]["failure"] += 1

            else:

                timeout_count += 1
                region_stats[region_name]["timeout"] += 1

            steps_list.append(terminal_step)
            min_hs_list.append(min_hs)

            # Clip h_B for REPORTING only. Divergent rollouts produce values on
            # the order of -1e6, which made the reported average meaningless.
            final_hb_list.append(max(final_hb, -1e3))

            region_stats[region_name]["steps"].append(terminal_step)
            region_stats[region_name]["final_hb"].append(max(final_hb, -1e3))

    success_rate = success_count / total_episodes
    success_horizon_rate = success_horizon_count / total_episodes
    failure_rate = failure_count / total_episodes
    timeout_rate = timeout_count / total_episodes
    avg_steps = np.mean(steps_list)
    avg_min_hs = np.mean(min_hs_list)
    avg_final_hb = np.mean(final_hb_list)
    median_final_hb = np.median(final_hb_list)

    # ---- mu_SA estimate: fixed-weight combination of per-region rates ----
    mu_sa = 0.0
    weight_total = 0.0

    for name, stats in region_stats.items():
        w = MU_SA_WEIGHTS.get(name, 0.0)
        if w <= 0.0 or stats["n"] == 0:
            continue
        mu_sa += w * (stats["success_horizon"] / stats["n"])
        weight_total += w

    if weight_total > 0.0:
        mu_sa = mu_sa / weight_total

    worst_region_success = min(
        stats["success_horizon"] / stats["n"]
        for stats in region_stats.values()
    )

    near_ceiling_rate = None
    if "near_ceiling" in region_stats and region_stats["near_ceiling"]["n"] > 0:
        near_ceiling_rate = (
            region_stats["near_ceiling"]["success_horizon"]
            / region_stats["near_ceiling"]["n"]
        )

    print("---------------------------------------")
    print(f"Evaluation over {total_episodes} episodes")
    print(f"Difficulty s: {s:.3f}")
    print(f"mu_SA (weighted, T={success_horizon_steps} steps): {mu_sa:.3f}")
    if near_ceiling_rate is not None:
        print(f"  near_ceiling rate: {near_ceiling_rate:.3f}   (paper: 0.693)")
    print(f"Success rate (any horizon): {success_rate:.3f}")
    print(f"Success <= {success_horizon_steps} steps: {success_horizon_rate:.3f}")
    print(f"Failure rate: {failure_rate:.3f}")
    print(f"Timeout rate: {timeout_rate:.3f}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average min h_S: {avg_min_hs:.3f}")
    print(f"Average final h_B (clipped): {avg_final_hb:.3f}")
    print(f"Median final h_B: {median_final_hb:.3f}")
    print(f"Worst-region success_horizon: {worst_region_success:.3f}")
    print("")
    print("Per-region evaluation:")

    for name, stats in region_stats.items():

        n = stats["n"]
        s_rate = stats["success"] / n
        sh_rate = stats["success_horizon"] / n
        f_rate = stats["failure"] / n
        t_rate = stats["timeout"] / n
        avg_region_steps = np.mean(stats["steps"])
        med_region_hb = np.median(stats["final_hb"])

        print(
            f"  {name}: "
            f"n={n}, "
            f"success={s_rate:.3f}, "
            f"success_horizon={sh_rate:.3f}, "
            f"failure={f_rate:.3f}, "
            f"timeout={t_rate:.3f}, "
            f"avg_steps={avg_region_steps:.1f}, "
            f"median_final_hB={med_region_hb:.1f}"
        )

    print("---------------------------------------")

    return (
        success_rate,
        success_horizon_rate,
        failure_rate,
        timeout_rate,
        avg_steps,
        avg_min_hs,
        avg_final_hb,
        median_final_hb,
        worst_region_success,
        mu_sa
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    np.random.seed(0)
    torch.manual_seed(0)

    rng = np.random.default_rng(0)

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    dyn = Dynamics()

    gains = LQRGain(
        dt=dyn.del_t,
        g=dyn.g
    )

    K, P = gains.gain()

    c_b = 8.0

    sets = SetIndicator(
        P=P,
        c_b=c_b,
        zceil=3.0
    )

    # Store P inside sets so the sampler can use it.
    sets.P = P

    lqr = LQRController(K=K, g=dyn.g, z_des=2.0)

    trace_states = generate_reference_trace(
        n_points=200,
        n_variants=20,
        seed=0
    )

    regions = classify_trace_states(
        sets=sets,
        trace_states=trace_states,
        P=P,
        c_b=c_b,
        near_ceiling_margin=0.25
    )

    print("Trace region sizes:")
    for name, states in regions.items():
        print(f"  {name}: {len(states)}")

    for s_probe in [0.0, 0.25, 0.5, 0.75, 1.0]:
        inspect_sampler(
            sets=sets,
            regions=regions,
            s=s_probe,
            rng=rng,
            n_samples=500
        )

    state_dim = 10
    action_dim = 4
    max_action = 1.0

    policy = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=0.99,
        tau=0.0025,
        policy_noise=0.10,
        noise_clip=0.10,
        policy_freq=2,
        actor_lr=1e-4,
        critic_lr=3e-4
    )

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim
    )

    model_name = "./models/td3_safe_arrival_v4"

    # ---- Table 8 (quadrotor, Phase I) ----
    max_timesteps = 5000000        # paper: 5e6
    start_timesteps = 5000
    eval_freq = 10000
    batch_size = 128               # paper: 128 (Phase I; Phase II uses 64)
    max_episode_steps = 300        # rollout budget; T itself is 100 steps
    success_horizon_steps = 100    # paper: backup horizon T = 2.0 s = 100 steps
    eval_episodes = 150            # stratified -> 30 per region

    expl_noise = 0.10

    # ---- FIX #6: warm start ----
    warm_start_transitions = 100000

    # ---- FIX #3: continuous curriculum ----
    s = 0.0
    s_step = 0.10                  # paper Table 8: delta_s = 0.10
    s_backoff = 0.05
    # Targets are grounded in the paper's own reported numbers rather than
    # invented: 85.9% overall recoverability and 69.3% near-ceiling for the
    # final learned policy (Sec. 5.2). Gating on ~0.90 would be gating on
    # something the paper itself never achieves.
    mu_sa_threshold = 0.80
    mu_sa_backoff_threshold = 0.40
    curriculum_window = 3          # rolling window over evals
    min_evals_between_updates = 3
    stall_patience = 12            # forced half-step if progress stalls
    stall_min_mu_sa = 0.55

    evals_since_curriculum_update = 0
    mu_sa_history = []
    best_mu_sa = -1.0

    evaluations = []

    # -----------------------------------------------------------------------
    # Warm start the replay buffer with LQR rollouts across ALL regions.
    # -----------------------------------------------------------------------
    warm_start_buffer(
        replay_buffer=replay_buffer,
        lqr=lqr,
        sets=sets,
        regions=regions,
        rng=rng,
        n_transitions=warm_start_transitions,
        max_episode_steps=max_episode_steps
    )

    state = sample_initial_state(
        sets=sets,
        regions=regions,
        s=s,
        rng=rng
    )

    reset_dynamics_state(dyn, state)

    episode_timesteps = 0
    episode_num = 0

    episode_success = 0
    episode_failure = 0
    episode_timeout = 0

    for t in range(max_timesteps):

        episode_timesteps += 1

        if t < start_timesteps:

            action_norm = np.random.uniform(
                -1.0,
                1.0,
                size=action_dim
            )

        else:

            action_norm = policy.select_action(np.array(state))

            action_norm = (
                action_norm
                + np.random.normal(0.0, expl_noise, size=action_dim)
            ).clip(-1.0, 1.0)

        action = scale_action(
            action_norm,
            dyn.g
        )

        # FIX #1: indicators of the CURRENT state x -- this is what Eq. (11)
        # requires. Computed BEFORE the b_next call below so that the trailing
        # values of sets.hb / sets.hs correspond to next_state.
        b_cur, f_cur, c_cur = compute_bfc(
            sets,
            state
        )

        next_state = dyn.step(action).copy()

        # Indicators of the successor x' -- used ONLY for episode termination.
        b_next, f_next, c_next = compute_bfc(
            sets,
            next_state
        )

        replay_buffer.add(
            state,
            action_norm,
            next_state,
            b_cur,
            c_cur
        )

        success = b_next == 1.0
        failure = f_next == 1.0

        # FIX #1 (terminal anchors): the episode breaks as soon as x' enters
        # B or F, so without this no transition is ever stored whose CURRENT
        # state lies in B or F -- and those two states are precisely the
        # ground-truth anchors Q|_B = 1 and Q|_F = 0 that pin the value scale.
        # Because c = 0 there, the bootstrap vanishes and the target is exactly
        # the constant b, so next_state appearing on both sides is harmless.
        #
        # Deliberately NOT done on timeout: timeout is not absorbing.
        if success or failure:

            b_term, f_term, c_term = compute_bfc(
                sets,
                next_state
            )

            replay_buffer.add(
                next_state,
                action_norm,
                next_state,
                b_term,
                c_term
            )

        state = next_state.copy()

        if t >= start_timesteps and replay_buffer.size >= batch_size:

            if t % 8 == 0:
                policy.train(
                    replay_buffer,
                    batch_size
                )

        timeout = episode_timesteps >= max_episode_steps

        done = success or failure or timeout

        if done:

            if success:
                episode_success += 1
            elif failure:
                episode_failure += 1
            else:
                episode_timeout += 1

            if (episode_num + 1) % 50 == 0:
                print(
                    f"Total T: {t + 1} "
                    f"Episode Num: {episode_num + 1} "
                    f"Episode T: {episode_timesteps} "
                    f"s: {s:.3f} "
                    f"Success: {episode_success} "
                    f"Failure: {episode_failure} "
                    f"Timeout: {episode_timeout}"
                )

            state = sample_initial_state(
                sets=sets,
                regions=regions,
                s=s,
                rng=rng
            )

            reset_dynamics_state(
                dyn,
                state
            )

            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % eval_freq == 0:

            eval_result = eval_policy(
                policy=policy,
                sets=sets,
                regions=regions,
                s=s,
                rng=rng,
                eval_episodes=eval_episodes,
                max_episode_steps=max_episode_steps,
                success_horizon_steps=success_horizon_steps,
                stratified=True
            )

            mu_sa = eval_result[9]

            evaluations.append(
                (s, *eval_result)
            )

            np.save(
                "./results/td3_safe_arrival_v4_eval.npy",
                np.array(evaluations, dtype=float)
            )

            policy.save(model_name)

            # Keep the best-mu_SA checkpoint separately: the curriculum can
            # back off, and save() always targets the same path, so without
            # this a later worse policy silently overwrites the best one.
            if mu_sa > best_mu_sa:
                best_mu_sa = mu_sa
                policy.save(model_name + "_best")
                print(f"New best mu_SA = {mu_sa:.3f} -- checkpoint saved")

            evals_since_curriculum_update += 1

            mu_sa_history.append(mu_sa)
            if len(mu_sa_history) > curriculum_window:
                mu_sa_history.pop(0)

            if (
                evals_since_curriculum_update >= min_evals_between_updates
                and len(mu_sa_history) >= curriculum_window
            ):

                windowed_mu_sa = float(np.mean(mu_sa_history))

                # Escape hatch: if mu_SA has plateaued just under threshold,
                # advance anyway with a half step. Spending the whole budget
                # at one difficulty is worse than advancing imperfectly.
                stalled = (
                    evals_since_curriculum_update >= stall_patience
                    and windowed_mu_sa >= stall_min_mu_sa
                )

                if (windowed_mu_sa >= mu_sa_threshold or stalled) and s < 1.0:

                    step = s_step if windowed_mu_sa >= mu_sa_threshold else 0.5 * s_step

                    s = min(1.0, s + step)
                    evals_since_curriculum_update = 0
                    mu_sa_history = []

                    tag = "advance" if not stalled else "forced (stall)"

                    print("=======================================")
                    print(f"Difficulty increased to s = {s:.3f}  [{tag}]")
                    print(f"  windowed mu_SA = {windowed_mu_sa:.3f}")
                    print("=======================================")

                elif windowed_mu_sa < mu_sa_backoff_threshold and s > 0.0:

                    s = max(0.0, s - s_backoff)
                    evals_since_curriculum_update = 0
                    mu_sa_history = []

                    print("=======================================")
                    print(f"Difficulty backed off to s = {s:.3f}")
                    print(f"  windowed mu_SA = {windowed_mu_sa:.3f}")
                    print("=======================================")