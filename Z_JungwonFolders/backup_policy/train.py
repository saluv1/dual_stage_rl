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


def passes_level_gate(state, P, c_b, zceil, curriculum_level):

    hb, hs, vnorm, att_err = state_difficulty(state, P, c_b, zceil)

    if curriculum_level == 0:
        hb_min = -1.0
        v_max = 1.0
        att_max = 0.30
        hs_min = 0.50

    elif curriculum_level == 1:
        hb_min = -8.0
        v_max = 1.8
        att_max = 0.50
        hs_min = 0.40

    elif curriculum_level == 2:
        hb_min = -20.0
        v_max = 2.5
        att_max = 0.75
        hs_min = 0.30

    elif curriculum_level == 3:
        hb_min = -40.0
        v_max = 3.5
        att_max = 1.00
        hs_min = 0.20

    elif curriculum_level == 4:
        hb_min = -80.0
        v_max = 4.5
        att_max = 1.40
        hs_min = 0.10

    else:
        hb_min = -150.0
        v_max = 5.5
        att_max = 2.00
        hs_min = 0.02

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

def get_radius_scale(curriculum_level):

    if curriculum_level == 0:
        return 0.10
    elif curriculum_level == 1:
        return 0.20
    elif curriculum_level == 2:
        return 0.35
    elif curriculum_level == 3:
        return 0.55
    elif curriculum_level == 4:
        return 0.75
    else:
        return 1.00


def get_curriculum_weights(curriculum_level):

    if curriculum_level == 0:
        return {
            "synthetic_capture": 0.90,
            "synthetic_mid": 0.10,
            "trace_general": 0.00,
            "near_ceiling": 0.00,
            "bridge": 0.00,
        }

    elif curriculum_level == 1:
        return {
            "synthetic_capture": 0.60,
            "synthetic_mid": 0.30,
            "trace_general": 0.10,
            "near_ceiling": 0.00,
            "bridge": 0.00,
        }

    elif curriculum_level == 2:
        return {
            "synthetic_capture": 0.35,
            "synthetic_mid": 0.30,
            "trace_general": 0.30,
            "near_ceiling": 0.05,
            "bridge": 0.00,
        }

    elif curriculum_level == 3:
        return {
            "synthetic_capture": 0.20,
            "synthetic_mid": 0.25,
            "trace_general": 0.35,
            "near_ceiling": 0.15,
            "bridge": 0.05,
        }

    elif curriculum_level == 4:
        return {
            "synthetic_capture": 0.10,
            "synthetic_mid": 0.15,
            "trace_general": 0.35,
            "near_ceiling": 0.25,
            "bridge": 0.15,
        }

    else:
        return {
            "synthetic_capture": 0.05,
            "synthetic_mid": 0.10,
            "trace_general": 0.35,
            "near_ceiling": 0.25,
            "bridge": 0.25,
        }


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
        curriculum_level,
        rng,
        max_curriculum_level=5,
        max_tries=10000
):
    """
    Hybrid curriculum sampler.

    Level 0:
        mostly synthetic states just outside B.

    Later levels:
        gradually mix in task-trace, near-ceiling, and bridge states.
    """

    weights = get_curriculum_weights(curriculum_level)

    region_names = []
    region_probs = []

    for name, weight in weights.items():

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

    radius_scale = get_radius_scale(curriculum_level)

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

        if not passes_level_gate(
            state=state,
            P=sets.P,
            c_b=sets.c_b,
            zceil=sets.zceil,
            curriculum_level=curriculum_level
        ):
            continue

        return state

    raise RuntimeError(
        f"Failed to sample valid state at curriculum level {curriculum_level}."
    )


def inspect_sampler(
        sets,
        regions,
        curriculum_level,
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

        state = sample_initial_state(
            sets=sets,
            regions=regions,
            curriculum_level=curriculum_level,
            rng=rng
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

        # Approximate source classification for diagnostics
        if hb > -1.05:
            region_count["synthetic_capture"] += 1
        elif hb > -8.5:
            region_count["synthetic_mid"] += 1
        else:
            region_count["trace_general"] += 1

    print("---------------------------------------")
    print(f"Sampler inspection, level {curriculum_level}")
    print(f"Radius scale: {get_radius_scale(curriculum_level):.2f}")
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
        curriculum_level,
        rng,
        eval_episodes=50,
        max_episode_steps=300,
        success_horizon_steps=100
):

    eval_dyn = Dynamics()

    success_count = 0
    success_horizon_count = 0
    failure_count = 0
    timeout_count = 0

    steps_list = []
    min_hs_list = []
    final_hb_list = []

    for _ in range(eval_episodes):

        state = sample_initial_state(
            sets=sets,
            regions=regions,
            curriculum_level=curriculum_level,
            rng=rng
        )

        reset_dynamics_state(eval_dyn, state)

        min_hs = 1e9
        final_hb = None

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
                success_count += 1

                if step + 1 <= success_horizon_steps:
                    success_horizon_count += 1

                steps_list.append(step + 1)
                break

            if f_next == 1.0:
                failure_count += 1
                steps_list.append(step + 1)
                break

            if step == max_episode_steps - 1:
                timeout_count += 1
                steps_list.append(max_episode_steps)

        min_hs_list.append(min_hs)
        final_hb_list.append(final_hb)

    success_rate = success_count / eval_episodes
    success_horizon_rate = success_horizon_count / eval_episodes
    failure_rate = failure_count / eval_episodes
    timeout_rate = timeout_count / eval_episodes
    avg_steps = np.mean(steps_list)
    avg_min_hs = np.mean(min_hs_list)
    avg_final_hb = np.mean(final_hb_list)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes")
    print(f"Curriculum level: {curriculum_level}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Success <= {success_horizon_steps} steps: {success_horizon_rate:.3f}")
    print(f"Failure rate: {failure_rate:.3f}")
    print(f"Timeout rate: {timeout_rate:.3f}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average min h_S: {avg_min_hs:.3f}")
    print(f"Average final h_B: {avg_final_hb:.3f}")
    print("---------------------------------------")

    return (
        success_rate,
        success_horizon_rate,
        failure_rate,
        timeout_rate,
        avg_steps,
        avg_min_hs,
        avg_final_hb
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

    max_curriculum_level = 5

    for level in range(max_curriculum_level + 1):
        inspect_sampler(
            sets=sets,
            regions=regions,
            curriculum_level=level,
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

    # Use separate save name so you do not overwrite older models.
    model_name = "./models/td3_safe_arrival_hybrid"

    max_timesteps = 800000
    start_timesteps = 5000
    eval_freq = 5000
    batch_size = 128
    max_episode_steps = 300
    success_horizon_steps = 100

    expl_noise = 0.10

    curriculum_level = 0
    curriculum_success_threshold = 0.80
    min_evals_between_updates = 2
    evals_since_curriculum_update = 0

    evaluations = []

    state = sample_initial_state(
        sets=sets,
        regions=regions,
        curriculum_level=curriculum_level,
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

        next_state = dyn.step(action).copy()

        b_next, f_next, c_next = compute_bfc(
            sets,
            next_state
        )

        replay_buffer.add(
            state,
            action_norm,
            next_state,
            b_next,
            c_next
        )

        state = next_state.copy()

        if t >= start_timesteps and replay_buffer.size >= batch_size:

            if t % 8 == 0:
                policy.train(
                    replay_buffer,
                    batch_size
                )

        success = b_next == 1.0
        failure = f_next == 1.0
        timeout = episode_timesteps >= max_episode_steps

        done = success or failure or timeout

        if done:

            if success:
                episode_success += 1
            elif failure:
                episode_failure += 1
            else:
                episode_timeout += 1

            print(
                f"Total T: {t + 1} "
                f"Episode Num: {episode_num + 1} "
                f"Episode T: {episode_timesteps} "
                f"Curriculum: {curriculum_level} "
                f"Success: {episode_success} "
                f"Failure: {episode_failure} "
                f"Timeout: {episode_timeout}"
            )

            state = sample_initial_state(
                sets=sets,
                regions=regions,
                curriculum_level=curriculum_level,
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
                curriculum_level=curriculum_level,
                rng=rng,
                eval_episodes=50,
                max_episode_steps=max_episode_steps,
                success_horizon_steps=success_horizon_steps
            )

            success_rate = eval_result[0]
            success_horizon_rate = eval_result[1]

            evaluations.append(
                (curriculum_level, *eval_result)
            )

            np.save(
                "./results/td3_safe_arrival_hybrid_eval.npy",
                evaluations
            )

            policy.save(model_name)

            evals_since_curriculum_update += 1

            # Advance based on reaching B within the 2 second backup horizon.
            if (
                success_horizon_rate >= curriculum_success_threshold
                and curriculum_level < max_curriculum_level
                and evals_since_curriculum_update >= min_evals_between_updates
            ):

                curriculum_level += 1
                evals_since_curriculum_update = 0

                print("=======================================")
                print(f"Curriculum increased to level {curriculum_level}")
                print("=======================================")