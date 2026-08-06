"""My-methods Phase-I initial-state sampler (ported verbatim from the original
monolithic train.py).

This is the hand-designed continuous-curriculum sampler that predates the
official reset_library approach. It controls difficulty via a scalar s in [0,1]
with smooth per-region triangular weights and per-region floors that keep the
HARD regions (near_ceiling, bridge) sampled from step 0. Synthetic shell states
are generated on the LQR ellipsoid; trace regions come from randomized
closed-loop tracker rollouts through the SAME Euler dynamics used in training.

The functions below are extracted unchanged from the original train.py so the
sampling behavior is identical to the previous method. Only the imports at the
top are new (they point at the current modular helpers).
"""
from __future__ import annotations

import os

import numpy as np

from backup_policy.phase1.state_action import (
    compute_bfc,
    compute_reduced_state,
    normalize_quat,
    quat_from_axis_angle,
    quat_mult,
    reset_dynamics_state,
)


def state_difficulty(state, P, c_b, zceil):

    xe = compute_reduced_state(state)

    hb = c_b - xe.T @ P @ xe
    hs = zceil - state[2]
    vnorm = np.linalg.norm(state[3:6])
    att_err = np.linalg.norm(xe[4:7])

    return hb, hs, vnorm, att_err


def gate_bounds(s):
    """Continuous gate bounds as a function of difficulty s in [0,1]."""

    def lerp(a, b):
        return a + (b - a) * s

    hb_min = lerp(-1.0, -150.0)
    v_max = lerp(1.0, 5.5)
    att_max = lerp(0.30, 2.00)
    hs_min = lerp(0.50, 0.02)

    return hb_min, v_max, att_max, hs_min


def passes_level_gate(state, P, c_b, zceil, s):

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
def get_radius_scale(s):
    """Trace perturbation radius grows smoothly with difficulty."""

    return 0.10 + 0.90 * s
def get_curriculum_weights(s):
    """
    Continuous region mixture.

    Each region has a "center" difficulty at which it peaks. Weights are
    smooth triangular bumps, so the mixture drifts gradually from synthetic
    capture states toward bridge/near-ceiling states. Crucially, EVERY region
    retains nonzero weight once it has been introduced, which prevents the
    catastrophic forgetting seen in the discrete version.
    """

    # Centers pulled earlier than the original (0/.25/.50/.75/1.00) spacing,
    # and floor raised from 0.06 to 0.10. With the gate fix above, the hard
    # trace regions are reachable from s=0 onward, so there is no longer a
    # reason to withhold most of their training exposure until s is near 1 --
    # doing so was exactly what left near_ceiling/bridge undertrained even
    # after they became samplable.
    centers = {
        "synthetic_capture": 0.00,
        "synthetic_mid": 0.15,
        "trace_general": 0.30,
        "near_ceiling": 0.50,
        "bridge": 0.70,
    }

    width = 0.35

    # FIX: per-region floors instead of a single global 0.10.
    #
    # The hard regions are the whole point of Phase I -- the paper's gain over
    # the analytic backup comes almost entirely from near-ceiling recoverability
    # (35.5% -> 69.3%, Sec. 5.2). Under a flat 0.10 floor, near_ceiling holds
    # only ~10% mixture weight until s reaches ~0.5, and since the observed run
    # never left s = 0, it received ~4% of samples for the ENTIRE 3.1M steps.
    # Giving the hard regions a higher floor guarantees they are trained from
    # step 0 regardless of how slowly the curriculum moves, which also makes
    # the run far less sensitive to the curriculum stalling at all.
    floors = {
        "synthetic_capture": 0.10,
        "synthetic_mid": 0.10,
        "trace_general": 0.15,
        "near_ceiling": 0.35,
        "bridge": 0.30,
    }

    weights = {}

    for name, center in centers.items():
        bump = max(0.0, 1.0 - abs(s - center) / width)
        weights[name] = floors.get(name, 0.10) + bump

    return weights
def sample_reduced_shell(P, c_b, delta_min, delta_max, rng):
    """
    Samples reduced state xe such that:
        xe^T P xe = c_b + delta
    Therefore:
        h_B = -delta
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
def powerloop_reference(t, center=np.array([0.0, 0.0, 2.0]), radius=1.5, v0=4.5):
    """Analytic power-loop reference position/velocity/attitude at time t."""

    omega = v0 / radius
    theta = -np.pi / 2.0 + omega * t

    px = center[0] + radius * (np.cos(theta) - np.cos(-np.pi / 2.0))
    py = center[1]
    pz = center[2] + radius * np.sin(theta)

    vx = -v0 * np.sin(theta)
    vy = 0.0
    vz = v0 * np.cos(theta)

    q = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), theta + np.pi / 2.0)

    return np.array([px, py, pz]), np.array([vx, vy, vz]), q
def _tracker_action(state, p_ref, v_ref, q_ref, g, gains, rng):
    """
    Randomized proportional tracker.

    Deliberately imperfect: the per-rollout gain jitter makes some rollouts
    sluggish and some aggressive, which is what produces the spread in tracking
    error that the design region needs.
    """

    kp, kv, katt = gains

    p = state[0:3]
    v = state[3:6]
    q = normalize_quat(state[6:10])

    # Desired acceleration in the inertial frame.
    a_des = kp * (p_ref - p) + kv * (v_ref - v) + np.array([0.0, 0.0, g])

    # Body z-axis from the current attitude.
    R = np.array([
        [1 - 2 * (q[2]**2 + q[3]**2),
         2 * (q[1] * q[2] - q[0] * q[3]),
         2 * (q[1] * q[3] + q[0] * q[2])],
        [2 * (q[1] * q[2] + q[0] * q[3]),
         1 - 2 * (q[1]**2 + q[3]**2),
         2 * (q[2] * q[3] - q[0] * q[1])],
        [2 * (q[1] * q[3] - q[0] * q[2]),
         2 * (q[2] * q[3] + q[0] * q[1]),
         1 - 2 * (q[1]**2 + q[2]**2)],
    ])

    b3 = R[:, 2]
    a_cmd = float(np.dot(a_des, b3))
    a_cmd = np.clip(a_cmd, 0.0, 4.0 * g)

    # Attitude target.
    #
    # Tracking the reference quaternion alone does NOT work here: the power-loop
    # reference commands a full 360-degree flip, so the reference body z-axis
    # points downward through much of the loop. Following it blindly makes the
    # tracker lose all altitude authority (thrust saturates at 0 and the vehicle
    # falls). We instead aim the body z-axis at the DESIRED ACCELERATION, which
    # is what any real cascaded tracker does, and blend in the reference flip so
    # the rollouts still sweep the loop attitudes the design region needs.
    a_dir = a_des / (np.linalg.norm(a_des) + 1e-9)

    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, a_dir)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-6:
        q_thrust = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        angle = np.arccos(np.clip(np.dot(z_axis, a_dir), -1.0, 1.0))
        q_thrust = quat_from_axis_angle(axis / axis_norm, angle)

    # Blend: mostly thrust-aligned (keeps the vehicle flying), partly toward the
    # reference attitude (keeps the loop character and the flip).
    blend = 0.25
    q_target = normalize_quat((1.0 - blend) * q_thrust + blend * q_ref)

    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    q_err = quat_mult(q_conj, q_target)

    if q_err[0] < 0.0:
        q_err = -q_err

    omega = katt * 2.0 * q_err[1:4]
    omega = omega + rng.normal(0.0, 0.25, size=3)
    omega = np.clip(omega, -18.0, 18.0)

    return np.array([a_cmd, omega[0], omega[1], omega[2]])
def generate_tracker_trace(dyn_factory, n_variants=20, horizon=106, seed=0):
    """
    Roll out a randomized tracker through the true dynamics and collect the
    visited states. This is the stand-in for the paper's vanilla SAC traces.
    """

    rng = np.random.default_rng(seed)
    trace_states = []

    for _ in range(n_variants):

        dyn = dyn_factory()
        g = dyn.g
        dt = dyn.del_t

        # Per-rollout gain jitter -> a spread of tracking competence.
        # Ranges chosen so the loop is actually flown: too-soft gains just fall
        # out of the sky and contribute states that are not task-relevant.
        gains = (
            rng.uniform(12.0, 26.0),   # kp
            rng.uniform(6.0, 12.0),    # kv
            rng.uniform(10.0, 20.0),   # katt
        )

        p0, v0_vec, q0 = powerloop_reference(0.0)

        state = np.concatenate([
            p0 + rng.normal(0.0, 0.05, size=3),
            v0_vec + rng.normal(0.0, 0.10, size=3),
            q0
        ])
        state[6:10] = normalize_quat(state[6:10])

        reset_dynamics_state(dyn, state)

        for k in range(horizon):

            t = k * dt
            p_ref, v_ref, q_ref = powerloop_reference(t)

            action = _tracker_action(
                state, p_ref, v_ref, q_ref, g, gains, rng
            )

            state = dyn.step(action).copy()

            if not np.all(np.isfinite(state)):
                break

            # Keep the rollout inside a task-relevant envelope. A tracker that
            # has fallen out of the loop is no longer producing states the PS2
            # policy will ever be asked to recover from, and including them just
            # dilutes the design region.
            if np.linalg.norm(state[3:6]) > 12.0:
                break

            if state[2] < -0.5 or state[2] > 4.5:
                break

            trace_states.append(state.copy())

    return trace_states
def generate_reference_trace(n_points=200, n_variants=20, seed=0):
    """
    Analytic power-loop reference sampler.

    Kept as a fallback only. Prefer `generate_tracker_trace`, which yields
    genuine closed-loop states; see the module comment above for why the
    analytic circle makes near_ceiling/bridge degenerate.
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
def load_or_generate_trace(
        trace_path,
        n_points=200,
        n_variants=20,
        seed=0,
        dyn_factory=None,
        mode="tracker"
):
    """
    Priority:
      1. Real saved traces from --trace_path (use vanilla SAC traces here once
         they exist -- this is the paper-faithful path).
      2. Closed-loop randomized-tracker rollouts (default stand-in).
      3. Analytic reference circle (fallback only).
    """

    if trace_path is not None and os.path.exists(trace_path):
        traces = np.load(trace_path)
        traces = traces.reshape(-1, 10)
        print(f"Loaded {traces.shape[0]} trace states from {trace_path}")
        return [traces[i].copy() for i in range(traces.shape[0])]

    if mode == "tracker" and dyn_factory is not None:
        states = generate_tracker_trace(
            dyn_factory=dyn_factory,
            n_variants=n_variants,
            seed=seed
        )
        print(f"Generated {len(states)} closed-loop tracker trace states")
        return states

    return generate_reference_trace(
        n_points=n_points,
        n_variants=n_variants,
        seed=seed
    )
def classify_trace_states(
        sets,
        trace_states,
        P,
        c_b,
        near_ceiling_margin=0.25
):

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

    `s` is the difficulty scalar in [0,1].
    `force_region` restricts sampling to one region (used by evaluation so we
    always get a fixed number of near_ceiling episodes regardless of mixture).
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

    # When a region is forced (evaluation), relax the gate so that the region
    # is actually representable; otherwise we would silently resample forever.
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

        # FIX: `passes_level_gate` was being applied uniformly to every
        # region, including the trace-derived ones (trace_general,
        # near_ceiling, bridge). Those are FIXED real states pulled from the
        # reference trajectory -- their h_B/attitude/etc. are not something
        # `s` controls, unlike the procedurally-generated synthetic shells.
        # near_ceiling/bridge states sit at att_err ~1.7 rad, but the gate's
        # att_max only reaches 1.7 once s -> ~0.99, so at any s below that
        # virtually every draw from these regions failed the gate and the
        # loop just silently retried a different region instead. Empirically
        # this gave near_ceiling/bridge/trace_general a ~0% pass rate for
        # s <= 0.6 despite having nonzero mixture weight -- i.e. the policy
        # was never actually trained on them for most of a run, which is the
        # same "hard region unreachable" failure mode the continuous
        # curriculum was written to fix, just reintroduced through the gate
        # instead of the old discrete levels. Only gate the synthetic shells,
        # whose difficulty really is controlled by `s`; trace regions are
        # already curated by `classify_trace_states` + the c==1 check above.
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
def inspect_sampler(sets, regions, s, rng, n_samples=500):

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
    print("Actual sampled regions:")

    for name, count in region_count.items():
        print(f"  {name}: {count}")

    print("")
    print(f"h_B mean/min/max: {np.mean(hb_list):.3f}, {np.min(hb_list):.3f}, {np.max(hb_list):.3f}")
    print(f"h_S mean/min/max: {np.mean(hs_list):.3f}, {np.min(hs_list):.3f}, {np.max(hs_list):.3f}")
    print(f"p_z mean/min/max: {np.mean(pz_list):.3f}, {np.min(pz_list):.3f}, {np.max(pz_list):.3f}")
    print(f"|v| mean/min/max: {np.mean(vnorm_list):.3f}, {np.min(vnorm_list):.3f}, {np.max(vnorm_list):.3f}")
    print(f"|att err| mean/min/max: {np.mean(att_list):.3f}, {np.min(att_list):.3f}, {np.max(att_list):.3f}")
    print("---------------------------------------")