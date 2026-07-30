"""Reference-trace generation and continuous curriculum initial-state sampling."""
import numpy as np
from .state_action import compute_reduced_state, compute_bfc, normalize_quat, quat_from_axis_angle, quat_mult

MU_SA_WEIGHTS = {
    "synthetic_capture": 0.10,
    "synthetic_mid": 0.15,
    "trace_general": 0.25,
    "near_ceiling": 0.30,
    "bridge": 0.20,
}

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

def get_radius_scale(s):
    """Trace perturbation radius grows smoothly with difficulty s."""

    return 0.10 + 0.90 * s

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
