import os
import sys
import argparse
import inspect
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sac.sac import SAC
from sac.powerloop_env import PowerLoopEnv


def make_env(seed, mode):
    sig = inspect.signature(PowerLoopEnv.__init__)
    params = sig.parameters

    kwargs = {"seed": seed}

    if "trajectory_mode" in params:
        kwargs["trajectory_mode"] = mode
    elif "ref_mode" in params:
        kwargs["ref_mode"] = mode

    if "terminate_on_crash" in params:
        kwargs["terminate_on_crash"] = False

    if "term_pos_err" in params:
        kwargs["term_pos_err"] = float("inf")

    if "term_att_deg" in params:
        kwargs["term_att_deg"] = 180.0

    return PowerLoopEnv(**kwargs)


def make_agent(env, hidden_size):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_low": env.action_space.low,
        "action_high": env.action_space.high,
        "hidden_dim": hidden_size,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha_init": 0.2,
        "alpha_min": 0.01,
        "target_entropy": -float(action_dim),
        "actor_lr": 5e-5,
        "critic_lr": 1e-4,
        "alpha_lr": 5e-5,
        "grad_clip": 5.0,
        "q_clip": 5e6,
    }

    sig = inspect.signature(SAC.__init__)
    supported = set(sig.parameters.keys())
    kwargs = {k: v for k, v in kwargs.items() if k in supported}

    return SAC(**kwargs)


def load_agent(agent, model_path):
    if hasattr(agent, "load"):
        try:
            agent.load(model_path, evaluate=True)
            return
        except TypeError:
            agent.load(model_path)
            if hasattr(agent, "policy"):
                agent.policy.eval()
            return

    checkpoint = torch.load(model_path, map_location=getattr(agent, "device", "cpu"))

    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        agent.policy.load_state_dict(checkpoint["policy"])
    elif isinstance(checkpoint, dict) and "actor" in checkpoint:
        agent.policy.load_state_dict(checkpoint["actor"])
    else:
        agent.policy.load_state_dict(checkpoint)

    agent.policy.eval()


def get_full_state(env):
    if hasattr(env, "dyn") and hasattr(env.dyn, "state"):
        return np.asarray(env.dyn.state, dtype=float).copy()

    if hasattr(env, "state"):
        return np.asarray(env.state, dtype=float).copy()

    raise RuntimeError("Could not find full state in env.dyn.state or env.state.")


def normalize_quat(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = q / n
    if q[0] < 0:
        q = -q
    return q


def quat_to_rpy(q):
    q = normalize_quat(q)
    qw, qx, qy, qz = q

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=float)


def quat_angle_error(q, q_ref):
    q = normalize_quat(q)
    q_ref = normalize_quat(q_ref)
    dot = abs(float(np.dot(q, q_ref)))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def get_reference_data(env, info):
    ref = {
        "p_ref": None,
        "v_ref": None,
        "q_ref": None,
        "omega_ref": None,
    }

    key_groups = {
        "p_ref": ["p_ref", "pos_ref", "position_ref", "reference_position", "ref_p"],
        "v_ref": ["v_ref", "vel_ref", "velocity_ref", "reference_velocity", "ref_v"],
        "q_ref": ["q_ref", "quat_ref", "att_ref", "reference_quat", "ref_q"],
        "omega_ref": ["omega_ref", "w_ref", "body_rate_ref", "reference_omega", "ref_omega"],
    }

    for out_key, keys in key_groups.items():
        for key in keys:
            if key in info:
                value = np.asarray(info[key], dtype=float)
                ref[out_key] = value.copy()
                break

    if all(value is not None for value in ref.values()):
        return ref

    possible_methods = ["reference", "get_reference", "_get_reference", "get_ref", "_get_ref"]

    for method_name in possible_methods:
        if not hasattr(env, method_name):
            continue

        method = getattr(env, method_name)

        possible_args = []
        if hasattr(env, "step_count"):
            possible_args.append((env.step_count,))
        if hasattr(env, "t"):
            possible_args.append((env.t,))
        possible_args.append(tuple())

        for args in possible_args:
            try:
                out = method(*args)
            except Exception:
                continue

            if isinstance(out, dict):
                for out_key, keys in key_groups.items():
                    if ref[out_key] is not None:
                        continue
                    for key in keys:
                        if key in out:
                            ref[out_key] = np.asarray(out[key], dtype=float).copy()
                            break

            elif isinstance(out, (tuple, list)):
                if len(out) > 0 and ref["p_ref"] is None:
                    ref["p_ref"] = np.asarray(out[0], dtype=float).copy()
                if len(out) > 1 and ref["v_ref"] is None:
                    ref["v_ref"] = np.asarray(out[1], dtype=float).copy()
                if len(out) > 2 and ref["q_ref"] is None:
                    ref["q_ref"] = np.asarray(out[2], dtype=float).copy()
                if len(out) > 3 and ref["omega_ref"] is None:
                    ref["omega_ref"] = np.asarray(out[3], dtype=float).copy()

            if any(value is not None for value in ref.values()):
                return ref

    return ref


def set_axes_equal(ax, xyz):
    x_min, x_max = np.min(xyz[:, 0]), np.max(xyz[:, 0])
    y_min, y_max = np.min(xyz[:, 1]), np.max(xyz[:, 1])
    z_min, z_max = np.min(xyz[:, 2]), np.max(xyz[:, 2])

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    radius = 0.5 * max(x_max - x_min, y_max - y_min, z_max - z_min)
    radius = max(radius, 1e-6)

    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_zlim(z_mid - radius, z_mid + radius)


def plot_components(t, actual, reference, labels, title, ylabel, save_path):
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 8), sharex=True)

    if len(labels) == 1:
        axes = [axes]

    for i, label in enumerate(labels):
        axes[i].plot(t, actual[:, i], label=f"actual {label}")
        if reference is not None and len(reference) == len(actual):
            axes[i].plot(t, reference[:, i], linestyle="--", label=f"ref {label}")
        axes[i].set_ylabel(ylabel)
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel("timestep")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="./models/sac_powerloop_vanilla.pt")
    parser.add_argument("--out_dir", type=str, default="./results/sac_trajectory_3d")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="circle")
    parser.add_argument("--ref_mode", type=str, default=None)
    parser.add_argument("--hidden_size", type=int, default=256)

    args = parser.parse_args()

    if args.ref_mode is not None:
        args.mode = args.ref_mode

    os.makedirs(args.out_dir, exist_ok=True)

    env = make_env(args.seed, args.mode)
    agent = make_agent(env, args.hidden_size)
    load_agent(agent, args.model)

    obs = env.reset()
    done = False

    positions = []
    velocities = []
    quaternions = []
    rpy = []

    references_p = []
    references_v = []
    references_q = []
    references_rpy = []
    references_omega = []

    actions = []
    angular_velocity_cmds = []
    rewards = []
    pos_errors = []
    vel_errors = []
    att_errors = []
    unsafe_steps = []

    while not done:
        action = agent.select_action(obs, evaluate=True)
        next_obs, reward, done, info = env.step(action)

        full_state = get_full_state(env)

        p = full_state[0:3].copy()
        v = full_state[3:6].copy()
        q = normalize_quat(full_state[6:10].copy())

        positions.append(p)
        velocities.append(v)
        quaternions.append(q)
        rpy.append(quat_to_rpy(q))

        ref = get_reference_data(env, info)

        if ref["p_ref"] is not None and len(ref["p_ref"]) >= 3:
            references_p.append(ref["p_ref"][:3].copy())
        else:
            references_p.append(np.full(3, np.nan))

        if ref["v_ref"] is not None and len(ref["v_ref"]) >= 3:
            references_v.append(ref["v_ref"][:3].copy())
        else:
            references_v.append(np.full(3, np.nan))

        if ref["q_ref"] is not None and len(ref["q_ref"]) >= 4:
            q_ref = normalize_quat(ref["q_ref"][:4].copy())
            references_q.append(q_ref)
            references_rpy.append(quat_to_rpy(q_ref))
        else:
            references_q.append(np.full(4, np.nan))
            references_rpy.append(np.full(3, np.nan))

        if ref["omega_ref"] is not None and len(ref["omega_ref"]) >= 3:
            references_omega.append(ref["omega_ref"][:3].copy())
        else:
            references_omega.append(np.full(3, np.nan))

        action = np.asarray(action, dtype=float).copy()
        actions.append(action)

        if len(action) >= 4:
            angular_velocity_cmds.append(action[1:4].copy())
        else:
            angular_velocity_cmds.append(np.full(3, np.nan))

        rewards.append(float(reward))
        pos_errors.append(float(info.get("tracking_error_pos", np.nan)))
        vel_errors.append(float(info.get("tracking_error_v", np.nan)))
        att_errors.append(float(info.get("tracking_error_att", np.nan)))
        unsafe_steps.append(float(info.get("unsafe", 0.0)))

        obs = next_obs

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    quaternions = np.asarray(quaternions, dtype=float)
    rpy = np.asarray(rpy, dtype=float)

    references_p = np.asarray(references_p, dtype=float)
    references_v = np.asarray(references_v, dtype=float)
    references_q = np.asarray(references_q, dtype=float)
    references_rpy = np.asarray(references_rpy, dtype=float)
    references_omega = np.asarray(references_omega, dtype=float)

    actions = np.asarray(actions, dtype=float)
    angular_velocity_cmds = np.asarray(angular_velocity_cmds, dtype=float)

    rewards = np.asarray(rewards, dtype=float)
    pos_errors = np.asarray(pos_errors, dtype=float)
    vel_errors = np.asarray(vel_errors, dtype=float)
    att_errors = np.asarray(att_errors, dtype=float)
    unsafe_steps = np.asarray(unsafe_steps, dtype=float)

    has_p_ref = not np.all(np.isnan(references_p))
    has_v_ref = not np.all(np.isnan(references_v))
    has_q_ref = not np.all(np.isnan(references_q))
    has_omega_ref = not np.all(np.isnan(references_omega))

    att_angle_errors = np.full(len(quaternions), np.nan)
    if has_q_ref:
        for i in range(len(quaternions)):
            if not np.any(np.isnan(references_q[i])):
                att_angle_errors[i] = quat_angle_error(quaternions[i], references_q[i])

    print("---------------------------------------")
    print("SAC 3D trajectory rollout")
    print(f"Model: {args.model}")
    print(f"Steps: {len(positions)}")
    print(f"Total reward: {np.nansum(rewards):.3f}")
    print(f"Mean position error: {np.nanmean(pos_errors):.3f}")
    print(f"Mean velocity error: {np.nanmean(vel_errors):.3f}")
    print(f"Mean attitude error from env: {np.nanmean(att_errors):.3f}")
    if not np.all(np.isnan(att_angle_errors)):
        print(f"Mean quaternion angle error: {np.nanmean(att_angle_errors):.3f} rad")
    print(f"Unsafe steps: {int(np.sum(unsafe_steps > 0.5))}")
    print("---------------------------------------")

    np.savez(
        os.path.join(args.out_dir, "sac_rollout_data.npz"),
        positions=positions,
        velocities=velocities,
        quaternions=quaternions,
        rpy=rpy,
        references_p=references_p,
        references_v=references_v,
        references_q=references_q,
        references_rpy=references_rpy,
        references_omega=references_omega,
        actions=actions,
        angular_velocity_cmds=angular_velocity_cmds,
        rewards=rewards,
        pos_errors=pos_errors,
        vel_errors=vel_errors,
        att_errors=att_errors,
        att_angle_errors=att_angle_errors,
        unsafe=unsafe_steps,
    )

    t = np.arange(len(positions))

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        linewidth=2,
        label="actual trajectory",
    )

    if has_p_ref:
        ax.plot(
            references_p[:, 0],
            references_p[:, 1],
            references_p[:, 2],
            linestyle="--",
            linewidth=2,
            label="reference trajectory",
        )

    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], s=60, label="start")
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], s=60, label="end")

    xyz_for_bounds = positions
    if has_p_ref:
        xyz_for_bounds = np.vstack([positions, references_p])

    x_min, x_max = np.nanmin(xyz_for_bounds[:, 0]), np.nanmax(xyz_for_bounds[:, 0])
    y_min, y_max = np.nanmin(xyz_for_bounds[:, 1]), np.nanmax(xyz_for_bounds[:, 1])

    xx, yy = np.meshgrid(
        np.linspace(x_min - 0.5, x_max + 0.5, 2),
        np.linspace(y_min - 0.5, y_max + 0.5, 2),
    )
    zz = np.ones_like(xx) * 3.0
    ax.plot_surface(xx, yy, zz, alpha=0.15)

    set_axes_equal(ax, xyz_for_bounds)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("SAC Powerloop 3D Trajectory")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "sac_trajectory_3d.png"), dpi=200)
    plt.close(fig)

    plot_components(
        t=t,
        actual=positions,
        reference=references_p if has_p_ref else None,
        labels=["x", "y", "z"],
        title="Position Components",
        ylabel="position [m]",
        save_path=os.path.join(args.out_dir, "sac_position_components.png"),
    )

    plot_components(
        t=t,
        actual=velocities,
        reference=references_v if has_v_ref else None,
        labels=["vx", "vy", "vz"],
        title="Velocity Components",
        ylabel="velocity [m/s]",
        save_path=os.path.join(args.out_dir, "sac_velocity_components.png"),
    )

    plot_components(
        t=t,
        actual=np.rad2deg(rpy),
        reference=np.rad2deg(references_rpy) if has_q_ref else None,
        labels=["roll", "pitch", "yaw"],
        title="Attitude Roll/Pitch/Yaw",
        ylabel="angle [deg]",
        save_path=os.path.join(args.out_dir, "sac_attitude_rpy.png"),
    )

    plot_components(
        t=t,
        actual=quaternions,
        reference=references_q if has_q_ref else None,
        labels=["qw", "qx", "qy", "qz"],
        title="Attitude Quaternion Components",
        ylabel="quaternion",
        save_path=os.path.join(args.out_dir, "sac_attitude_quaternion.png"),
    )

    plot_components(
        t=t,
        actual=angular_velocity_cmds,
        reference=references_omega if has_omega_ref else None,
        labels=["omega_x", "omega_y", "omega_z"],
        title="Angular Velocity / Body Rate Commands",
        ylabel="angular velocity [rad/s]",
        save_path=os.path.join(args.out_dir, "sac_angular_velocity.png"),
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, pos_errors, label="position error")
    ax.plot(t, vel_errors, label="velocity error")
    ax.plot(t, att_errors, label="attitude error from env")
    if not np.all(np.isnan(att_angle_errors)):
        ax.plot(t, att_angle_errors, label="quaternion angle error")
    ax.set_xlabel("timestep")
    ax.set_ylabel("error")
    ax.set_title("SAC Tracking Errors")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "sac_tracking_errors.png"), dpi=200)
    plt.close(fig)

    if actions.ndim == 2 and actions.shape[1] >= 4:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t, actions[:, 0], label="a_cmd")
        ax.plot(t, actions[:, 1], label="omega_x")
        ax.plot(t, actions[:, 2], label="omega_y")
        ax.plot(t, actions[:, 3], label="omega_z")
        ax.set_xlabel("timestep")
        ax.set_ylabel("action")
        ax.set_title("SAC Actions")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "sac_actions.png"), dpi=200)
        plt.close(fig)

    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()