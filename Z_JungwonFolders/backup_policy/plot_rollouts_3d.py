import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env.dynamics import Dynamics
from bcbf.lqrgain import LQRGain
from bcbf.set_indicator import SetIndicator
from backup_policy.td3 import TD3
from backup_policy.train import (
    compute_reduced_state,
    compute_bfc,
    scale_action,
    reset_dynamics_state,
    sample_initial_state
)


def rollout_policy(policy, dyn, sets, curriculum_level=3, max_episode_steps=300):

    state = sample_initial_state(sets, curriculum_level)
    reset_dynamics_state(dyn, state)

    states = []
    reduced_states = []
    hb_list = []
    hs_list = []
    action_list = []

    success = False
    failure = False
    timeout = False

    for step in range(max_episode_steps):

        states.append(state.copy())

        reduced_state = compute_reduced_state(state)
        reduced_states.append(reduced_state.copy())

        sets.compute_hb(reduced_state)
        sets.compute_hs(state)

        hb_list.append(sets.hb)
        hs_list.append(sets.hs)

        action_norm = policy.select_action(np.array(state))
        action_norm = np.clip(action_norm, -1.0, 1.0)

        action = scale_action(action_norm, dyn.g)
        action_list.append(action.copy())

        next_state = dyn.step(action).copy()

        b_next, f_next, c_next = compute_bfc(sets, next_state)

        state = next_state.copy()

        if b_next == 1.0:
            success = True

            states.append(state.copy())
            reduced_state = compute_reduced_state(state)
            reduced_states.append(reduced_state.copy())

            sets.compute_hb(reduced_state)
            sets.compute_hs(state)

            hb_list.append(sets.hb)
            hs_list.append(sets.hs)

            break

        if f_next == 1.0:
            failure = True

            states.append(state.copy())
            reduced_state = compute_reduced_state(state)
            reduced_states.append(reduced_state.copy())

            sets.compute_hb(reduced_state)
            sets.compute_hs(state)

            hb_list.append(sets.hb)
            hs_list.append(sets.hs)

            break

        if step == max_episode_steps - 1:
            timeout = True

            states.append(state.copy())
            reduced_state = compute_reduced_state(state)
            reduced_states.append(reduced_state.copy())

            sets.compute_hb(reduced_state)
            sets.compute_hs(state)

            hb_list.append(sets.hb)
            hs_list.append(sets.hs)

    states = np.array(states)
    reduced_states = np.array(reduced_states)
    hb_list = np.array(hb_list)
    hs_list = np.array(hs_list)
    action_list = np.array(action_list)

    return states, reduced_states, hb_list, hs_list, action_list, success, failure, timeout


def plot_3d_rollouts(rollouts, z_des=2.0):

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for trial, rollout in enumerate(rollouts):

        states = rollout["states"]

        x = states[:, 0]
        y = states[:, 1]
        z = states[:, 2]

        ax.plot(x, y, z, linewidth=1.8)

        ax.scatter(x[0], y[0], z[0], marker="o", s=45)
        ax.scatter(x[-1], y[-1], z[-1], marker="x", s=60)

    # Draw desired/base altitude plane region only as a visual reference.
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 2),
        np.linspace(y_min, y_max, 2)
    )
    zz = z_des * np.ones_like(xx)

    ax.plot_surface(xx, yy, zz, alpha=0.12)

    ax.set_title("Safe-Arrival Backup Policy Rollouts")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    legend_elements = [
        Line2D([0], [0], color="black", lw=2, label="trajectory"),
        Line2D([0], [0], marker="o", color="black", linestyle="None", label="start"),
        Line2D([0], [0], marker="x", color="black", linestyle="None", label="end"),
        Line2D([0], [0], color="gray", lw=6, alpha=0.25, label="z_des = 2 m plane"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0
    )

    plt.tight_layout()
    plt.show()


def plot_time_histories(rollouts, dyn, z_des=2.0):

    fig1, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    for rollout in rollouts:

        states = rollout["states"]
        time = np.arange(len(states)) * dyn.del_t

        axs[0].plot(time, states[:, 2])
        axs[1].plot(time, states[:, 3])
        axs[1].plot(time, states[:, 4])
        axs[1].plot(time, states[:, 5])
        axs[2].plot(time, rollout["hb"])

    axs[0].axhline(z_des, linestyle="--", linewidth=1.5, label="z_des = 2 m")
    axs[0].set_ylabel("p_z [m]")
    axs[0].set_title("Altitude / Vertical Position Toward Base Set")
    axs[0].legend()

    axs[1].set_ylabel("velocity [m/s]")
    axs[1].set_title("Velocity Components")
    axs[1].legend(["v_x", "v_y", "v_z"])

    axs[2].axhline(0.0, linestyle="--", linewidth=1.5, label="h_B = 0")
    axs[2].set_ylabel("h_B")
    axs[2].set_xlabel("time [s]")
    axs[2].set_title("Base Set Margin")
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    fig2, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    for rollout in rollouts:

        reduced_states = rollout["reduced_states"]
        states = rollout["states"]
        time = np.arange(len(states)) * dyn.del_t

        axs[0].plot(time, reduced_states[:, 4])
        axs[0].plot(time, reduced_states[:, 5])
        axs[0].plot(time, reduced_states[:, 6])

        axs[1].plot(time, rollout["hs"])

        if len(rollout["actions"]) > 0:
            action_time = np.arange(len(rollout["actions"])) * dyn.del_t
            axs[2].plot(action_time, rollout["actions"][:, 0])

    axs[0].set_ylabel("attitude error")
    axs[0].set_title("Reduced Attitude Error Components")
    axs[0].legend(["2q_x", "2q_y", "2q_z"])

    axs[1].axhline(0.0, linestyle="--", linewidth=1.5, label="h_S = 0")
    axs[1].set_ylabel("h_S")
    axs[1].set_title("Safety Margin")
    axs[1].legend()

    axs[2].axhline(dyn.g, linestyle="--", linewidth=1.5, label="hover thrust accel g")
    axs[2].set_ylabel("a_cmd")
    axs[2].set_xlabel("time [s]")
    axs[2].set_title("Thrust Acceleration Command")
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def plot_multiple_rollouts(num_trials=10, curriculum_level=3):

    dyn = Dynamics()

    gains = LQRGain(dt=dyn.del_t, g=dyn.g)
    K, P = gains.gain()

    sets = SetIndicator(
        P=P,
        c_b=8.0,
        zceil=3.0
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

    policy.load("./models/td3_safe_arrival")

    rollouts = []

    success_count = 0
    failure_count = 0
    timeout_count = 0

    final_hb_list = []
    min_hs_list = []
    final_pz_list = []
    final_velocity_norm_list = []
    final_attitude_error_norm_list = []

    while len(rollouts) < num_trials:

        states, reduced_states, hb_list, hs_list, action_list, success, failure, timeout = rollout_policy(
            policy=policy,
            dyn=dyn,
            sets=sets,
            curriculum_level=curriculum_level,
            max_episode_steps=300
        )

        # Keep only successful safe-arrival trials for the plot
        if not success:
            if failure:
                failure_count += 1
            else:
                timeout_count += 1
            continue

        success_count += 1

        rollout = {
            "states": states,
            "reduced_states": reduced_states,
            "hb": hb_list,
            "hs": hs_list,
            "actions": action_list,
            "success": success,
            "failure": failure,
            "timeout": timeout
        }

        rollouts.append(rollout)

        final_hb_list.append(hb_list[-1])
        min_hs_list.append(np.min(hs_list))
        final_pz_list.append(states[-1, 2])

        final_v = states[-1, 3:6]
        final_velocity_norm_list.append(np.linalg.norm(final_v))

        final_attitude_error = reduced_states[-1, 4:7]
        final_attitude_error_norm_list.append(np.linalg.norm(final_attitude_error))

    print("---------------------------------------")
    print(f"Successful rollouts plotted: {len(rollouts)}")
    print(f"Rejected failures while sampling: {failure_count}")
    print(f"Rejected timeouts while sampling: {timeout_count}")
    print(f"Curriculum level: {curriculum_level}")
    print("")
    print("Final state summary for plotted successful rollouts:")
    print(f"Average final h_B: {np.mean(final_hb_list):.3f}")
    print(f"Average min h_S: {np.mean(min_hs_list):.3f}")
    print(f"Average final p_z: {np.mean(final_pz_list):.3f} m")
    print(f"Average final |v|: {np.mean(final_velocity_norm_list):.3f} m/s")
    print(f"Average final attitude error norm: {np.mean(final_attitude_error_norm_list):.3f}")
    print("---------------------------------------")

    plot_3d_rollouts(rollouts, z_des=2.0)
    plot_time_histories(rollouts, dyn, z_des=2.0)


if __name__ == "__main__":

    plot_multiple_rollouts(
        num_trials=10,
        curriculum_level=3
    )