import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env.dynamics import Dynamics
from bcbf.lqrgain import LQRGain
from bcbf.set_indicator import SetIndicator
from backup_policy.td3 import TD3
from backup_policy.train import (
    compute_bfc,
    scale_action,
    reset_dynamics_state,
    sample_initial_state
)


def rollout_policy(policy, dyn, sets, curriculum_level=3, max_episode_steps=300):

    state = sample_initial_state(sets, curriculum_level)
    reset_dynamics_state(dyn, state)

    states = []
    actions = []
    hb_list = []
    hs_list = []
    indicator_list = []

    for step in range(max_episode_steps):

        states.append(state.copy())

        action_norm = policy.select_action(np.array(state))
        action_norm = np.clip(action_norm, -1.0, 1.0)

        action = scale_action(action_norm, dyn.g)
        actions.append(action.copy())

        next_state = dyn.step(action).copy()

        b_next, f_next, c_next = compute_bfc(sets, next_state)

        hb_list.append(sets.hb)
        hs_list.append(sets.hs)

        if b_next == 1.0:
            indicator_list.append(0)
        elif f_next == 1.0:
            indicator_list.append(2)
        else:
            indicator_list.append(1)

        state = next_state.copy()

        if b_next == 1.0 or f_next == 1.0:
            states.append(state.copy())
            break

    return np.array(states), np.array(actions), np.array(hb_list), np.array(hs_list), np.array(indicator_list)


if __name__ == "__main__":

    dyn = Dynamics()

    gains = LQRGain(dt=dyn.del_t, g=dyn.g)
    K, P = gains.gain()

    sets = SetIndicator(P=P, c_b=8.0, zceil=3.0)

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

    states, actions, hb_list, hs_list, indicator_list = rollout_policy(
        policy=policy,
        dyn=dyn,
        sets=sets,
        curriculum_level=3,
        max_episode_steps=300
    )

    t = np.arange(states.shape[0]) * dyn.del_t

    plt.figure()
    plt.plot(t, states[:, 2], label="p_z")
    plt.axhline(3.0, linestyle="--", label="ceiling")
    plt.axhline(2.0, linestyle="--", label="z_des")
    plt.xlabel("Time [s]")
    plt.ylabel("z position [m]")
    plt.title("Backup Policy Rollout: Altitude")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(hb_list, label="h_B")
    plt.plot(hs_list, label="h_S")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("Barrier value")
    plt.title("Base and Safe Set Values")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(states[:, 3], label="v_x")
    plt.plot(states[:, 4], label="v_y")
    plt.plot(states[:, 5], label="v_z")
    plt.xlabel("Step")
    plt.ylabel("Velocity [m/s]")
    plt.title("Velocity Rollout")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(actions[:, 0], label="a_cmd")
    plt.plot(actions[:, 1], label="omega_x")
    plt.plot(actions[:, 2], label="omega_y")
    plt.plot(actions[:, 3], label="omega_z")
    plt.xlabel("Step")
    plt.ylabel("Action")
    plt.title("Physical Actions")
    plt.grid()
    plt.legend()
    plt.show()