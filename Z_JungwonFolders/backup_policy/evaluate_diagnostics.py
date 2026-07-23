import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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
    reset_dynamics_state
)

# Try importing PS2-like sampler functions from your current train.py.
# If they are not there, the script will still evaluate the simple sampler.
try:
    from backup_policy.train import (
        generate_reference_trace,
        classify_trace_states,
        sample_initial_state as sample_initial_state_ps2
    )
    HAS_PS2_SAMPLER = True
except Exception:
    HAS_PS2_SAMPLER = False


# ---------------------------------------------------------------------
# Simple old curriculum sampler
# ---------------------------------------------------------------------

def quat_normalize(q):

    q = np.array(q, dtype=float)
    q_norm = np.linalg.norm(q)

    if q_norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])

    return q / q_norm


def simple_curriculum_ranges(curriculum_level):

    if curriculum_level == 0:
        pz_range = (1.8, 2.4)
        v_range = 0.35
        q_range = 0.035

    elif curriculum_level == 1:
        pz_range = (1.6, 2.6)
        v_range = 0.60
        q_range = 0.060

    elif curriculum_level == 2:
        pz_range = (1.4, 2.8)
        v_range = 1.00
        q_range = 0.100

    else:
        pz_range = (1.2, 2.95)
        v_range = 1.40
        q_range = 0.140

    return pz_range, v_range, q_range


def sample_initial_state_simple(sets, curriculum_level, rng, max_tries=5000):

    pz_range, v_range, q_range = simple_curriculum_ranges(curriculum_level)

    for _ in range(max_tries):

        px = rng.uniform(-0.25, 0.25)
        py = rng.uniform(-0.25, 0.25)
        pz = rng.uniform(pz_range[0], pz_range[1])

        vx = rng.uniform(-v_range, v_range)
        vy = rng.uniform(-v_range, v_range)
        vz = rng.uniform(-v_range, v_range)

        qx = rng.uniform(-q_range, q_range)
        qy = rng.uniform(-q_range, q_range)
        qz = rng.uniform(-q_range, q_range)

        qw_sq = 1.0 - qx**2 - qy**2 - qz**2

        if qw_sq <= 0.0:
            continue

        qw = np.sqrt(qw_sq)

        q = quat_normalize(np.array([qw, qx, qy, qz]))

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

        b, f, c = compute_bfc(sets, state)

        if c == 1.0:
            return state

    raise RuntimeError("Failed to sample a valid simple-curriculum initial state.")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def state_metrics(sets, state, P):

    reduced_state = compute_reduced_state(state)

    sets.compute_hb(reduced_state)
    sets.compute_hs(state)

    hb = sets.hb
    hs = sets.hs

    pz = state[2]
    v_norm = np.linalg.norm(state[3:6])
    att_err_norm = np.linalg.norm(reduced_state[4:7])
    ellipsoid_value = reduced_state @ P @ reduced_state

    return {
        "hB": hb,
        "hS": hs,
        "pz": pz,
        "v_norm": v_norm,
        "att_err_norm": att_err_norm,
        "ellipsoid_value": ellipsoid_value
    }


def rollout_policy(
        policy,
        dyn,
        sets,
        P,
        initial_state,
        max_episode_steps=300
):

    state = initial_state.copy()
    reset_dynamics_state(dyn, state)

    states = []
    reduced_states = []
    actions = []

    hb_list = []
    hs_list = []
    ellipsoid_list = []
    v_norm_list = []
    att_err_list = []

    outcome = "timeout"
    terminal_step = max_episode_steps

    for step in range(max_episode_steps):

        states.append(state.copy())

        reduced_state = compute_reduced_state(state)
        reduced_states.append(reduced_state.copy())

        metrics = state_metrics(sets, state, P)
        hb_list.append(metrics["hB"])
        hs_list.append(metrics["hS"])
        ellipsoid_list.append(metrics["ellipsoid_value"])
        v_norm_list.append(metrics["v_norm"])
        att_err_list.append(metrics["att_err_norm"])

        action_norm = policy.select_action(np.array(state))
        action_norm = np.clip(action_norm, -1.0, 1.0)

        action = scale_action(action_norm, dyn.g)
        actions.append(action.copy())

        next_state = dyn.step(action).copy()

        b_next, f_next, c_next = compute_bfc(sets, next_state)

        state = next_state.copy()

        if b_next == 1.0:
            outcome = "success"
            terminal_step = step + 1
            states.append(state.copy())

            reduced_state = compute_reduced_state(state)
            reduced_states.append(reduced_state.copy())

            metrics = state_metrics(sets, state, P)
            hb_list.append(metrics["hB"])
            hs_list.append(metrics["hS"])
            ellipsoid_list.append(metrics["ellipsoid_value"])
            v_norm_list.append(metrics["v_norm"])
            att_err_list.append(metrics["att_err_norm"])

            break

        if f_next == 1.0:
            outcome = "failure"
            terminal_step = step + 1
            states.append(state.copy())

            reduced_state = compute_reduced_state(state)
            reduced_states.append(reduced_state.copy())

            metrics = state_metrics(sets, state, P)
            hb_list.append(metrics["hB"])
            hs_list.append(metrics["hS"])
            ellipsoid_list.append(metrics["ellipsoid_value"])
            v_norm_list.append(metrics["v_norm"])
            att_err_list.append(metrics["att_err_norm"])

            break

    result = {
        "states": np.array(states),
        "reduced_states": np.array(reduced_states),
        "actions": np.array(actions),
        "hb": np.array(hb_list),
        "hs": np.array(hs_list),
        "ellipsoid": np.array(ellipsoid_list),
        "v_norm": np.array(v_norm_list),
        "att_err_norm": np.array(att_err_list),
        "outcome": outcome,
        "terminal_step": terminal_step
    }

    return result


def evaluate_sampler(
        policy,
        sets,
        P,
        sampler_name,
        sample_fn,
        curriculum_level,
        rng,
        n_episodes=100,
        max_episode_steps=300
):

    dyn = Dynamics()

    rows = []
    rollouts = []

    for ep in range(n_episodes):

        initial_state = sample_fn(sets, curriculum_level, rng)

        init_metrics = state_metrics(sets, initial_state, P)

        rollout = rollout_policy(
            policy=policy,
            dyn=dyn,
            sets=sets,
            P=P,
            initial_state=initial_state,
            max_episode_steps=max_episode_steps
        )

        final_state = rollout["states"][-1]
        final_metrics = state_metrics(sets, final_state, P)

        row = {
            "sampler": sampler_name,
            "curriculum_level": curriculum_level,
            "episode": ep,
            "outcome": rollout["outcome"],
            "terminal_step": rollout["terminal_step"],

            "init_hB": init_metrics["hB"],
            "init_hS": init_metrics["hS"],
            "init_pz": init_metrics["pz"],
            "init_v_norm": init_metrics["v_norm"],
            "init_att_err_norm": init_metrics["att_err_norm"],
            "init_ellipsoid": init_metrics["ellipsoid_value"],

            "final_hB": final_metrics["hB"],
            "final_hS": final_metrics["hS"],
            "final_pz": final_metrics["pz"],
            "final_v_norm": final_metrics["v_norm"],
            "final_att_err_norm": final_metrics["att_err_norm"],
            "final_ellipsoid": final_metrics["ellipsoid_value"],

            "min_hS": np.min(rollout["hs"]),
            "max_hB": np.max(rollout["hb"]),
            "final_minus_init_hB": final_metrics["hB"] - init_metrics["hB"],
            "final_minus_init_v_norm": final_metrics["v_norm"] - init_metrics["v_norm"],
            "final_minus_init_att_err": final_metrics["att_err_norm"] - init_metrics["att_err_norm"],
        }

        rows.append(row)
        rollouts.append(rollout)

    df = pd.DataFrame(rows)

    return df, rollouts


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def outcome_rates(df):

    total = len(df)

    success = np.mean(df["outcome"] == "success")
    failure = np.mean(df["outcome"] == "failure")
    timeout = np.mean(df["outcome"] == "timeout")

    avg_steps = np.mean(df["terminal_step"])
    avg_init_hB = np.mean(df["init_hB"])
    avg_final_hB = np.mean(df["final_hB"])
    avg_min_hS = np.mean(df["min_hS"])

    return {
        "n": total,
        "success_rate": success,
        "failure_rate": failure,
        "timeout_rate": timeout,
        "avg_steps": avg_steps,
        "avg_init_hB": avg_init_hB,
        "avg_final_hB": avg_final_hB,
        "avg_min_hS": avg_min_hS
    }


def plot_rates(summary_df, out_dir):

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = []
    x = np.arange(len(summary_df))

    for _, row in summary_df.iterrows():
        labels.append(f"{row['sampler']}\nL{int(row['curriculum_level'])}")

    ax.bar(x - 0.25, summary_df["success_rate"], width=0.25, label="success")
    ax.bar(x, summary_df["failure_rate"], width=0.25, label="failure")
    ax.bar(x + 0.25, summary_df["timeout_rate"], width=0.25, label="timeout")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylabel("rate")
    ax.set_title("Safe-Arrival Evaluation Outcome Rates")
    ax.grid(True, axis="y")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "01_outcome_rates.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_initial_scatter(df, out_dir):

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    outcomes = ["success", "failure", "timeout"]

    for outcome in outcomes:
        mask = df["outcome"] == outcome

        axs[0].scatter(
            df.loc[mask, "init_hB"],
            df.loc[mask, "init_v_norm"],
            label=outcome,
            alpha=0.75
        )

        axs[1].scatter(
            df.loc[mask, "init_hB"],
            df.loc[mask, "init_att_err_norm"],
            label=outcome,
            alpha=0.75
        )

        axs[2].scatter(
            df.loc[mask, "init_pz"],
            df.loc[mask, "init_hB"],
            label=outcome,
            alpha=0.75
        )

    axs[0].axvline(0.0, linestyle="--")
    axs[0].set_xlabel("initial h_B")
    axs[0].set_ylabel("initial |v|")

    axs[1].axvline(0.0, linestyle="--")
    axs[1].set_xlabel("initial h_B")
    axs[1].set_ylabel("initial attitude error norm")

    axs[2].axhline(0.0, linestyle="--")
    axs[2].axvline(2.0, linestyle="--")
    axs[2].set_xlabel("initial p_z")
    axs[2].set_ylabel("initial h_B")

    axs[0].legend()

    fig.suptitle("Initial State Difficulty vs Outcome")
    plt.tight_layout()

    path = os.path.join(out_dir, "02_initial_difficulty_scatter.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_improvement_scatter(df, out_dir):

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    outcomes = ["success", "failure", "timeout"]

    for outcome in outcomes:
        mask = df["outcome"] == outcome

        axs[0].scatter(
            df.loc[mask, "init_hB"],
            df.loc[mask, "final_hB"],
            label=outcome,
            alpha=0.75
        )

        axs[1].scatter(
            df.loc[mask, "init_v_norm"],
            df.loc[mask, "final_v_norm"],
            label=outcome,
            alpha=0.75
        )

        axs[2].scatter(
            df.loc[mask, "init_att_err_norm"],
            df.loc[mask, "final_att_err_norm"],
            label=outcome,
            alpha=0.75
        )

    axs[0].axhline(0.0, linestyle="--")
    axs[0].axvline(0.0, linestyle="--")
    axs[0].set_xlabel("initial h_B")
    axs[0].set_ylabel("final h_B")

    axs[1].plot(
        [0.0, max(df["init_v_norm"].max(), df["final_v_norm"].max())],
        [0.0, max(df["init_v_norm"].max(), df["final_v_norm"].max())],
        linestyle="--"
    )
    axs[1].set_xlabel("initial |v|")
    axs[1].set_ylabel("final |v|")

    axs[2].plot(
        [0.0, max(df["init_att_err_norm"].max(), df["final_att_err_norm"].max())],
        [0.0, max(df["init_att_err_norm"].max(), df["final_att_err_norm"].max())],
        linestyle="--"
    )
    axs[2].set_xlabel("initial attitude error norm")
    axs[2].set_ylabel("final attitude error norm")

    axs[0].legend()

    fig.suptitle("Does the Policy Move States Toward the Base Set?")
    plt.tight_layout()

    path = os.path.join(out_dir, "03_improvement_scatter.png")
    plt.savefig(path, dpi=200)
    plt.close()


def pick_representative_rollouts(df, rollouts, max_per_outcome=3):

    selected = []

    for outcome in ["success", "failure", "timeout"]:

        idxs = df.index[df["outcome"] == outcome].tolist()

        if len(idxs) == 0:
            continue

        idxs = idxs[:max_per_outcome]

        for idx in idxs:
            selected.append((idx, outcome, rollouts[idx]))

    return selected


def plot_time_histories(df, rollouts, out_dir, name):

    selected = pick_representative_rollouts(df, rollouts, max_per_outcome=3)

    if len(selected) == 0:
        return

    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    for idx, outcome, rollout in selected:

        states = rollout["states"]
        time = np.arange(len(states)) * 0.02

        label = f"ep {idx} {outcome}"

        axs[0].plot(time, rollout["hb"], label=label)
        axs[1].plot(time, rollout["hs"], label=label)
        axs[2].plot(time, states[:, 2], label=label)
        axs[3].plot(time, rollout["v_norm"], label=label)
        axs[4].plot(time, rollout["att_err_norm"], label=label)

    axs[0].axhline(0.0, linestyle="--")
    axs[0].set_ylabel("h_B")
    axs[0].set_title("Base Set Margin")

    axs[1].axhline(0.0, linestyle="--")
    axs[1].set_ylabel("h_S")
    axs[1].set_title("Safety Margin")

    axs[2].axhline(2.0, linestyle="--")
    axs[2].set_ylabel("p_z")
    axs[2].set_title("Altitude / z-position")

    axs[3].set_ylabel("|v|")
    axs[3].set_title("Velocity Norm")

    axs[4].set_ylabel("|att err|")
    axs[4].set_xlabel("time [s]")
    axs[4].set_title("Attitude Error Norm")

    axs[0].legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    path = os.path.join(out_dir, f"04_time_histories_{name}.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_3d_rollouts(df, rollouts, out_dir, name):

    selected = pick_representative_rollouts(df, rollouts, max_per_outcome=4)

    if len(selected) == 0:
        return

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for idx, outcome, rollout in selected:

        states = rollout["states"]

        x = states[:, 0]
        y = states[:, 1]
        z = states[:, 2]

        ax.plot(x, y, z, label=f"ep {idx} {outcome}")
        ax.scatter(x[0], y[0], z[0], marker="o")
        ax.scatter(x[-1], y[-1], z[-1], marker="x")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"Representative 3D Rollouts: {name}")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    plt.tight_layout()

    path = os.path.join(out_dir, f"05_3d_rollouts_{name}.png")
    plt.savefig(path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="./models/td3_safe_arrival")
    parser.add_argument("--out_dir", type=str, default="./results/safe_arrival_diagnostics")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_episode_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rng = np.random.default_rng(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    dyn = Dynamics()

    gains = LQRGain(dt=dyn.del_t, g=dyn.g)
    K, P = gains.gain()

    sets = SetIndicator(P=P, c_b=8.0, zceil=3.0)

    policy = TD3(
        state_dim=10,
        action_dim=4,
        max_action=1.0,
        discount=0.99,
        tau=0.0025,
        policy_noise=0.10,
        noise_clip=0.10,
        policy_freq=2,
        actor_lr=1e-4,
        critic_lr=3e-4
    )

    policy.load(args.model)

    all_dfs = []
    all_rollout_sets = {}

    print("---------------------------------------")
    print("Evaluating simple hover-centered sampler")
    print("---------------------------------------")

    for level in range(4):

        def sample_fn(sets_local, level_local, rng_local):
            return sample_initial_state_simple(
                sets=sets_local,
                curriculum_level=level_local,
                rng=rng_local
            )

        df, rollouts = evaluate_sampler(
            policy=policy,
            sets=sets,
            P=P,
            sampler_name="simple",
            sample_fn=sample_fn,
            curriculum_level=level,
            rng=rng,
            n_episodes=args.episodes,
            max_episode_steps=args.max_episode_steps
        )

        all_dfs.append(df)
        all_rollout_sets[f"simple_L{level}"] = (df, rollouts)

        plot_time_histories(df, rollouts, args.out_dir, f"simple_L{level}")
        plot_3d_rollouts(df, rollouts, args.out_dir, f"simple_L{level}")

        print(f"simple L{level}:")
        print(outcome_rates(df))

    if HAS_PS2_SAMPLER:

        print("---------------------------------------")
        print("Building PS2-like powerloop-centered regions")
        print("---------------------------------------")

        trace_states = generate_reference_trace(
            n_points=200,
            n_variants=20,
            seed=args.seed
        )

        regions = classify_trace_states(
            sets=sets,
            trace_states=trace_states,
            P=P,
            c_b=8.0,
            near_ceiling_margin=0.25,
            capture_shell_mult=1.5
        )

        print("Region sizes:")
        for name, states in regions.items():
            print(f"  {name}: {len(states)}")

        print("---------------------------------------")
        print("Evaluating PS2-like sampler")
        print("---------------------------------------")

        for level in range(4):

            def sample_fn_ps2(sets_local, level_local, rng_local):
                return sample_initial_state_ps2(
                    sets=sets_local,
                    regions=regions,
                    curriculum_level=level_local,
                    rng=rng_local,
                    max_curriculum_level=3
                )

            df, rollouts = evaluate_sampler(
                policy=policy,
                sets=sets,
                P=P,
                sampler_name="ps2_like",
                sample_fn=sample_fn_ps2,
                curriculum_level=level,
                rng=rng,
                n_episodes=args.episodes,
                max_episode_steps=args.max_episode_steps
            )

            all_dfs.append(df)
            all_rollout_sets[f"ps2_like_L{level}"] = (df, rollouts)

            plot_time_histories(df, rollouts, args.out_dir, f"ps2_like_L{level}")
            plot_3d_rollouts(df, rollouts, args.out_dir, f"ps2_like_L{level}")

            print(f"ps2_like L{level}:")
            print(outcome_rates(df))

    full_df = pd.concat(all_dfs, ignore_index=True)

    summary_rows = []

    for (sampler, level), group in full_df.groupby(["sampler", "curriculum_level"]):

        rates = outcome_rates(group)

        row = {
            "sampler": sampler,
            "curriculum_level": level,
            **rates
        }

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    per_episode_path = os.path.join(args.out_dir, "per_episode_results.csv")
    summary_path = os.path.join(args.out_dir, "summary_results.csv")
    npz_path = os.path.join(args.out_dir, "diagnostic_arrays.npz")

    full_df.to_csv(per_episode_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_rates(summary_df, args.out_dir)
    plot_initial_scatter(full_df, args.out_dir)
    plot_improvement_scatter(full_df, args.out_dir)

    # Save compact rollout arrays for the first few cases of each setting.
    save_dict = {}

    for key, (df, rollouts) in all_rollout_sets.items():

        for i in range(min(5, len(rollouts))):

            save_dict[f"{key}_ep{i}_states"] = rollouts[i]["states"]
            save_dict[f"{key}_ep{i}_hb"] = rollouts[i]["hb"]
            save_dict[f"{key}_ep{i}_hs"] = rollouts[i]["hs"]
            save_dict[f"{key}_ep{i}_v_norm"] = rollouts[i]["v_norm"]
            save_dict[f"{key}_ep{i}_att_err_norm"] = rollouts[i]["att_err_norm"]

    np.savez(npz_path, **save_dict)

    print("---------------------------------------")
    print("Diagnostics saved.")
    print(f"Output directory: {args.out_dir}")
    print(f"Per-episode CSV: {per_episode_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Arrays: {npz_path}")
    print("---------------------------------------")

    print("")
    print("Summary:")
    print(summary_df)


if __name__ == "__main__":
    main()