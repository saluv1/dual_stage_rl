import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="./results/sac_powerloop")
    args = parser.parse_args()

    train_path = os.path.join(args.result_dir, "training_log.csv")
    eval_path = os.path.join(args.result_dir, "eval_log.csv")

    if os.path.exists(train_path):
        df = pd.read_csv(train_path)

        plt.figure(figsize=(9, 5))
        plt.plot(df["total_step"], df["episode_reward"])
        plt.xlabel("environment steps")
        plt.ylabel("episode reward")
        plt.title("SAC Powerloop Training Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.result_dir, "training_reward.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(df["total_step"], df["episode_pos_error"], label="position")
        plt.plot(df["total_step"], df["episode_v_error"], label="velocity")
        plt.plot(df["total_step"], df["episode_att_error"], label="attitude")
        plt.xlabel("environment steps")
        plt.ylabel("mean episode error")
        plt.title("SAC Powerloop Tracking Errors")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.result_dir, "training_errors.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(df["total_step"], df["episode_unsafe_steps"])
        plt.xlabel("environment steps")
        plt.ylabel("unsafe steps per episode")
        plt.title("Unsafe Steps During Vanilla SAC Training")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.result_dir, "training_unsafe_steps.png"), dpi=200)
        plt.close()

    if os.path.exists(eval_path):
        df = pd.read_csv(eval_path)

        plt.figure(figsize=(9, 5))
        plt.plot(df["total_step"], df["eval_reward"])
        plt.xlabel("environment steps")
        plt.ylabel("eval reward")
        plt.title("SAC Powerloop Evaluation Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.result_dir, "eval_reward.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(df["total_step"], df["eval_pos_error"], label="position")
        plt.plot(df["total_step"], df["eval_v_error"], label="velocity")
        plt.plot(df["total_step"], df["eval_att_error"], label="attitude")
        plt.xlabel("environment steps")
        plt.ylabel("eval error")
        plt.title("SAC Powerloop Evaluation Errors")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.result_dir, "eval_errors.png"), dpi=200)
        plt.close()

    print(f"Plots saved to {args.result_dir}")


if __name__ == "__main__":
    main()
