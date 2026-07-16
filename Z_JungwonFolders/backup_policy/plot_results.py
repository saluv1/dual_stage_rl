import numpy as np
import matplotlib.pyplot as plt


def plot_results():

    data = np.load("./results/td3_safe_arrival_eval.npy", allow_pickle=True)

    curriculum = []
    success_rate = []
    failure_rate = []
    timeout_rate = []
    avg_steps = []
    avg_min_hs = []

    for row in data:
        curriculum.append(row[0])
        success_rate.append(row[1])
        failure_rate.append(row[2])
        timeout_rate.append(row[3])
        avg_steps.append(row[4])
        avg_min_hs.append(row[5])

    eval_idx = np.arange(len(success_rate))

    plt.figure()
    plt.plot(eval_idx, success_rate, label="Success rate")
    plt.plot(eval_idx, failure_rate, label="Failure rate")
    plt.plot(eval_idx, timeout_rate, label="Timeout rate")
    plt.xlabel("Evaluation index")
    plt.ylabel("Rate")
    plt.title("Safe-Arrival Evaluation Rates")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(eval_idx, avg_steps, label="Average steps")
    plt.xlabel("Evaluation index")
    plt.ylabel("Steps")
    plt.title("Average Steps to Termination")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(eval_idx, avg_min_hs, label="Average min h_S")
    plt.xlabel("Evaluation index")
    plt.ylabel("min h_S")
    plt.title("Safety Margin During Evaluation")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.step(eval_idx, curriculum, where="post", label="Curriculum level")
    plt.xlabel("Evaluation index")
    plt.ylabel("Curriculum level")
    plt.title("Curriculum Progress")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_results()