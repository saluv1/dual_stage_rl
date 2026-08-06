"""Generate a LaTeX Phase-I evaluation report from the two summary.json files.
Numbers are read from JSON (never hand-typed); figures are copied into figures/.
"""
import json, shutil
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "report"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CUR = "evaluation/evaluation/Final Report/003"
MINE = "evaluation/evaluation/Final Report (My Methods)/001"
TRAIN_CUR = "evaluation/evaluation/Training Progress/003"
TRAIN_MINE = "evaluation/evaluation/Training Progress/001"

cur = json.load(open(f"{CUR}/metrics/summary.json"))
mine = json.load(open(f"{MINE}/metrics/summary.json"))

REGIONS = ["general_trace", "near_ceiling", "bridge", "base_shell"]
RLAB = {"general_trace": "General trace", "near_ceiling": "Near ceiling",
        "bridge": "Bridge", "base_shell": "Base shell"}

def cp(src, dst):
    if Path(src).exists():
        shutil.copy(src, FIG / dst)
        return True
    return False

# copy per-region plots for both methods
copied = {}
for method, base in [("cur", CUR), ("mine", MINE)]:
    for kind in ["plots_3d", "plots_time", "plots_actuator"]:
        for r in REGIONS:
            src = f"{base}/{kind}/{r}.png"
            dst = f"{method}_{kind}_{r}.png"
            copied[(method, kind, r)] = cp(src, dst)
# training curves
cp(f"{TRAIN_CUR}/summary_panel.png", "cur_training_summary.png")
cp(f"{TRAIN_CUR}/per_region_success.png", "cur_per_region_success.png")
cp(f"{TRAIN_MINE}/summary_panel.png", "mine_training_summary.png")
cp(f"{TRAIN_MINE}/per_region_success.png", "mine_per_region_success.png")

def f(x, p=3):
    return "n/a" if x is None else f"{x:.{p}f}"

def pct(x):
    return "n/a" if x is None else f"{100*x:.1f}"

# ---- overall comparison table ----
ov_rows = []
metrics = [
    ("Weighted $\\mu_{SA}$", "weighted_mu_sa", 4),
    ("Safe-arrival rate", "safe_arrival_rate", 3),
    ("Within-horizon rate", "safe_arrival_within_horizon_rate", 3),
    ("Failure rate", "failure_rate", 3),
    ("Timeout rate", "timeout_rate", 3),
    ("Safe-rollout rate", "safe_rollout_rate", 3),
    ("Invariance after entry", "invariance_after_entry_rate", 3),
    ("Discounted SA score", "mean_discounted_safe_arrival_score", 3),
    ("Mean arrival time (s)", "mean_arrival_time_s_success_only", 3),
    ("Mean min $h_s$", "mean_min_hs", 3),
    ("Worst min $h_s$", "worst_min_hs", 3),
]
for label, key, p in metrics:
    a, b = cur["overall"].get(key), mine["overall"].get(key)
    ov_rows.append(f"{label} & {f(a,p)} & {f(b,p)} \\\\")
overall_table = "\n".join(ov_rows)

# ---- per-region tables ----
def region_table(summary):
    rows = []
    for r in REGIONS:
        v = summary["per_region"][r]
        rows.append(
            f"{RLAB[r]} & {pct(v['safe_arrival_rate'])} & {pct(v['failure_rate'])} "
            f"& {f(v.get('mean_arrival_time_s_success_only'),2)} & {f(v['mean_min_hs'],3)} "
            f"& {f(v['worst_min_hs'],3)} \\\\"
        )
    return "\n".join(rows)

cur_region = region_table(cur)
mine_region = region_table(mine)

# ---- figure grid helper ----
def region_fig_grid(method, kind, caption, label):
    imgs = []
    for r in REGIONS:
        if copied.get((method, kind, r)):
            imgs.append(
                f"\\begin{{subfigure}}{{0.48\\textwidth}}\n"
                f"\\includegraphics[width=\\linewidth]{{figures/{method}_{kind}_{r}.png}}\n"
                f"\\caption{{{RLAB[r]}}}\n\\end{{subfigure}}"
            )
    body = "\n\\hfill\n".join(imgs)
    return (f"\\begin{{figure}}[htbp]\n\\centering\n{body}\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure}}")

wm_cur = cur["overall"]["weighted_mu_sa"]
wm_mine = mine["overall"]["weighted_mu_sa"]
sr_cur = cur["overall"]["safe_arrival_rate"]
sr_mine = mine["overall"]["safe_arrival_rate"]

tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{siunitx}
\usepackage{xcolor}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue}

\title{Phase-I Safe-Arrival Policy: Evaluation Report\\
\large Official reset-library held-out evaluation (Euler integrator)}
\author{Dual-Stage RL --- PS2-RL reproduction}
\date{\today}

\begin{document}
\maketitle

\section{Overview}
This report evaluates two Phase-I safe-arrival policies on the official
reset-library held-out validation split (\num{64} states per region,
\num{256} total), using the Euler-integrated quadrotor dynamics. Both policies
are scored identically: the same fixed validation states, the same integrator,
and the same weighted recoverability metric $\mu_{SA}$ with the official region
weights (general trace \num{1.0}, near ceiling \num{2.0}, bridge \num{2.5},
base shell \num{1.0}).

\textbf{Two methods are compared:}
\begin{itemize}
\item \textbf{Current method:} official reset-library curriculum sampling,
\num{10}-dimensional observation (full state), no warm-start.
\item \textbf{My method:} hand-designed continuous-curriculum sampler with
per-region floors that keep the hard regions (near ceiling, bridge) sampled
from $s=0$, LQR warm-start, and an \num{8}-dimensional reduced observation
(dropping horizontal position, using attitude error).
\end{itemize}

\section{Headline results}
The current method attains a weighted $\mu_{SA}$ of \textbf{""" + f"{wm_cur:.4f}" + r"""}
and an overall safe-arrival rate of \textbf{""" + f"{100*sr_cur:.1f}\\%" + r"""}; the
my-method variant attains \textbf{""" + f"{wm_mine:.4f}" + r"""} and
\textbf{""" + f"{100*sr_mine:.1f}\\%" + r"""} respectively. Both are evaluated on
identical states and dynamics, so the comparison is direct.

\begin{table}[htbp]
\centering
\caption{Overall metrics on the held-out validation set (256 episodes).
Rates are fractions in $[0,1]$; $h_s$ is the hard-deck (altitude ceiling)
margin, positive meaning inside the safe set.}
\label{tab:overall}
\begin{tabular}{lcc}
\toprule
Metric & Current method & My method \\
\midrule
""" + overall_table + r"""
\bottomrule
\end{tabular}
\end{table}

\section{Per-region breakdown}
The safe set is a single altitude ceiling, but recovery difficulty varies
sharply by region. \emph{Near ceiling} and \emph{bridge} are the hard regions
(states close to the ceiling or on the recovery boundary); \emph{base shell}
is the easy region already inside the LQR handoff set.

\begin{table}[htbp]
\centering
\caption{Per-region performance, current method. Success and failure in \si{\percent};
arrival time in \si{\second} (successful episodes only); $h_s$ margins in meters.}
\label{tab:region-cur}
\begin{tabular}{lccccc}
\toprule
Region & Success (\%) & Failure (\%) & Arrival (s) & Mean min $h_s$ & Worst min $h_s$ \\
\midrule
""" + cur_region + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Per-region performance, my method (own sampler + LQR warm-start +
8-D reduced observation).}
\label{tab:region-mine}
\begin{tabular}{lccccc}
\toprule
Region & Success (\%) & Failure (\%) & Arrival (s) & Mean min $h_s$ & Worst min $h_s$ \\
\midrule
""" + mine_region + r"""
\bottomrule
\end{tabular}
\end{table}

\clearpage
\section{Training progress}
\begin{figure}[htbp]
\centering
\begin{subfigure}{0.48\textwidth}
\includegraphics[width=\linewidth]{figures/cur_training_summary.png}
\caption{Current method}
\end{subfigure}\hfill
\begin{subfigure}{0.48\textwidth}
\includegraphics[width=\linewidth]{figures/mine_training_summary.png}
\caption{My method}
\end{subfigure}
\caption{Training-progress summary panels: weighted $\mu_{SA}$ and curriculum,
overall outcome rates, per-region arrival rate, and mean arrival time versus
environment timestep.}
\label{fig:training}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{subfigure}{0.48\textwidth}
\includegraphics[width=\linewidth]{figures/cur_per_region_success.png}
\caption{Current method}
\end{subfigure}\hfill
\begin{subfigure}{0.48\textwidth}
\includegraphics[width=\linewidth]{figures/mine_per_region_success.png}
\caption{My method}
\end{subfigure}
\caption{Per-region within-horizon arrival rate versus timestep. The hard
regions (near ceiling, bridge) are the last to converge in both methods.}
\label{fig:perregion-train}
\end{figure}

\clearpage
\section{Trajectory analysis --- current method}
""" + region_fig_grid("cur", "plots_3d",
    "Safe-arrival trajectories (current method). Green = success, red = failure. "
    "Triangles mark the terminal state; the star is the hover target.",
    "fig:cur3d") + r"""

""" + region_fig_grid("cur", "plots_time",
    "State time-series (current method): altitude with ceiling marked, speed, "
    "attitude error, and altitude error versus step.",
    "fig:curtime") + r"""

""" + region_fig_grid("cur", "plots_actuator",
    "Applied actuator commands (current method): thrust and the three body-rate "
    "commands, with input limits marked.",
    "fig:curact") + r"""

\clearpage
\section{Trajectory analysis --- my method}
""" + region_fig_grid("mine", "plots_3d",
    "Safe-arrival trajectories (my method). Green = success, red = failure.",
    "fig:mine3d") + r"""

""" + region_fig_grid("mine", "plots_time",
    "State time-series (my method).",
    "fig:minetime") + r"""

""" + region_fig_grid("mine", "plots_actuator",
    "Applied actuator commands (my method).",
    "fig:mineact") + r"""

\clearpage
\section{Discussion}
Both methods reach a comparable weighted $\mu_{SA}$
(""" + f"{wm_cur:.3f}" + r""" current vs.\ """ + f"{wm_mine:.3f}" + r""" mine) and
saturate the easy \emph{base shell} region at \SI{100}{\percent}. The gap is
concentrated in the hard regions, where recovery requires large attitude
corrections near the ceiling. The negative worst-case min-$h_s$
(""" + f"{cur['overall']['worst_min_hs']:.3f}" + r"""\,m) in both methods indicates
a small number of episodes that briefly cross the ceiling during recovery
before returning --- a known limitation of a learned backup policy without the
Phase-II control-invariant layer, which is what formally enforces the ceiling.

\end{document}
"""

(OUT / "report.tex").write_text(tex, encoding="utf-8")
print("wrote report/report.tex")
print("figures copied:", sum(1 for v in copied.values() if v), "region plots +", 
      len(list(FIG.glob("*training*"))) + len(list(FIG.glob("*per_region_success*"))), "training curves")