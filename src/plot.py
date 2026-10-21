"""Regenerate every plot from the raw CSV logs (no retraining needed).

    python plot.py                       # fast tier (default)
    python plot.py --env gz              # the Gazebo validation runs
    python plot.py --layout 3            # which fixed layout the trajectory overlay uses

Reads  results/runs/<env>_<agent>_seed*/{eval.csv,train_episodes.csv,final_eval_traj.npz}
       results/transfer_diag/gz_gt/*_traj.npz: fast-trained policies in Gazebo (trajectory overlay)  [optional]
       results/report/summary.csv: held-out report-layout evaluations (evaluate.py --layouts report)  [optional;
       when present the bar chart and the summary use it instead of the last periodic eval]
Writes results/*.png, results/summary.csv and results/stability.csv. For --env gz the names get a `_gz` suffix.
Learning curves always come from the periodic evaluations, which run on the *tuning* layouts.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from envs.layout import eval_layouts  # noqa: E402
from utils import format_summary_table, load_config, results_dir  # noqa: E402

# two-sided 97.5% Student-t quantiles for df = 1..30 (n seeds -> df = n-1)
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
         2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
C = {"mean": "#1f77b4", "goal": "#2ca02c", "collision": "#d62728", "timeout": "#ff7f0e", "fast": "#1f77b4",
     "gz": "#9467bd", "random": "#7f7f7f", "eps": "#ff7f0e"}


def ci95(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Half-width of the 95% CI of the mean across `axis` (Student-t; 0 if only one sample)."""
    n = x.shape[axis]
    if n < 2:
        return np.zeros(np.delete(np.array(x.shape), axis))
    t = _T975[n - 2] if n - 1 <= len(_T975) else 1.96
    return t * x.std(axis=axis, ddof=1) / np.sqrt(n)


def load_runs(res: Path, env: str, agent: str) -> dict[str, dict[str, pd.DataFrame]]:
    runs: dict[str, dict[str, pd.DataFrame]] = {}
    for d in sorted(glob.glob(str(res / "runs" / f"{env}_{agent}_seed*"))):
        d = Path(d)
        if (d / "eval.csv").exists():
            runs[d.name] = {"eval": pd.read_csv(d / "eval.csv")}
            if (d / "train_episodes.csv").exists():
                runs[d.name]["train"] = pd.read_csv(d / "train_episodes.csv")
    if not runs:
        raise SystemExit(f"no runs found for {env}/{agent} in {res / 'runs'} (train first)")
    return runs


def random_baseline(res: Path, env: str) -> float | None:
    """Random-policy success on the held-out report layouts (the curves use the tuning layouts; the two random rates
    differ by ~1-2 points)."""
    f = res / "report" / "summary.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    row = df[(df["run"] == "random_dqn") & (df["eval_env"] == env)]
    return float(row["success_rate"].iloc[0]) * 100 if len(row) else None


def fig_success(runs: dict, env: str, x: str, out: Path, baseline: float | None, suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    evs = [r["eval"] for r in runs.values()]
    if x == "step":
        grid = np.array(sorted(set.intersection(*[set(e["step"]) for e in evs])))
        Y = np.stack([e.set_index("step").loc[grid, "success_rate"].to_numpy() * 100 for e in evs])
        xlabel = "Training steps (env steps)"
    else:
        hi = min(e["episodes"].max() for e in evs)
        grid = np.linspace(0, hi, 200)
        Y = np.stack([np.interp(grid, e["episodes"], e["success_rate"] * 100) for e in evs])
        xlabel = "Training episodes"
    for y in Y:
        ax.plot(grid, y, color=C["mean"], alpha=0.18, lw=0.9)
    m, h = Y.mean(0), ci95(Y)
    ax.plot(grid, m, color=C["mean"], lw=2.2, label=f"DQN mean over {len(evs)} seeds")
    ax.fill_between(grid, np.clip(m - h, 0, 100), np.clip(m + h, 0, 100), color=C["mean"], alpha=0.25, label="95% CI")
    if baseline is not None:
        ax.axhline(baseline, color=C["random"], ls="--", lw=1.3, label=f"random policy ({baseline:.0f}%, held-out layouts)")
    ax.set(xlabel=xlabel, ylabel="Goal success rate (%)", ylim=(0, 100),
           title=f"Goal success during training ({env} tier, greedy eval, 50 tuning layouts)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / f"goal_success_over_training{'_episodes' if x == 'episodes' else ''}{suffix}.png", dpi=150)
    plt.close(fig)


def final_rows(runs: dict) -> pd.DataFrame:
    return pd.DataFrame([r["eval"].iloc[-1] for r in runs.values()])


def report_rows(res: Path, env: str, agent: str, eval_env: str, runs: dict) -> pd.DataFrame | None:
    """Held-out report-layout evaluations of this tier's final checkpoints (one row per seed), if they exist."""
    f = res / "report" / "summary.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df[(df["eval_env"] == eval_env) & df["run"].isin(list(runs)) & df["run"].str.startswith(f"{env}_{agent}")]
    return df if len(df) == len(runs) else None


def fig_bars(runs: dict, env: str, res: Path, out: Path, suffix: str, agent: str = "dqn") -> None:
    cats = [("success_rate", "Goal reached", C["goal"]), ("collision_rate", "Collision", C["collision"]),
            ("timeout_rate", "Timeout", C["timeout"])]
    cols = [c[0] for c in cats]
    rep = report_rows(res, env, agent, env, runs)
    if rep is not None:
        n_ep = int(rep["episodes"].iloc[0])
        series: list[tuple[str, np.ndarray]] = [(f"{env}-trained, {env} eval ({n_ep} held-out layouts)", rep[cols].to_numpy() * 100)]
        rep_gz = report_rows(res, env, agent, "gz", runs) if env == "fast" else None
        if rep_gz is not None:
            series.append((f"{env}-trained, Gazebo eval (same layouts)", rep_gz[cols].to_numpy() * 100))
        title = "Final checkpoints on held-out report layouts (mean over seeds, 95% CI)"
        _bars(series, cats, out / f"success_vs_collisions_bar{suffix}.png", title)
        return
    series = [(f"{env} tier: final eval", final_rows(runs)[cols].to_numpy() * 100)]
    summ = res / "transfer" / "summary.csv"
    if env == "fast" and summ.exists():
        df = pd.read_csv(summ)
        t = df[(df["eval_env"] == "gz") & df["run"].str.startswith("fast_dqn")]
        if len(t):
            series.append(("fast-trained, evaluated in Gazebo", t[[c[0] for c in cats]].to_numpy() * 100))
    _bars(series, cats, out / f"success_vs_collisions_bar{suffix}.png",
          "Final evaluation outcomes (mean over seeds, 95% CI error bars)")


def _bars(series: list[tuple[str, np.ndarray]], cats: list, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    w = 0.8 / len(series)
    for j, (label, arr) in enumerate(series):
        xs = np.arange(len(cats)) + (j - (len(series) - 1) / 2) * w
        ax.bar(xs, arr.mean(0), w * 0.95, yerr=ci95(arr), capsize=4, label=f"{label} (n={len(arr)})",
               color=[c[2] for c in cats], alpha=0.95 if j == 0 else 0.55, edgecolor="black", hatch="" if j == 0 else "//")
    ax.set_xticks(range(len(cats)), [c[1] for c in cats])
    ax.set(ylabel="Share of evaluation episodes (%)", ylim=(0, 100), title=title)
    handles = [Rectangle((0, 0), 1, 1, fc="white", ec="black", hatch="" if j == 0 else "//", label=s[0] + f" (n={len(s[1])})")
               for j, s in enumerate(series)]
    ax.legend(handles=handles, loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


DIP_POINTS = 15.0   # a "dip": an evaluation more than this many points below the seed's running best


def stability_table(runs: dict, warmup: int = 0) -> pd.DataFrame:
    """Per-seed dips: evaluations after the random-action warm-up that are > DIP_POINTS below the running best."""
    rows = []
    for name, r in runs.items():
        e = r["eval"]
        y = e["success_rate"].to_numpy() * 100
        best = np.maximum.accumulate(y)
        drop = best - y
        after = e["step"].to_numpy() > warmup
        dips = (drop > DIP_POINTS) & after
        rows.append({"run": name, "evals": len(y), "dips": int(dips.sum()),
                     "dip_steps": " ".join(str(int(s)) for s in e["step"][dips]), "max_drop": float(drop[after].max()),
                     "best_success": float(y.max()), "best_step": int(e["step"].iloc[int(y.argmax())]),
                     "final_success": float(y[-1]), "best_minus_final": float(y.max() - y[-1]),
                     "last4_mean": float(y[-4:].mean())})
    return pd.DataFrame(rows)


def fig_success_per_seed(runs: dict, env: str, out: Path, suffix: str, warmup: int = 0) -> None:
    n = len(runs)
    cols = min(5, n)
    rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(3.1 * cols, 2.5 * rws), sharex=True, sharey=True, squeeze=False)
    for ax, (name, r) in zip(axes.flat, runs.items()):
        e = r["eval"]
        y = e["success_rate"].to_numpy() * 100
        best = np.maximum.accumulate(y)
        dips = (best - y > DIP_POINTS) & (e["step"].to_numpy() > warmup)
        ax.plot(e["step"], y, color=C["mean"], lw=1.4, label="success")
        ax.plot(e["step"], best, color="grey", lw=0.9, ls=":", label="running best")
        ax.plot(e["step"][dips], y[dips], "v", color=C["collision"], ms=6, label=f"dip > {DIP_POINTS:.0f} pts")
        ax.set_title(f"{name.split('_')[-1]}: {int(dips.sum())} dip(s)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 100)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Training steps")
    for ax in axes[:, 0]:
        ax.set_ylabel("Success (%)")
    axes.flat[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Per-seed goal success during training ({env} tier, 50 tuning layouts per eval; "
                 f"dips counted after the {warmup // 1000}k warm-up)")
    fig.tight_layout()
    fig.savefig(out / f"success_per_seed{suffix}.png", dpi=140)
    plt.close(fig)


def fig_q_stability(runs: dict, env: str, out: Path, suffix: str) -> None:
    """TD loss, mean max-Q, gradient norm (training batches) and the eval overestimation probe, per seed."""
    panels = [("upd_loss_mean", "TD loss (Huber), interval mean"), ("upd_q_max_mean_mean", "mean max_a Q(s,a), batches"),
              ("upd_grad_norm_mean", "grad norm before clip, interval mean"), ("q0_minus_G", "Q(s0) - realised disc. return (eval)")]
    have = [r for r in runs.values() if "upd_loss_mean" in r["eval"]]
    if not have:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (col, title) in zip(axes.flat, panels):
        for name, r in runs.items():
            e = r["eval"]
            if col == "q0_minus_G":
                if "q0_mean" not in e:
                    continue
                y = e["q0_mean"] - e["disc_return_mean"]
            elif col in e:
                y = e[col]
            else:
                continue
            ax.plot(e["step"], y, lw=1.1, alpha=0.8, label=name.split("_")[-1])
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 1].set_yscale("symlog")
    axes[1, 1].axhline(0, color="k", lw=0.8)
    for ax in axes[1]:
        ax.set_xlabel("Training steps")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Q-learning stability diagnostics ({env} tier; logging only, vanilla DQN)")
    fig.tight_layout()
    fig.savefig(out / f"q_stability{suffix}.png", dpi=140)
    plt.close(fig)


def fig_return(runs: dict, env: str, out: Path, bin_w: int, suffix: str) -> None:
    curves = []
    for r in runs.values():
        t = r.get("train")
        if t is None or t.empty:
            continue
        b = (t["step"] - 1) // bin_w
        curves.append(t.groupby(b)["return"].mean())
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    if curves:
        n = min(len(c) for c in curves)
        Y = np.stack([c.iloc[:n].to_numpy() for c in curves])
        xs = (np.arange(n) + 0.5) * bin_w
        m, h = Y.mean(0), ci95(Y)
        ax.plot(xs, m, color=C["mean"], lw=2.2, label=f"training episode return (mean of {len(curves)} seeds)")
        ax.fill_between(xs, m - h, m + h, color=C["mean"], alpha=0.25, label="95% CI")
    ev = [r["eval"].set_index("step")["mean_return"] for r in runs.values()]
    grid = sorted(set.intersection(*[set(e.index) for e in ev]))
    ax.plot(grid, np.mean([e.loc[grid].to_numpy() for e in ev], axis=0), color="black", ls=":", lw=1.6,
            label="greedy eval return (mean)")
    ax.set(xlabel="Training steps (env steps)", ylabel="Mean episode return",
           title=f"Training return and exploration schedule ({env} tier)")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    eps = pd.concat([r["eval"].set_index("step")["epsilon"] for r in runs.values()], axis=1).mean(axis=1)
    ax2.plot(eps.index, eps.values, color=C["eps"], lw=1.6, ls="--", label="epsilon")
    ax2.set(ylabel="epsilon", ylim=(0, 1.05))
    l1, n1 = ax.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, n1 + n2, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"training_return{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_traj(env: str, res: Path, out: Path, layout_idx: int, cfg: dict, suffix: str) -> None:
    layout = eval_layouts(cfg)[layout_idx]
    half, oc = cfg["arena"]["size"] / 2.0, cfg["obstacles"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(Rectangle((-half, -half), 2 * half, 2 * half, fill=False, lw=2, ec="#444"))
    for o in layout.obstacles:
        if o.kind == "cylinder":
            ax.add_patch(Circle((o.x, o.y), oc["cylinder_radius"], fc="#d9826b", ec="k"))
        else:
            s = oc["box_size"]
            ax.add_patch(Rectangle((-s / 2, -s / 2), s, s, fc="#6b8fd9", ec="k",
                                   transform=matplotlib.transforms.Affine2D().rotate(o.yaw).translate(o.x, o.y) + ax.transData))
    ax.add_patch(Circle(layout.goal, cfg["task"]["goal_radius"], fc="#2ca02c", alpha=0.5, ec="g", label="goal"))
    ax.plot(*layout.start[:2], "ks", label="start")
    sources = [(env, C[env], "-", sorted(glob.glob(str(res / "runs" / f"{env}_dqn_seed*" / "final_eval_traj.npz"))))]
    if env == "fast":
        # fast-trained policies in the final Gazebo tier, same tuning layouts (transfer-gap recordings, §6)
        sources.append(("gz (transfer)", C["gz"], "--",
                        sorted(glob.glob(str(res / "transfer_diag" / "gz_gt" / "fast_dqn_seed*__on_gz_traj.npz")))))
    drawn = False
    for name, colour, ls, files in sources:
        for k, f in enumerate(files):
            with np.load(f) as d:
                key = f"ep{layout_idx}"
                if key in d:
                    xy = d[key]
                    ax.plot(xy[:, 0], xy[:, 1], color=colour, ls=ls, lw=1.6, alpha=0.85, label=f"{name}" if k == 0 else None)
                    drawn = True
    ax.set(xlim=(-half - 0.2, half + 0.2), ylim=(-half - 0.2, half + 0.2), aspect="equal", xlabel="x (m)", ylabel="y (m)",
           title=f"Trajectories on fixed layout {layout_idx} (one line per seed)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    if drawn:
        fig.tight_layout()
        fig.savefig(out / f"trajectory_overlay{suffix}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="fast", choices=["fast", "gz"])
    p.add_argument("--agent", default="dqn")
    p.add_argument("--layout", type=int, default=0, help="fixed layout index for the trajectory overlay")
    p.add_argument("--return-bin", type=int, default=5000, help="step bin width for the training-return curve")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config, tier=args.env)
    res = results_dir(cfg)
    suffix = "" if args.env == "fast" else f"_{args.env}"
    runs = load_runs(res, args.env, args.agent)
    base = random_baseline(res, args.env)

    fig_success(runs, args.env, "step", res, base, suffix)
    fig_success(runs, args.env, "episodes", res, base, suffix)
    fig_bars(runs, args.env, res, res, suffix, args.agent)
    fig_return(runs, args.env, res, args.return_bin, suffix)
    fig_traj(args.env, res, res, args.layout, cfg, suffix)
    warmup = int(cfg["run"]["warmup_steps"])
    fig_success_per_seed(runs, args.env, res, suffix, warmup)
    fig_q_stability(runs, args.env, res, suffix)
    stab = stability_table(runs, warmup)
    stab.to_csv(res / f"stability{suffix}.csv", index=False)

    keys = ["success_rate", "collision_rate", "timeout_rate", "mean_return"]
    rep = report_rows(res, args.env, args.agent, args.env, runs)
    if rep is not None:   # held-out report layouts: the headline numbers
        rows = [{"label": f"{args.env} (report, n={len(rep)})", **rep[keys].mean().to_dict(),
                 **{f"{k}_ci95": float(ci95(rep[[k]].to_numpy())[0]) for k in keys}}]
        rep_gz = report_rows(res, args.env, args.agent, "gz", runs) if args.env == "fast" else None
        if rep_gz is not None:
            rows.append({"label": f"{args.env}->gz (report, n={len(rep_gz)})", **rep_gz[keys].mean().to_dict(),
                         **{f"{k}_ci95": float(ci95(rep_gz[[k]].to_numpy())[0]) for k in keys}})
        summ = res / "report" / "summary.csv"
        df = pd.read_csv(summ)
        for name, sub in {"random policy": df[(df["run"] == "random_dqn") & (df["eval_env"] == args.env)]}.items():
            if len(sub):
                rows.append({"label": f"{name} (report)", **sub[keys].mean().to_dict()})
        pd.DataFrame(rows).to_csv(res / f"summary{suffix}.csv", index=False)
        print(f"seeds: {sorted(runs)}")
        print("\n" + format_summary_table(rows))
        print("\nstability (periodic evals on tuning layouts):\n" + stab.drop(columns=["dip_steps"]).to_string(index=False))
        print(f"\nplots written to {res}")
        return

    fin = final_rows(runs)
    rows = [{"label": f"{args.env} (mean of {len(fin)} seeds)", **fin[keys].mean().to_dict()}]
    summ = res / "transfer" / "summary.csv"
    if summ.exists():
        df = pd.read_csv(summ)
        extra = {"random policy": df[(df["run"] == "random_dqn") & (df["eval_env"] == args.env)],
                 "fast-trained -> gz": df[(df["eval_env"] == "gz") & df["run"].str.startswith("fast_dqn")]
                 if args.env == "fast" else df.iloc[0:0]}
        for name, sub in extra.items():
            if len(sub):
                rows.append({"label": f"{name} (n={len(sub)})",
                             **sub[["success_rate", "collision_rate", "timeout_rate", "mean_return"]].mean().to_dict()})
    pd.DataFrame(rows).to_csv(res / f"summary{suffix}.csv", index=False)
    print(f"seeds: {sorted(runs)}")
    print("\n" + format_summary_table(rows))
    print(f"\nplots written to {res}")


if __name__ == "__main__":
    main()
