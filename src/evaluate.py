"""Evaluation on the fixed layouts (epsilon = 0), plus a CLI for sim-to-sim transfer.

    python evaluate.py --checkpoint ../results/runs/fast_dqn_seed0/final.pt --env gz
    python evaluate.py --checkpoint <ckpt> [<ckpt> ...] --env fast
    python evaluate.py --random --env fast          # random-policy baseline
    python evaluate.py --checkpoint <ckpt> --env fast --layouts report        # held-out layouts (final numbers)
    python evaluate.py --checkpoint <ckpt> --env gz --record-steps --out-dir ../results/transfer_diag
"""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from agents import make_agent
from agents.base import Agent
from envs import make_env
from envs.common import TB3BaseEnv
from envs.layout import Layout, eval_layouts
from utils import apply_overrides, format_summary_table, load_config, results_dir, seed_everything


def run_eval(agent: Agent, env: TB3BaseEnv, layouts: list[Layout], n_episodes: int,
             record_traj: bool = False, step_log: list[dict[str, np.ndarray]] | None = None,
             ) -> tuple[dict[str, float], list[dict[str, Any]], list[np.ndarray]]:
    """Run `n_episodes` greedy episodes; episode i uses layout i (cycling). Returns
    (aggregate metrics, per-episode records, xy trajectories).

    If `step_log` is a list, one dict of per-step arrays is appended per episode (diagnostics only;
    does not change behaviour): obs (T+1, obs_dim), action (T,), q (T, n_actions) if the agent has
    `q_values`, pose (T+1, 3), reward (T,), min_range (T+1,), sim_dt (T,) (Gazebo: measured period)."""
    episodes: list[dict[str, Any]] = []
    trajs: list[np.ndarray] = []
    q_fn = getattr(agent, "q_values", None)
    gamma = getattr(agent, "gamma", None)     # for the Q-overestimation probe (agents with Q-values only)
    for i in range(n_episodes):
        obs, info = env.reset(options={"layout": layouts[i % len(layouts)]})
        xy = [info["pose"][:2].copy()]
        rec: dict[str, list] | None = None
        if step_log is not None:
            rec = {"obs": [obs.copy()], "action": [], "q": [], "pose": [info["pose"].copy()], "reward": [],
                   "min_range": [info["min_range"]], "sim_dt": []}
        ret, steps, dws, disc = 0.0, 0, [], 0.0
        q0 = float(np.max(q_fn(obs))) if q_fn is not None else float("nan")
        while True:
            action = agent.act(obs, explore=False)
            if rec is not None:
                rec["action"].append(action)
                if q_fn is not None:
                    rec["q"].append(q_fn(obs))
            obs, r, terminated, truncated, info = env.step(action)
            if gamma is not None:
                disc += gamma ** steps * r
            ret, steps = ret + r, steps + 1
            dws.append(info["dw"])
            if record_traj:
                xy.append(info["pose"][:2].copy())
            if rec is not None:
                sim_dts = getattr(env, "_sim_dts", None)
                for k, val in (("obs", obs.copy()), ("pose", info["pose"].copy()), ("reward", r),
                               ("min_range", info["min_range"]), ("sim_dt", sim_dts[-1] if sim_dts else env.dt)):
                    rec[k].append(val)
            if terminated or truncated:
                break
        episodes.append({"layout": i % len(layouts), "outcome": info["outcome"], "return": ret,
                         "steps": steps, "mean_abs_dw": float(np.mean(dws)), "q0": q0,
                         "disc_return": disc if gamma is not None else float("nan")})
        if record_traj:
            trajs.append(np.asarray(xy))
        if rec is not None:
            step_log.append({**{k: np.asarray(v) for k, v in rec.items()}, "outcome": np.array(info["outcome"])})
    df = pd.DataFrame(episodes)
    goal = df["outcome"] == "goal"
    metrics = {
        "success_rate": float(goal.mean()),
        "collision_rate": float((df["outcome"] == "collision").mean()),
        "timeout_rate": float((df["outcome"] == "timeout").mean()),
        "mean_return": float(df["return"].mean()),
        "mean_steps_to_goal": float(df.loc[goal, "steps"].mean()) if goal.any() else float("nan"),
        "mean_abs_dw": float(df["mean_abs_dw"].mean()),
        # Overestimation probe: max_a Q(s0, a) vs the realised discounted greedy return from s0 (reward_scale 1).
        "q0_mean": float(df["q0"].mean()),
        "disc_return_mean": float(df["disc_return"].mean()),
    }
    return metrics, episodes, trajs


def _run_name(ckpt: Path) -> str:
    return ckpt.parent.name   # e.g. fast_dqn_seed0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", nargs="+", default=[], help="one or more agent checkpoints (.pt)")
    p.add_argument("--random", action="store_true", help="evaluate a uniform-random policy instead")
    p.add_argument("--oracle", action="store_true", help="evaluate the heading-only oracle (ignores the LiDAR) instead")
    p.add_argument("--env", choices=["fast", "gz"], default="fast")
    p.add_argument("--agent", default="dqn")
    p.add_argument("--episodes", type=int, default=None, help="default: eval.episodes from config")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=None)
    p.add_argument("--out-dir", default=None, help="default: results/transfer")
    p.add_argument("--layouts", choices=["tuning", "report"], default="tuning",
                   help="fixed layout set (config eval.*_layout_seed); 'report' = held out, final numbers only")
    p.add_argument("--record-steps", action="store_true",
                   help="also save per-step obs/action/Q/pose/reward to <name>__on_<env>_steps.npz")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL",
                   help="config override for the eval env, e.g. --set run.act_lag_tau_w=0.05 (repeatable)")
    args = p.parse_args()
    if not args.checkpoint and not args.random and not args.oracle:
        p.error("give --checkpoint, --random or --oracle")

    cfg = load_config(args.config, tier=args.env)
    apply_overrides(cfg, args.overrides)
    if args.agent == "td3":
        cfg["actions"]["mode"] = "continuous"
    torch.set_num_threads(int(cfg["experiment"]["torch_threads"]))
    n_eps = args.episodes or cfg["eval"]["episodes" if args.layouts == "tuning" else "report_episodes"]
    layouts = eval_layouts(cfg, args.layouts)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir(cfg) / "transfer"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env, cfg)
    if args.env == "gz":
        env.check_connections(verbose=True)
    env.reset(seed=10_000 + args.seed)

    baseline = "random" if args.random else "oracle" if args.oracle else None
    targets: list[tuple[str, Path | None]] = [(f"{baseline}_{args.agent}", None)] if baseline else [
        (_run_name(Path(c)), Path(c)) for c in args.checkpoint]
    rows, table = [], []
    for name, ckpt in targets:
        seed_everything(args.seed)
        agent = make_agent(baseline or args.agent, cfg, env, args.seed)
        if ckpt is not None:
            agent.load(ckpt)
        step_log: list[dict[str, np.ndarray]] | None = [] if args.record_steps else None
        metrics, episodes, trajs = run_eval(agent, env, layouts, n_eps, record_traj=True, step_log=step_log)
        pd.DataFrame(episodes).to_csv(out_dir / f"{name}__on_{args.env}.csv", index=False)
        np.savez(out_dir / f"{name}__on_{args.env}_traj.npz", **{f"ep{i}": t for i, t in enumerate(trajs)})
        if step_log is not None:
            np.savez_compressed(out_dir / f"{name}__on_{args.env}_steps.npz",
                                **{f"ep{i}_{k}": v for i, rec in enumerate(step_log) for k, v in rec.items()})
        rows.append({"run": name, "eval_env": args.env, "layouts": args.layouts, "episodes": n_eps, **metrics})
        table.append({"label": f"{args.env}: {name}", **metrics})

    summary = out_dir / "summary.csv"
    new = pd.DataFrame(rows)
    lock = open(out_dir / ".summary.lock", "w")      # several evaluate.py processes may share out_dir
    fcntl.flock(lock, fcntl.LOCK_EX)
    if summary.exists():   # keep one row per (run, eval_env, layouts): latest wins
        old = pd.read_csv(summary)
        if "layouts" not in old:                  # files written before layout sets existed
            old["layouts"] = "tuning"
        key = ["run", "eval_env", "layouts"]
        keep = ~old.set_index(key).index.isin(new.set_index(key).index)
        new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(summary, index=False)
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    print("\n" + format_summary_table(table))
    print(f"\nper-episode CSVs and summary written to {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
