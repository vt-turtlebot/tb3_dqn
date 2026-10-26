"""Train an agent on the FAST or GAZEBO tier.

    python train.py --agent dqn --env fast --seed 0
    python train.py --agent dqn --env gz   --seed 0     # needs ./start_gz.sh running
    python train.py --agent dqn --env fast --smoke      # ~2k steps, quick end-to-end check
    python train.py --agent td3 --env fast --seed 0     # continuous actions (actions.mode is set from --agent)

Outputs (results/runs/<env>_<agent>_seed<seed>/):
    train_episodes.csv   one row per finished training episode
    eval.csv             one row per periodic evaluation (this is what plot.py reads)
    ckpt_latest.pt(+.replay.npz) + state.json   resume point (written after every eval)
    final.pt, final_eval_episodes.csv, final_eval_traj.npz
    env_info.json        package versions, config hash and source hash (the project is not under git)
Re-running the same command resumes from the last checkpoint (use --fresh to start over). On the fast tier a resumed
run is bit-identical to an uninterrupted one (the env RNG states are checkpointed too; README "Reproducibility").
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from agents import make_agent, set_action_mode
from envs import make_env
from envs.layout import eval_layouts
from evaluate import run_eval
from utils import apply_overrides, env_info, format_summary_table, load_config, results_dir, seed_everything


def _append_row(path: Path, row: dict[str, Any]) -> None:
    new = pd.DataFrame([row])
    if path.exists():
        with open(path) as f:
            header = f.readline().rstrip("\n").split(",")
        if header != list(new.columns):          # column set changed (e.g. update metrics appear after warm-up)
            pd.concat([pd.read_csv(path), new], ignore_index=True).to_csv(path, index=False)
            return
    new.to_csv(path, mode="a", header=not path.exists(), index=False)


def _truncate_csv(path: Path, max_step: int) -> None:
    """On resume, drop rows logged after the checkpoint so the CSV matches the restored agent."""
    if path.exists():
        df = pd.read_csv(path)
        df[df["step"] <= max_step].to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default="dqn", choices=["dqn", "td3"])
    p.add_argument("--env", default="fast", choices=["fast", "gz"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="short run (~2k steps); gz also checks topics")
    p.add_argument("--fresh", action="store_true", help="ignore/delete any existing checkpoint and restart")
    p.add_argument("--total-steps", type=int, default=None, help="override tier default")
    p.add_argument("--config", default=None)
    p.add_argument("--run-dir", default=None, help="override output directory")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL",
                   help="config override, e.g. --set dqn.lr=1e-4 --set run.eps_decay_steps=50000 (repeatable)")
    p.add_argument("--tag", default=None, help="tuning run name; output goes to results/tuning/<tag>_seed<N>")
    args = p.parse_args()

    cfg = load_config(args.config, tier=args.env, smoke=args.smoke)
    set_action_mode(cfg, args.agent)
    apply_overrides(cfg, args.overrides)
    if args.total_steps:
        cfg["run"]["total_steps"] = args.total_steps
    torch.set_num_threads(int(cfg["experiment"]["torch_threads"]))
    total = int(cfg["run"]["total_steps"])
    eval_every, eval_eps = int(cfg["eval"]["interval"]), int(cfg["eval"]["episodes"])

    base = results_dir(cfg) / ("smoke" if args.smoke else "tuning" if args.tag else "runs")
    name = f"{args.tag}_seed{args.seed}" if args.tag else f"{args.env}_{args.agent}_seed{args.seed}"
    run_dir = Path(args.run_dir) if args.run_dir else base / name
    if args.fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_csv, eval_csv = run_dir / "train_episodes.csv", run_dir / "eval.csv"
    ckpt, state_file = run_dir / "ckpt_latest.pt", run_dir / "state.json"

    seed_everything(args.seed)
    env = make_env(args.env, cfg)
    # The 2D sim gets its own eval env (independent RNG); Gazebo has one world, so it is shared.
    eval_env = make_env("fast", cfg) if args.env == "fast" else env
    if args.env == "gz":
        env.check_connections(verbose=args.smoke)
    agent = make_agent(args.agent, cfg, env, args.seed)
    layouts = eval_layouts(cfg)              # tuning set: periodic evals never touch the report layouts

    episodes_done = 0
    resumed = ckpt.exists() and state_file.exists()
    if resumed:
        agent.load(ckpt, load_replay=True)
        episodes_done = int(json.loads(state_file.read_text())["episodes"])
        _truncate_csv(train_csv, agent.steps)
        _truncate_csv(eval_csv, agent.steps)
        print(f"[resume] {run_dir.name}: continuing from step {agent.steps} ({episodes_done} episodes)")
    else:
        for f in (train_csv, eval_csv):
            f.unlink(missing_ok=True)
    with open(run_dir / "config_snapshot.json", "w") as f:
        json.dump({k: v for k, v in cfg.items()}, f, indent=2)
    (run_dir / "env_info.json").write_text(json.dumps(env_info(cfg, vars(args)), indent=2))

    if eval_env is not env:
        eval_env.reset(seed=10_000 + args.seed)
    # Exact resume: restore the env RNG streams saved with the checkpoint (absent in checkpoints written before this
    # existed -> old behaviour: re-seed the training env from (seed, step), deterministic but not bit-identical).
    saved = json.loads(state_file.read_text()) if resumed else {}
    rng_restored = resumed and agent.steps > 0 and "train_env_rng" in saved
    if rng_restored:
        env.np_random.bit_generator.state = saved["train_env_rng"]
    if resumed and "eval_env_rng" in saved and eval_env is not env:
        eval_env.np_random.bit_generator.state = saved["eval_env_rng"]
    env_seed = args.seed * 100_003 + agent.steps
    print(f"[train] tier={args.env} agent={args.agent} seed={args.seed} steps={total} "
          f"eval every {eval_every} x {eval_eps} eps  ->  {run_dir}")

    recent_returns: list[float] = []     # training-episode returns since the previous eval
    recent_updates: dict[str, list[float]] = {}   # agent.update() metrics since the previous eval (logging only)
    t_start, t_last = time.time(), time.time()
    last_metrics: dict[str, float] | None = None

    def do_eval(final: bool = False) -> dict[str, float]:
        metrics, eps, trajs = run_eval(agent, eval_env, layouts, eval_eps, record_traj=final)
        row = {"step": agent.steps, "episodes": episodes_done, **metrics,
               "train_return_mean": float(np.mean(recent_returns)) if recent_returns else float("nan"),
               "epsilon": agent.info().get("epsilon", float("nan")),
               **{f"upd_{k}_{fn.__name__}": float(fn(v)) for k, v in recent_updates.items() for fn in (np.mean, np.max)},
               "wall_time_s": round(time.time() - t_start, 1)}
        _append_row(eval_csv, row)
        print(f"[eval @ {agent.steps:>7}] success {100 * metrics['success_rate']:5.1f}%  "
              f"collision {100 * metrics['collision_rate']:5.1f}%  timeout {100 * metrics['timeout_rate']:5.1f}%  "
              f"return {metrics['mean_return']:8.1f}  |dw| {metrics['mean_abs_dw']:.2f}  "
              f"eps {row['epsilon']:.2f}", flush=True)
        recent_returns.clear()
        recent_updates.clear()
        if final:
            pd.DataFrame(eps).to_csv(run_dir / "final_eval_episodes.csv", index=False)
            np.savez(run_dir / "final_eval_traj.npz", **{f"ep{i}": t for i, t in enumerate(trajs)})
        return metrics

    def checkpoint() -> None:
        agent.save(ckpt, include_replay=cfg["experiment"]["checkpoint_every_eval"])
        state = {"steps": agent.steps, "episodes": episodes_done,
                 "train_env_rng": env.np_random.bit_generator.state}
        if eval_env is not env:
            state["eval_env_rng"] = eval_env.np_random.bit_generator.state
        tmp = state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(state_file)

    if not resumed:
        do_eval()                       # step-0 point so curves start from the untrained policy
        checkpoint()

    # Fresh runs (and step-0 resumes) seed the stream; a resumed run continues it exactly where the checkpoint was.
    obs, _ = env.reset() if rng_restored else env.reset(seed=env_seed)
    ep_ret, ep_len, last_eval_step = 0.0, 0, agent.steps
    while agent.steps < total:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.observe(obs, action, reward, next_obs, terminated, truncated)
        for k, v in agent.update().items():
            recent_updates.setdefault(k, []).append(v)
        obs, ep_ret, ep_len = next_obs, ep_ret + reward, ep_len + 1

        if terminated or truncated:
            episodes_done += 1
            recent_returns.append(ep_ret)
            _append_row(train_csv, {"step": agent.steps, "episode": episodes_done, "return": ep_ret,
                                    "length": ep_len, "outcome": info["outcome"],
                                    "epsilon": agent.info().get("epsilon", float("nan"))})
            ep_ret, ep_len = 0.0, 0
            obs, _ = env.reset()

        if agent.steps % eval_every == 0 and agent.steps != last_eval_step:
            now = time.time()
            print(f"[train @ {agent.steps:>7}] {eval_every / (now - t_last):6.0f} steps/s  "
                  f"episodes {episodes_done}", flush=True)
            last_metrics = do_eval(final=agent.steps >= total)
            last_eval_step = agent.steps
            checkpoint()
            t_last = time.time()
            obs, _ = env.reset()        # evaluation moved the world; start a fresh training episode
            ep_ret, ep_len = 0.0, 0

    if last_metrics is None or last_eval_step != agent.steps:   # total not a multiple of the interval
        last_metrics = do_eval(final=True)
        checkpoint()
    agent.save(run_dir / "final.pt")
    print(f"\n[done] {agent.steps} steps in {(time.time() - t_start) / 60:.1f} min -> {run_dir / 'final.pt'}")
    print(format_summary_table([{"label": f"{args.env} (seed {args.seed})", **last_metrics}]))
    env.close()


if __name__ == "__main__":
    main()
