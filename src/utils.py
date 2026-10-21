"""Config loading, seeding and small helpers shared by train/evaluate/plot."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SRC_DIR = Path(__file__).resolve().parent


def load_config(path: str | Path | None = None, tier: str | None = None, smoke: bool = False) -> dict[str, Any]:
    """Load config.yaml and resolve tier-specific / smoke overrides into flat keys.

    Adds `cfg["tier"]` and `cfg["run"]` (total_steps, eps_decay_steps, warmup_steps, ...).
    """
    path = Path(path) if path else SRC_DIR / "config.yaml"
    with open(path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["tier"] = tier
    run: dict[str, Any] = {}
    if tier is not None:
        run.update(cfg["tiers"][tier])
    if tier != "fast":
        run.setdefault("sim_substeps", cfg["tiers"]["fast"]["sim_substeps"])
        run.setdefault("action_noise_std", 0.0)
        for k in ("act_delay_s", "act_delay_jitter_s", "act_lag_tau_s"):   # a fast env built from a gz config
            run.setdefault(k, 0.0)
    if smoke:
        sm = cfg["smoke"]
        run.update(total_steps=sm["total_steps"], warmup_steps=sm["warmup_steps"],
                   eps_decay_steps=sm["eps_decay_steps"])
        cfg["eval"]["interval"] = sm["eval_interval"]
        cfg["eval"]["episodes"] = sm["eval_episodes"]
        cfg["dqn"]["target_update_every"] = sm["target_update_every"]
    cfg["run"] = run
    cfg["smoke_mode"] = smoke
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> None:
    """Apply `section.key=value` overrides in place (value parsed as YAML). Used for tuning runs;
    tier values live under `run.` once load_config has merged them (e.g. run.eps_decay_steps=50000)."""
    for item in overrides:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key {key!r}")
        node[parts[-1]] = yaml.safe_load(raw)


def results_dir(cfg: dict[str, Any]) -> Path:
    p = (SRC_DIR / cfg["experiment"]["results_dir"]).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch. (Env seeding happens through env.reset(seed=...).)"""
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_summary_table(rows: list[dict[str, Any]]) -> str:
    """Summary table printed at the end of a run: tier, success %, collision %, timeout %, mean return."""
    head = f"{'env tier':<22}{'success %':>10}{'collision %':>13}{'timeout %':>11}{'mean return':>13}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['label']:<22}{100 * r['success_rate']:>10.1f}{100 * r['collision_rate']:>13.1f}"
                     f"{100 * r['timeout_rate']:>11.1f}{r['mean_return']:>13.1f}")
    return "\n".join(lines)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def env_info(cfg: dict[str, Any], args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Provenance record written next to every run (the project is not a git repo): package versions, a hash of the
    resolved config, and a hash of every source file, so a result can be tied to the exact code that produced it."""
    import gymnasium
    import pandas
    import torch
    src = sorted(p for p in SRC_DIR.rglob("*") if p.suffix in (".py", ".yaml", ".sh") and "__pycache__" not in p.parts)
    files = {str(p.relative_to(SRC_DIR)): _sha256(p.read_bytes()) for p in src}
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv, "args": args or {},
        "python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "numpy": np.__version__,
        "gymnasium": gymnasium.__version__, "pandas": pandas.__version__, "pyyaml": yaml.__version__,
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "config_sha256": _sha256(json.dumps(cfg, sort_keys=True, default=str).encode()),
        "source_sha256": _sha256(json.dumps(files, sort_keys=True).encode()),
        "source_files": files,
    }
