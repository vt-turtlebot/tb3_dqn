"""Public entry point for the pure-Python TurtleBot3 Burger sim (no ROS / Gazebo needed).

    from tb3sim import make_fast_env
    env = make_fast_env()                      # Gymnasium env, 5 discrete actions
    obs, info = env.reset(seed=0)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from envs.common import TB3BaseEnv
from envs.env_fast import FastEnv
from utils import apply_overrides, load_config

__all__ = ["make_fast_env", "FastEnv", "TB3BaseEnv"]


def make_fast_env(config_path: str | Path | None = None, overrides: list[str] | None = None) -> FastEnv:
    """Build a FastEnv from config.yaml (default: the repo's src/config.yaml).

    overrides: `section.key=value` strings, e.g. ["env.max_steps=300"].
    """
    cfg: dict[str, Any] = load_config(config_path, tier="fast")
    if overrides:
        apply_overrides(cfg, overrides)
    return FastEnv(cfg)
