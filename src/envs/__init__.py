"""Environment factory. The Gazebo tier is imported lazily so the fast tier needs no ROS."""
from __future__ import annotations

from typing import Any

from envs.common import TB3BaseEnv


def make_env(tier: str, cfg: dict[str, Any]) -> TB3BaseEnv:
    if tier == "fast":
        from envs.env_fast import FastEnv
        return FastEnv(cfg)
    if tier == "gz":
        from envs.env_gz import GazeboEnv
        return GazeboEnv(cfg)
    raise ValueError(f"unknown env tier {tier!r} (expected 'fast' or 'gz')")
