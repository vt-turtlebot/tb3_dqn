"""Tier-independent environment logic.

`TB3BaseEnv` owns everything that must be *identical* in the FAST and GAZEBO tiers:
action decoding, observation construction, reward, termination and the heading-angle
wrapping. Subclasses only implement how the world is reset and advanced by one control
period, returning the robot pose and the 24 LiDAR bin ranges (metres).

Observation (float32, 28-dim):
    [0:24]  LiDAR bins, min(range, lidar_clip) / lidar_clip in [0, 1]   (bin 0 = straight ahead, CCW)
    [24]    distance to goal / observation.dist_norm
    [25]    heading to goal, wrapped to [-pi, pi] (radians; positive = goal is to the left)
    [26]    last commanded v / v_max
    [27]    last commanded w / w_max
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.layout import Layout, LayoutSampler


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi]. (Forgetting this makes the robot spin in place forever.)"""
    return float(np.arctan2(np.sin(a), np.cos(a)))


class ScanBinner:
    """Reduces a raw 360-sample scan to `n_bins` bins by taking the min range in each sector.

    Used by both tiers so the observation pipeline is identical. Bin k is centred on
    k * 360/n_bins degrees (CCW from the robot's x axis), i.e. bin 0 looks straight ahead.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.lc = cfg["lidar"]
        self.n_bins = int(self.lc["n_bins"])
        self._cache_key: tuple | None = None
        self._idx = np.zeros(0, dtype=np.int64)

    def __call__(self, ranges: np.ndarray, angle_min: float, angle_inc: float) -> np.ndarray:
        lc = self.lc
        r = np.nan_to_num(np.asarray(ranges, dtype=np.float64), nan=lc["range_max"],
                          posinf=lc["range_max"], neginf=lc["range_min"])
        r = np.clip(r, lc["range_min"], lc["range_max"])
        key = (len(r), angle_min, angle_inc)
        if key != self._cache_key:
            ang = angle_min + np.arange(len(r)) * angle_inc
            self._idx = np.round(ang / (2 * np.pi / self.n_bins)).astype(np.int64) % self.n_bins
            self._cache_key = key
        bins = np.full(self.n_bins, lc["range_max"])
        np.minimum.at(bins, self._idx, r)
        return bins


class TB3BaseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.rc, self.tc, self.rw = cfg["robot"], cfg["task"], cfg["reward"]
        self.n_bins = int(cfg["lidar"]["n_bins"])
        self.range_max = float(cfg["lidar"]["range_max"])
        self.dt = 1.0 / float(cfg["control"]["rate_hz"])
        self.dist_norm = float(cfg["observation"]["dist_norm"])
        self.lidar_clip = float(cfg["observation"]["lidar_clip"])   # ranges beyond this all read 1.0
        self.sampler = LayoutSampler(cfg)

        ac = cfg["actions"]
        self.action_mode = ac["mode"]
        if self.action_mode == "discrete":
            self._w_table = np.asarray(ac["discrete"]["w"], dtype=np.float64)
            self._v_const = float(ac["discrete"]["v"])
            assert 0.0 <= self._v_const <= self.rc["v_max"], "discrete v exceeds Burger limits"
            assert np.all(np.abs(self._w_table) <= self.rc["w_max"]), "discrete w exceeds Burger limits"
            self.action_space = spaces.Discrete(len(self._w_table))
        else:  # continuous: 1-D angular velocity, same constant v and w range as the discrete set
            self._v_const = float(ac["continuous"]["v"])
            w_lim = float(ac["continuous"]["w_max"])
            assert 0.0 <= self._v_const <= self.rc["v_max"] and w_lim <= self.rc["w_max"], "continuous action exceeds Burger limits"
            self.action_space = spaces.Box(low=-w_lim, high=w_lim, shape=(1,), dtype=np.float32)

        obs_dim = self.n_bins + 4
        low = np.concatenate([np.zeros(self.n_bins), [0.0, -np.pi, 0.0, -1.0]]).astype(np.float32)
        high = np.concatenate([np.ones(self.n_bins), [np.inf, np.pi, 1.0, 1.0]]).astype(np.float32)
        self.observation_space = spaces.Box(low, high, shape=(obs_dim,), dtype=np.float32)

        self._layout: Layout | None = None
        self._pose = np.zeros(3)
        self._goal = np.zeros(2)
        self._last_v = 0.0
        self._last_w = 0.0
        self._prev_dist = 0.0
        self._steps = 0

    # ---- hooks implemented by each tier -------------------------------------------------
    @abstractmethod
    def _reset_world(self, layout: Layout) -> tuple[np.ndarray, np.ndarray]:
        """Place robot/obstacles/goal. Returns (pose[x,y,yaw] in the world frame, bin ranges in m)."""

    @abstractmethod
    def _advance(self, v: float, w: float) -> tuple[np.ndarray, np.ndarray]:
        """Apply (v, w) for one control period. Returns (pose, bin ranges in m)."""

    # ---- shared logic -----------------------------------------------------------------
    def _decode_action(self, action: Any) -> tuple[float, float]:
        if self.action_mode == "discrete":
            return self._v_const, float(self._w_table[int(action)])
        w_lim = float(self.action_space.high[0])
        return self._v_const, float(np.clip(np.asarray(action, dtype=np.float64).reshape(1)[0], -w_lim, w_lim))

    def _goal_geometry(self, pose: np.ndarray) -> tuple[float, float]:
        dx, dy = self._goal[0] - pose[0], self._goal[1] - pose[1]
        dist = float(np.hypot(dx, dy))
        heading = wrap_angle(np.arctan2(dy, dx) - pose[2])
        return dist, heading

    def _obs(self, pose: np.ndarray, bins: np.ndarray) -> np.ndarray:
        dist, heading = self._goal_geometry(pose)
        return np.concatenate([
            np.minimum(bins, self.lidar_clip) / self.lidar_clip,
            [dist / self.dist_norm, heading, self._last_v / self.rc["v_max"], self._last_w / self.rc["w_max"]],
        ]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        layout = (options or {}).get("layout")
        if layout is None:
            layout = self.sampler.sample(self.np_random)
        self._layout = layout
        self._goal = np.asarray(layout.goal, dtype=np.float64)
        self._last_v = self._last_w = 0.0
        self._steps = 0
        pose, bins = self._reset_world(layout)
        self._pose = np.asarray(pose, dtype=np.float64)
        self._prev_dist, _ = self._goal_geometry(self._pose)
        info = {"pose": self._pose.copy(), "goal": self._goal.copy(), "min_range": float(bins.min())}
        return self._obs(self._pose, bins), info

    def step(self, action: Any):
        v, w = self._decode_action(action)
        pose, bins = self._advance(v, w)
        self._pose = np.asarray(pose, dtype=np.float64)
        self._steps += 1

        dist, _ = self._goal_geometry(self._pose)
        min_range = float(bins.min())
        reward = self.rw["progress_weight"] * (self._prev_dist - dist) - self.rw["step_penalty"]
        self._prev_dist = dist

        outcome: str | None = None
        if min_range < self.tc["collision_dist"]:            # collision has priority over goal
            outcome, reward = "collision", reward + self.rw["collision"]
        elif dist < self.tc["goal_radius"]:
            outcome, reward = "goal", reward + self.rw["goal"]
        terminated = outcome is not None
        truncated = (not terminated) and self._steps >= self.tc["max_steps"]
        if truncated:
            outcome, reward = "timeout", reward + self.rw["timeout_penalty"]

        info = {"outcome": outcome, "success": outcome == "goal", "dist": dist, "min_range": min_range,
                "dw": abs(w - self._last_w), "v": v, "w": w, "pose": self._pose.copy()}
        self._last_v, self._last_w = v, w
        return self._obs(self._pose, bins), float(reward), terminated, truncated, info
