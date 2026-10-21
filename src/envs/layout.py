"""Layout (obstacles + start + goal) generation shared by the FAST and GAZEBO tiers.

A layout is plain data with no simulator dependency, so the exact same layout can be
instantiated in the 2D sim and in Gazebo (needed for a fair sim-to-sim transfer test).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Obstacle:
    slot: int          # index into the obstacle pool (Gazebo keeps one model per slot)
    kind: str          # "cylinder" | "box"
    x: float
    y: float
    yaw: float         # only meaningful for boxes


@dataclass(frozen=True)
class Layout:
    obstacles: tuple[Obstacle, ...]
    start: tuple[float, float, float]   # x, y, yaw
    goal: tuple[float, float]


def slot_kind(slot: int) -> str:
    """Even slots are cylinders, odd slots are boxes (mirrored by the generated Gazebo world)."""
    return "cylinder" if slot % 2 == 0 else "box"


def bounding_radius(kind: str, cfg: dict[str, Any]) -> float:
    if kind == "cylinder":
        return float(cfg["obstacles"]["cylinder_radius"])
    return float(cfg["obstacles"]["box_size"]) * np.sqrt(2.0) / 2.0


def surface_dist(ob: Obstacle, px: float, py: float, cfg: dict[str, Any]) -> float:
    """Signed distance from a point to the obstacle surface (negative inside)."""
    if ob.kind == "cylinder":
        return float(np.hypot(px - ob.x, py - ob.y) - cfg["obstacles"]["cylinder_radius"])
    h = cfg["obstacles"]["box_size"] / 2.0
    c, s = np.cos(ob.yaw), np.sin(ob.yaw)
    dx, dy = px - ob.x, py - ob.y
    qx, qy = abs(c * dx + s * dy) - h, abs(-s * dx + c * dy) - h
    outside = np.hypot(max(qx, 0.0), max(qy, 0.0))
    return float(outside + min(max(qx, qy), 0.0))


class LayoutSampler:
    """Rejection-samples collision-free layouts according to the `arena`/`obstacles` config."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.half = cfg["arena"]["size"] / 2.0

    def _free_of_obstacles(self, obstacles: list[Obstacle], x: float, y: float, clearance: float) -> bool:
        return all(surface_dist(o, x, y, self.cfg) >= clearance for o in obstacles)

    def _sample_obstacles(self, rng: np.random.Generator) -> list[Obstacle]:
        oc, ac = self.cfg["obstacles"], self.cfg["arena"]
        n = int(rng.integers(oc["min_count"], oc["max_count"] + 1))
        slots = rng.choice(oc["max_count"], size=n, replace=False)
        lim = self.half - ac["obstacle_wall_margin"]
        placed: list[Obstacle] = []
        for slot in map(int, slots):
            kind = slot_kind(slot)
            rad = bounding_radius(kind, self.cfg)
            for _ in range(200):
                x, y = rng.uniform(-lim, lim, size=2)
                if all(np.hypot(x - o.x, y - o.y) >=
                       rad + bounding_radius(o.kind, self.cfg) + ac["obstacle_gap"] for o in placed):
                    yaw = float(rng.uniform(-np.pi, np.pi)) if kind == "box" else 0.0
                    placed.append(Obstacle(slot, kind, float(x), float(y), yaw))
                    break
        return placed

    def sample(self, rng: np.random.Generator) -> Layout:
        ac = self.cfg["arena"]
        while True:
            obstacles = self._sample_obstacles(rng)
            sl = self.half - ac["start_wall_margin"]
            gl = self.half - ac["goal_wall_margin"]
            start = goal = None
            for _ in range(200):
                sx, sy = rng.uniform(-sl, sl, size=2)
                if self._free_of_obstacles(obstacles, sx, sy, ac["start_obstacle_clearance"]):
                    start = (float(sx), float(sy), float(rng.uniform(-np.pi, np.pi)))
                    break
            if start is None:
                continue
            for _ in range(200):
                gx, gy = rng.uniform(-gl, gl, size=2)
                d = np.hypot(gx - start[0], gy - start[1])
                if (ac["start_goal_min_dist"] <= d <= ac["goal_max_dist"]
                        and self._free_of_obstacles(obstacles, gx, gy, ac["goal_obstacle_clearance"])):
                    goal = (float(gx), float(gy))
                    break
            if goal is not None:
                return Layout(tuple(obstacles), start, goal)


def eval_layouts(cfg: dict[str, Any], which: str = "tuning") -> list[Layout]:
    """A fixed set of evaluation layouts, deterministic given its seed.

    which="tuning": `eval.tuning_layout_seed` x `eval.n_layouts` -- the periodic evaluations during training
                    and every selection decision (hyper-parameters, env-fidelity options) use these.
    which="report": `eval.report_layout_seed` x `eval.report_n_layouts` -- held out; final numbers only.
    """
    ec = cfg["eval"]
    if which not in ("tuning", "report"):
        raise ValueError(f"unknown layout set {which!r}")
    n = ec["n_layouts"] if which == "tuning" else ec["report_n_layouts"]
    rng = np.random.default_rng(ec[f"{which}_layout_seed"])
    sampler = LayoutSampler(cfg)
    return [sampler.sample(rng) for _ in range(n)]
