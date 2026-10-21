"""Common agent interface. Any agent (DQN now, TD3 in phase 2) implements exactly this,
so train.py / evaluate.py / plot.py never change when a new agent is added."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class Agent(ABC):
    steps: int = 0   # env steps observed so far (drives warmup / schedules / target updates)

    @abstractmethod
    def act(self, obs: np.ndarray, explore: bool = True) -> Any:
        """Return an action for `obs`. explore=False is the greedy/deterministic policy (eval)."""

    @abstractmethod
    def observe(self, obs: np.ndarray, action: Any, reward: float, next_obs: np.ndarray,
                terminated: bool, truncated: bool) -> None:
        """Record one transition and advance the step counter."""

    @abstractmethod
    def update(self) -> dict[str, float]:
        """Do (at most) one learning step. Returns metrics, or {} if no update happened."""

    @abstractmethod
    def save(self, path: str | Path, include_replay: bool = False) -> None: ...

    @abstractmethod
    def load(self, path: str | Path, load_replay: bool = False) -> None: ...

    def info(self) -> dict[str, float]:
        """Extra values worth logging (e.g. epsilon). Continuous agents may return {}."""
        return {}


class RandomAgent(Agent):
    """Uniform-random policy; used for the random baseline."""

    def __init__(self, action_space: Any, seed: int = 0) -> None:
        self.action_space = action_space
        self.action_space.seed(seed)

    def act(self, obs: np.ndarray, explore: bool = True) -> Any:
        return self.action_space.sample()

    def observe(self, *args: Any, **kwargs: Any) -> None:
        self.steps += 1

    def update(self) -> dict[str, float]:
        return {}

    def save(self, path: str | Path, include_replay: bool = False) -> None:
        pass

    def load(self, path: str | Path, load_replay: bool = False) -> None:
        pass


class HeadingOracleAgent(RandomAgent):
    """Baseline that ignores the LiDAR: turn toward the goal, drive forward when roughly facing it (discrete actions
    0 = hard left .. 4 = hard right). Same controller as validate_gz.py's oracle."""

    def act(self, obs: np.ndarray, explore: bool = True) -> Any:
        h = float(obs[-3])                       # heading to goal (obs[25] with 24 LiDAR bins)
        if abs(h) < 0.2:
            return 2
        if abs(h) < 0.8:
            return 1 if h > 0 else 3
        return 0 if h > 0 else 4
