"""Agent factory. Adding an agent touches only this file (plus its own module and config section)."""
from __future__ import annotations

from typing import Any

import gymnasium as gym

from agents.base import Agent, HeadingOracleAgent, RandomAgent

# Env action mode each agent needs (config `actions.mode`). train.py / evaluate.py set it from --agent, so the same
# config.yaml serves both agents. The random baseline uses whichever mode its --agent counterpart uses.
ACTION_MODE = {"dqn": "discrete", "td3": "continuous", "oracle": "discrete"}


def set_action_mode(cfg: dict[str, Any], agent: str) -> None:
    cfg["actions"]["mode"] = ACTION_MODE[agent]


def make_agent(name: str, cfg: dict[str, Any], env: gym.Env, seed: int) -> Agent:
    if name == "dqn":
        from agents.dqn import DQNAgent
        assert isinstance(env.action_space, gym.spaces.Discrete), "DQN needs the discrete action mode"
        return DQNAgent(cfg, int(env.observation_space.shape[0]), int(env.action_space.n), seed)
    if name == "td3":
        from agents.td3 import TD3Agent
        assert isinstance(env.action_space, gym.spaces.Box), "TD3 needs the continuous action mode"
        return TD3Agent(cfg, int(env.observation_space.shape[0]), env.action_space.low, env.action_space.high, seed)
    if name == "random":
        return RandomAgent(env.action_space, seed)
    if name == "oracle":
        assert isinstance(env.action_space, gym.spaces.Discrete), "the heading oracle uses the discrete actions"
        return HeadingOracleAgent(env.action_space, seed)
    raise ValueError(f"unknown agent {name!r}")
