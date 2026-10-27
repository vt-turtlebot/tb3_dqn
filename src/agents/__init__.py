"""Agent factory. Phase 2 adds `td3` here without touching train/evaluate/plot."""
from __future__ import annotations

from typing import Any

import gymnasium as gym

from agents.base import Agent, HeadingOracleAgent, RandomAgent


def make_agent(name: str, cfg: dict[str, Any], env: gym.Env, seed: int) -> Agent:
    if name == "dqn":
        from agents.dqn import DQNAgent
        assert isinstance(env.action_space, gym.spaces.Discrete), "DQN needs the discrete action mode"
        return DQNAgent(cfg, int(env.observation_space.shape[0]), int(env.action_space.n), seed)
    if name == "td3":
        from agents.td3 import TD3Agent
        assert isinstance(env.action_space, gym.spaces.Box), "TD3 needs the continuous action mode"
        return TD3Agent(cfg, int(env.observation_space.shape[0]), env.action_space.high, seed)
    if name == "random":
        return RandomAgent(env.action_space, seed)
    if name == "oracle":
        return HeadingOracleAgent(env.action_space, seed)
    raise ValueError(f"unknown agent {name!r}")
