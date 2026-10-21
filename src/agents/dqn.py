"""Vanilla DQN (Mnih et al. 2015 style) -- deliberately NO Double/Dueling/PER/n-step.

TD target:  y = r + gamma * max_a' Q_target(s', a') * (1 - done)
The target network both selects and evaluates the max action (that is what makes it
"vanilla" rather than Double DQN).

`done` is true only for real terminations (goal / collision). A time-limit truncation is
not a terminal state (the observation has no clock), so it still bootstraps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import Agent


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Uniform replay stored in preallocated float32 arrays (fast, and trivially checkpointable)."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def add(self, obs: np.ndarray, act: int, rew: float, next_obs: np.ndarray, done: bool) -> None:
        i = self.pos
        self.obs[i], self.next_obs[i], self.act[i], self.rew[i], self.done[i] = obs, next_obs, act, rew, float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
        idx = rng.integers(0, self.size, size=batch)
        return (torch.from_numpy(self.obs[idx]), torch.from_numpy(self.act[idx]),
                torch.from_numpy(self.rew[idx]), torch.from_numpy(self.next_obs[idx]),
                torch.from_numpy(self.done[idx]))

    def state(self) -> dict[str, Any]:
        n = self.size
        return {"obs": self.obs[:n], "next_obs": self.next_obs[:n], "act": self.act[:n], "rew": self.rew[:n],
                "done": self.done[:n], "pos": np.array(self.pos)}

    def restore(self, d: Any) -> None:
        n = len(d["act"])
        self.obs[:n], self.next_obs[:n], self.act[:n] = d["obs"], d["next_obs"], d["act"]
        self.rew[:n], self.done[:n] = d["rew"], d["done"]
        self.size, self.pos = n, int(d["pos"])


class DQNAgent(Agent):
    def __init__(self, cfg: dict[str, Any], obs_dim: int, n_actions: int, seed: int = 0) -> None:
        dc, rc = cfg["dqn"], cfg["run"]
        self.n_actions = n_actions
        self.gamma = float(dc["gamma"])
        self.batch_size = int(dc["batch_size"])
        self.train_freq = int(dc["train_freq"])
        self.target_update_every = int(dc["target_update_every"])
        self.grad_clip = float(dc["grad_clip"])
        self.loss_name = dc["loss"]
        self.reward_scale = float(dc["reward_scale"])   # keeps TD targets O(1); env rewards stay unscaled
        self.eps_start, self.eps_end = float(dc["eps_start"]), float(dc["eps_end"])
        self.eps_decay_steps = int(rc["eps_decay_steps"])
        self.warmup_steps = int(rc["warmup_steps"])

        self.q = QNetwork(obs_dim, n_actions, list(dc["hidden"]))
        self.q_target = QNetwork(obs_dim, n_actions, list(dc["hidden"]))
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=float(dc["lr"]))
        self.buffer = ReplayBuffer(int(dc["buffer_size"]), obs_dim)
        self.rng = np.random.default_rng(seed)   # exploration + replay sampling
        self.steps = 0

    # ---- policy -------------------------------------------------------------------------
    @property
    def epsilon(self) -> float:
        frac = min(self.steps / max(self.eps_decay_steps, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        if explore and (self.steps < self.warmup_steps or self.rng.random() < self.epsilon):
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            return int(self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax(dim=1).item())

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Q(obs, .) for diagnostics (not part of the common Agent interface)."""
        with torch.no_grad():
            return self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

    # ---- learning -----------------------------------------------------------------------
    def observe(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray,
                terminated: bool, truncated: bool) -> None:
        self.buffer.add(obs, int(action), reward, next_obs, terminated)   # truncation is NOT terminal
        self.steps += 1

    def update(self) -> dict[str, float]:
        if (self.steps < self.warmup_steps or self.buffer.size < self.batch_size
                or self.steps % self.train_freq != 0):
            return {}
        obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size, self.rng)
        q_all = self.q(obs)
        q_sa = q_all.gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Vanilla DQN: the target net both selects and evaluates the max action.
            y = rew * self.reward_scale + self.gamma * self.q_target(next_obs).max(dim=1).values * (1.0 - done)
        loss = F.smooth_l1_loss(q_sa, y) if self.loss_name == "huber" else F.mse_loss(q_sa, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)   # norm *before* clipping
        self.opt.step()
        if self.steps % self.target_update_every == 0:                    # hard target update
            self.q_target.load_state_dict(self.q.state_dict())
        # Logging only (stability diagnostics); none of these feed back into learning.
        return {"loss": float(loss.item()), "q_mean": float(q_sa.mean().item()),
                "q_max_mean": float(q_all.detach().max(dim=1).values.mean().item()),
                "target_mean": float(y.mean().item()), "grad_norm": float(grad_norm.item())}

    def info(self) -> dict[str, float]:
        return {"epsilon": self.epsilon}

    # ---- checkpointing ------------------------------------------------------------------
    def save(self, path: str | Path, include_replay: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"q": self.q.state_dict(), "q_target": self.q_target.state_dict(), "opt": self.opt.state_dict(),
                   "steps": self.steps, "rng": self.rng.bit_generator.state, "torch_rng": torch.get_rng_state()}
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)                                  # atomic: a killed WSL never leaves a half file
        if include_replay:
            rtmp = path.with_suffix(".replay.tmp.npz")
            np.savez(rtmp, **self.buffer.state())
            rtmp.replace(path.with_suffix(".replay.npz"))

    def load(self, path: str | Path, load_replay: bool = False) -> None:
        path = Path(path)
        ck = torch.load(path, weights_only=False)
        self.q.load_state_dict(ck["q"])
        self.q_target.load_state_dict(ck["q_target"])
        self.opt.load_state_dict(ck["opt"])
        self.steps = int(ck["steps"])
        self.rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        if load_replay:
            with np.load(path.with_suffix(".replay.npz")) as d:
                self.buffer.restore(d)
