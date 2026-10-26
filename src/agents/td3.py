"""Vanilla TD3 (Fujimoto et al. 2018) -- the continuous-action counterpart of agents/dqn.py. No extras.

The three TD3 ingredients, nothing else:
  * clipped double-Q:          y = r + gamma * min(Q1', Q2')(s', a~') * (1 - done)
  * target policy smoothing:   a~' = clip(pi'(s') + clip(N(0, target_noise), -c, c), -1, 1)
  * delayed policy updates:    actor and the (Polyak) target nets are updated every `policy_delay` critic updates.

Actions are learned in a normalised box [-1, 1]^2 (tanh actor output); `act()` maps them to the env's
Box([v_min, -w_max], [v_max, w_max]) and the replay buffer stores the normalised action.

`done` is true only for real terminations (goal / collision), as in DQN: a time-limit truncation still bootstraps.
All agent randomness (warm-up actions, exploration noise, replay indices, smoothing noise) comes from one NumPy
generator that is checkpointed, so fast-tier runs stay bit-reproducible across resumes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import Agent
from agents.dqn import ReplayBuffer


def mlp(inp: int, out: int, hidden: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = inp
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers.append(nn.Linear(last, out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int]) -> None:
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))          # normalised action in [-1, 1]


class TwinCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int]) -> None:
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=1)
        return self.q1(x).squeeze(1), self.q2(x).squeeze(1)

    def q1_only(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([obs, act], dim=1)).squeeze(1)


class ContinuousReplayBuffer(ReplayBuffer):
    """DQN's uniform replay with a float32 (capacity, act_dim) action array; add/sample/state/restore are reused."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        super().__init__(capacity, obs_dim)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)


class TD3Agent(Agent):
    def __init__(self, cfg: dict[str, Any], obs_dim: int, act_low: np.ndarray, act_high: np.ndarray,
                 seed: int = 0) -> None:
        tc, rc = cfg["td3"], cfg["run"]
        self.act_low = np.asarray(act_low, dtype=np.float64)
        self.act_high = np.asarray(act_high, dtype=np.float64)
        self.act_dim = len(self.act_low)
        self.gamma = float(tc["gamma"])
        self.batch_size = int(tc["batch_size"])
        self.train_freq = int(tc["train_freq"])
        self.tau = float(tc["tau"])
        self.policy_delay = int(tc["policy_delay"])
        self.expl_noise = float(tc["expl_noise"])            # std, in normalised action units
        self.target_noise = float(tc["target_noise"])
        self.target_noise_clip = float(tc["target_noise_clip"])
        self.grad_clip = float(tc["grad_clip"])               # <= 0: no clipping (vanilla TD3); the norm is still logged
        self.loss_name = tc["loss"]
        self.reward_scale = float(tc["reward_scale"])
        self.warmup_steps = int(rc["warmup_steps"])

        hidden = list(tc["hidden"])
        self.actor = Actor(obs_dim, self.act_dim, hidden)
        self.critic = TwinCritic(obs_dim, self.act_dim, hidden)
        self.actor_target = Actor(obs_dim, self.act_dim, hidden)
        self.critic_target = TwinCritic(obs_dim, self.act_dim, hidden)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.requires_grad_(False)
        self.critic_target.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(tc["actor_lr"]))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(tc["critic_lr"]))
        self.buffer = ContinuousReplayBuffer(int(tc["buffer_size"]), obs_dim, self.act_dim)
        self.rng = np.random.default_rng(seed)   # warm-up actions, exploration noise, replay sampling, smoothing noise
        self.steps = 0
        self.n_updates = 0                       # critic updates so far (drives the policy delay)

    # ---- action scaling -----------------------------------------------------------------
    def to_env(self, a: np.ndarray) -> np.ndarray:
        """Normalised [-1, 1] -> env units [low, high]."""
        return (self.act_low + (np.asarray(a, dtype=np.float64) + 1.0) * 0.5 * (self.act_high - self.act_low)
                ).astype(np.float32)

    def to_norm(self, a: np.ndarray) -> np.ndarray:
        """Env units -> normalised [-1, 1] (clipped: the env clips out-of-range commands the same way)."""
        a = np.asarray(a, dtype=np.float64)
        return np.clip(2.0 * (a - self.act_low) / (self.act_high - self.act_low) - 1.0, -1.0, 1.0).astype(np.float32)

    # ---- policy -------------------------------------------------------------------------
    def _pi(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        if explore and self.steps < self.warmup_steps:
            a = self.rng.uniform(-1.0, 1.0, size=self.act_dim)
        else:
            a = self._pi(obs).astype(np.float64)
            if explore:
                a = np.clip(a + self.rng.normal(0.0, self.expl_noise, size=self.act_dim), -1.0, 1.0)
        return self.to_env(a)

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """[Q1(obs, pi(obs))] in unscaled reward units, for diagnostics: evaluate.py's overestimation probe takes the
        max of this (a single entry, so never the larger of the twin critics) and compares it with the realised return."""
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return self.critic.q1_only(o, self.actor(o)).numpy() / self.reward_scale

    # ---- learning -----------------------------------------------------------------------
    def observe(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray,
                terminated: bool, truncated: bool) -> None:
        self.buffer.add(obs, self.to_norm(action), reward, next_obs, terminated)   # truncation is NOT terminal
        self.steps += 1

    def update(self) -> dict[str, float]:
        if (self.steps < self.warmup_steps or self.buffer.size < self.batch_size
                or self.steps % self.train_freq != 0):
            return {}
        obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size, self.rng)
        noise = np.clip(self.rng.normal(0.0, self.target_noise, size=(self.batch_size, self.act_dim)),
                        -self.target_noise_clip, self.target_noise_clip)
        with torch.no_grad():
            next_act = (self.actor_target(next_obs) + torch.from_numpy(noise.astype(np.float32))).clamp(-1.0, 1.0)
            tq1, tq2 = self.critic_target(next_obs, next_act)
            y = rew * self.reward_scale + self.gamma * torch.min(tq1, tq2) * (1.0 - done)
        q1, q2 = self.critic(obs, act)
        lf = F.smooth_l1_loss if self.loss_name == "huber" else F.mse_loss
        loss = lf(q1, y) + lf(q2, y)
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(),
                                             self.grad_clip if self.grad_clip > 0 else float("inf"))   # pre-clip norm
        self.critic_opt.step()
        self.n_updates += 1

        metrics = {"loss": float(loss.item()), "q_mean": float(q1.mean().item()),
                   "target_mean": float(y.mean().item()), "grad_norm": float(grad_norm.item())}
        if self.n_updates % self.policy_delay == 0:
            actor_loss = -self.critic.q1_only(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            with torch.no_grad():
                for net, tgt in ((self.actor, self.actor_target), (self.critic, self.critic_target)):
                    for p, tp in zip(net.parameters(), tgt.parameters()):
                        tp.mul_(1.0 - self.tau).add_(p, alpha=self.tau)
            metrics["actor_loss"] = float(actor_loss.item())
        # Logging only. q_max_mean = Q1(s, pi(s)): the policy's own value, the continuous analogue of DQN's max_a Q(s, a)
        # (plot.py's stability panel reads this column).
        with torch.no_grad():
            metrics["q_max_mean"] = float(self.critic.q1_only(obs, self.actor(obs)).mean().item())
        return metrics

    # ---- checkpointing ------------------------------------------------------------------
    def save(self, path: str | Path, include_replay: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"agent": "td3", "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                   "actor_target": self.actor_target.state_dict(), "critic_target": self.critic_target.state_dict(),
                   "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
                   "steps": self.steps, "n_updates": self.n_updates, "rng": self.rng.bit_generator.state,
                   "torch_rng": torch.get_rng_state(), "act_low": self.act_low, "act_high": self.act_high}
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
        if ck.get("agent") != "td3":
            raise ValueError(f"{path} is not a TD3 checkpoint (evaluate.py needs --agent matching the checkpoint)")
        if not (np.allclose(ck["act_low"], self.act_low) and np.allclose(ck["act_high"], self.act_high)):
            raise ValueError(f"{path}: action bounds {ck['act_low']}..{ck['act_high']} differ from the env's "
                             f"{self.act_low}..{self.act_high} (robot.v_*/w_max changed?)")
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.actor_target.load_state_dict(ck["actor_target"])
        self.critic_target.load_state_dict(ck["critic_target"])
        self.actor_opt.load_state_dict(ck["actor_opt"])
        self.critic_opt.load_state_dict(ck["critic_opt"])
        self.steps, self.n_updates = int(ck["steps"]), int(ck["n_updates"])
        self.rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        if load_replay:
            with np.load(path.with_suffix(".replay.npz")) as d:
                self.buffer.restore(d)
