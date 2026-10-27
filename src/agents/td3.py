"""TD3 (Fujimoto et al. 2018): twin critics, delayed actor/target updates, target policy smoothing.

Actions live in the env's physical units (rad/s); the actor outputs tanh * act_high.
`done` is true only for real terminations; time-limit truncation still bootstraps (as in DQN).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import Agent
from agents.dqn import ReplayBuffer


def _mlp(inp: int, out: int, hidden: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(inp, h), nn.ReLU()]
        inp = h
    return nn.Sequential(*layers, nn.Linear(inp, out))


class TD3Agent(Agent):
    def __init__(self, cfg: dict[str, Any], obs_dim: int, act_high: np.ndarray, seed: int = 0) -> None:
        tc, rc = cfg["td3"], cfg["run"]
        hidden = list(tc["hidden"])
        self.act_high = np.asarray(act_high, dtype=np.float32)
        self.act_dim = len(self.act_high)
        self._high_t = torch.as_tensor(self.act_high)
        self.gamma, self.tau = float(tc["gamma"]), float(tc["tau"])
        self.batch_size, self.policy_delay = int(tc["batch_size"]), int(tc["policy_delay"])
        self.policy_noise, self.noise_clip = float(tc["policy_noise"]), float(tc["noise_clip"])
        self.expl_noise, self.grad_clip = float(tc["expl_noise"]), float(tc["grad_clip"])
        self.warmup_steps = int(rc["warmup_steps"])

        self.actor = _mlp(obs_dim, self.act_dim, hidden)
        self.q1, self.q2 = _mlp(obs_dim + self.act_dim, 1, hidden), _mlp(obs_dim + self.act_dim, 1, hidden)
        self.actor_t, self.q1_t, self.q2_t = (copy.deepcopy(m).requires_grad_(False)
                                              for m in (self.actor, self.q1, self.q2))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(tc["lr_actor"]))
        self.critic_opt = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=float(tc["lr_critic"]))
        self.buffer = ReplayBuffer(int(tc["buffer_size"]), obs_dim, self.act_dim)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.updates = 0    # critic updates; drives the delayed actor update (must survive resume)

    def _pi(self, net: nn.Module, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(net(obs)) * self._high_t

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        if explore and self.steps < self.warmup_steps:
            return self.rng.uniform(-self.act_high, self.act_high).astype(np.float32)
        with torch.no_grad():
            a = self._pi(self.actor, torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        if explore:
            a = a + self.rng.normal(0.0, self.expl_noise * self.act_high)
        return np.clip(a, -self.act_high, self.act_high).astype(np.float32)

    def observe(self, obs: np.ndarray, action: Any, reward: float, next_obs: np.ndarray,
                terminated: bool, truncated: bool) -> None:
        self.buffer.add(obs, np.asarray(action, dtype=np.float32), reward, next_obs, terminated)
        self.steps += 1

    def update(self) -> dict[str, float]:
        if self.steps < self.warmup_steps or self.buffer.size < self.batch_size:
            return {}
        obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size, self.rng)
        with torch.no_grad():
            noise = (torch.as_tensor(self.rng.normal(0.0, 1.0, act.shape), dtype=torch.float32)
                     * self.policy_noise * self._high_t).clamp(-self.noise_clip * self._high_t, self.noise_clip * self._high_t)
            a2 = (self._pi(self.actor_t, next_obs) + noise).clamp(-self._high_t, self._high_t)
            sa2 = torch.cat([next_obs, a2], dim=1)
            y = rew + self.gamma * torch.min(self.q1_t(sa2), self.q2_t(sa2)).squeeze(1) * (1.0 - done)
        sa = torch.cat([obs, act], dim=1)
        q1, q2 = self.q1(sa).squeeze(1), self.q2(sa).squeeze(1)
        loss = F.smooth_l1_loss(q1, y) + F.smooth_l1_loss(q2, y)
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([*self.q1.parameters(), *self.q2.parameters()], self.grad_clip)
        self.critic_opt.step()
        self.updates += 1
        out = {"loss": float(loss.item()), "q_mean": float(q1.mean().item()), "target_mean": float(y.mean().item())}
        if self.updates % self.policy_delay == 0:
            actor_loss = -self.q1(torch.cat([obs, self._pi(self.actor, obs)], dim=1)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_opt.step()
            with torch.no_grad():   # polyak update
                for net, tgt in ((self.actor, self.actor_t), (self.q1, self.q1_t), (self.q2, self.q2_t)):
                    for p, tp in zip(net.parameters(), tgt.parameters()):
                        tp.mul_(1 - self.tau).add_(self.tau * p)
            out["actor_loss"] = float(actor_loss.item())
        return out

    def info(self) -> dict[str, float]:
        return {}   # no epsilon: exploration is gaussian noise

    _NETS = ("actor", "q1", "q2", "actor_t", "q1_t", "q2_t", "actor_opt", "critic_opt")

    def save(self, path: str | Path, include_replay: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: getattr(self, k).state_dict() for k in self._NETS}
        payload.update(steps=self.steps, updates=self.updates, rng=self.rng.bit_generator.state,
                       torch_rng=torch.get_rng_state())
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        if include_replay:
            rtmp = path.with_suffix(".replay.tmp.npz")
            np.savez(rtmp, **self.buffer.state())
            rtmp.replace(path.with_suffix(".replay.npz"))

    def load(self, path: str | Path, load_replay: bool = False) -> None:
        path = Path(path)
        ck = torch.load(path, weights_only=False)
        for k in self._NETS:
            getattr(self, k).load_state_dict(ck[k])
        self.steps, self.updates = int(ck["steps"]), int(ck["updates"])
        self.rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        if load_replay:
            with np.load(path.with_suffix(".replay.npz")) as d:
                self.buffer.restore(d)
