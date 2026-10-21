"""FAST tier: pure-Python/NumPy 2D differential-drive simulator.

It emulates the Burger's LDS-01 (360 samples, 0.12-3.5 m, gaussian noise) and then runs the
*same* ScanBinner as the Gazebo tier, so both tiers hand the agent the same kind of
observation. Obstacles are circles and oriented boxes; the arena walls are four thin boxes
matching the generated Gazebo world.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from envs.common import ScanBinner, TB3BaseEnv, wrap_angle
from envs.layout import Layout


class FastEnv(TB3BaseEnv):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        lc = cfg["lidar"]
        self._binner = ScanBinner(cfg)
        n = int(lc["n_samples"])
        self._angle_min = float(lc["angle_min"])
        self._angle_inc = (lc["angle_max"] - lc["angle_min"]) / (n - 1)
        self._ray_ang = self._angle_min + np.arange(n) * self._angle_inc
        self._substeps = int(cfg["run"]["sim_substeps"])
        self._act_noise = float(cfg["run"].get("action_noise_std", 0.0))
        self._v_actual = 0.0
        # Optional actuation-fidelity model (README section 5; all default 0 = the original instant actuation).
        self._delay = float(cfg["run"].get("act_delay_s", 0.0))            # pure command delay
        self._delay_jitter = float(cfg["run"].get("act_delay_jitter_s", 0.0))   # per-episode U(-j, +j) on the delay
        self._lag_tau = float(cfg["run"].get("act_lag_tau_s", 0.0))        # first-order lag on w
        if not 0.0 <= self._delay + self._delay_jitter < self.dt:
            raise ValueError("act_delay_s + act_delay_jitter_s must be in [0, control period)")
        self._ep_delay = self._delay
        self._w_actual = 0.0
        self._cmd_prev = (0.0, 0.0)

        # Static arena walls as boxes (cx, cy, hx, hy, yaw): inner faces at +-size/2.
        half, t = cfg["arena"]["size"] / 2.0, cfg["arena"]["wall_thickness"]
        c, hl = half + t / 2.0, half + t
        self._walls = np.array([[c, 0, t / 2, hl, 0], [-c, 0, t / 2, hl, 0],
                                [0, c, hl, t / 2, 0], [0, -c, hl, t / 2, 0]], dtype=np.float64)
        self._circles = np.zeros((0, 3))
        self._boxes = self._walls.copy()

    # ---- world ------------------------------------------------------------------------
    def _reset_world(self, layout: Layout) -> tuple[np.ndarray, np.ndarray]:
        oc = self.cfg["obstacles"]
        circles, boxes = [], [self._walls]
        h = oc["box_size"] / 2.0
        for o in layout.obstacles:
            if o.kind == "cylinder":
                circles.append([o.x, o.y, oc["cylinder_radius"]])
            else:
                boxes.append(np.array([[o.x, o.y, h, h, o.yaw]]))
        self._circles = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
        self._boxes = np.concatenate(boxes, axis=0)
        self._state = np.asarray(layout.start, dtype=np.float64).copy()
        self._v_actual = 0.0
        self._w_actual = 0.0
        self._cmd_prev = (0.0, 0.0)
        if self._delay_jitter > 0.0:
            self._ep_delay = self._delay + float(self.np_random.uniform(-self._delay_jitter, self._delay_jitter))
        return self._state.copy(), self._lidar()

    def _advance(self, v: float, w: float) -> tuple[np.ndarray, np.ndarray]:
        if self._act_noise > 0.0:
            v += self.np_random.normal(0.0, self._act_noise)
            w += self.np_random.normal(0.0, self._act_noise)
            v = float(np.clip(v, self.rc["v_min"], self.rc["v_max"]))
            w = float(np.clip(w, -self.rc["w_max"], self.rc["w_max"]))
        if self._ep_delay > 0.0 or self._lag_tau > 0.0:
            return self._advance_delayed(v, w)
        h = self.dt / self._substeps
        x, y, yaw = self._state
        max_dv = self.rc["max_linear_accel"] * h   # same accel limit as the Gazebo DiffDrive plugin
        for _ in range(self._substeps):
            self._v_actual += float(np.clip(v - self._v_actual, -max_dv, max_dv))
            x += self._v_actual * np.cos(yaw + 0.5 * w * h) * h
            y += self._v_actual * np.sin(yaw + 0.5 * w * h) * h
            yaw = wrap_angle(yaw + w * h)
        self._state = np.array([x, y, yaw])
        return self._state.copy(), self._lidar()

    def _advance_delayed(self, v: float, w: float) -> tuple[np.ndarray, np.ndarray]:
        """Like `_advance`, but the previous command stays active for the first `delay` seconds of the period
        (measured in Gazebo, README section 6) and w optionally follows the command with a first-order lag."""
        d = self._ep_delay
        segments = [(self._cmd_prev, d), ((v, w), self.dt - d)] if d > 0.0 else [((v, w), self.dt)]
        h_nom = self.dt / self._substeps
        x, y, yaw = self._state
        for (vc, wc), dur in segments:
            n = max(1, int(np.ceil(dur / h_nom - 1e-9)))
            h = dur / n
            max_dv = self.rc["max_linear_accel"] * h
            # first-order lag: w(t) = wc + (w0 - wc) e^{-t/tau}; `keep` = remaining fraction after h,
            # `mean_keep` = its average over the sub-step (so the yaw integral is exact). No lag: both 0.
            keep = float(np.exp(-h / self._lag_tau)) if self._lag_tau > 0.0 else 0.0
            mean_keep = self._lag_tau / h * (1.0 - keep) if self._lag_tau > 0.0 else 0.0
            for _ in range(n):
                self._v_actual += float(np.clip(vc - self._v_actual, -max_dv, max_dv))
                w_mean = wc + (self._w_actual - wc) * mean_keep
                x += self._v_actual * np.cos(yaw + 0.5 * w_mean * h) * h
                y += self._v_actual * np.sin(yaw + 0.5 * w_mean * h) * h
                yaw = wrap_angle(yaw + w_mean * h)
                self._w_actual = wc + (self._w_actual - wc) * keep
        self._cmd_prev = (v, w)
        self._state = np.array([x, y, yaw])
        return self._state.copy(), self._lidar()

    # ---- LiDAR ------------------------------------------------------------------------
    def _lidar(self) -> np.ndarray:
        """Cast the 360 rays, add noise, and reduce to the observation bins (metres)."""
        x, y, yaw = self._state
        ox = x + self.rc["lidar_offset_x"] * np.cos(yaw)
        oy = y + self.rc["lidar_offset_x"] * np.sin(yaw)
        th = yaw + self._ray_ang
        dx, dy = np.cos(th), np.sin(th)
        best = np.full(th.shape, np.inf)

        if len(self._circles):
            ocx, ocy = ox - self._circles[:, 0], oy - self._circles[:, 1]
            b = dx[:, None] * ocx[None] + dy[:, None] * ocy[None]
            disc = b * b - (ocx ** 2 + ocy ** 2 - self._circles[:, 2] ** 2)[None]
            t = -b - np.sqrt(np.maximum(disc, 0.0))
            t = np.where((disc >= 0) & (t > 0), t, np.inf)
            best = np.minimum(best, t.min(axis=1))

        bx, by, hx, hy, byaw = self._boxes.T
        c, s = np.cos(byaw), np.sin(byaw)
        px, py = ox - bx, oy - by
        lx, ly = c * px + s * py, -s * px + c * py                 # ray origin in box frames
        ldx = c[None] * dx[:, None] + s[None] * dy[:, None]        # ray dirs in box frames
        ldy = -s[None] * dx[:, None] + c[None] * dy[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            t1x, t2x = (-hx - lx)[None] / ldx, (hx - lx)[None] / ldx
            t1y, t2y = (-hy - ly)[None] / ldy, (hy - ly)[None] / ldy
        tmin = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
        tmax = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
        t = np.where((tmax >= tmin) & (tmax > 0), np.maximum(tmin, 0.0), np.inf)
        best = np.minimum(best, t.min(axis=1))

        lc = self.cfg["lidar"]
        raw = np.where(np.isfinite(best), best, lc["range_max"])
        if lc["noise_std"] > 0.0:
            raw = raw + self.np_random.normal(0.0, lc["noise_std"], size=raw.shape)
        return self._binner(raw, self._angle_min, self._angle_inc)
