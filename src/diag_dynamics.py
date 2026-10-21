"""Actuation step response of a tier (README section 6, "transfer gap"; plan item A2).

    python diag_dynamics.py --env fast
    python diag_dynamics.py --env gz              # needs ./start_gz.sh (same GZ_PARTITION / ROS_DOMAIN_ID)
    python diag_dynamics.py --env fast --set run.act_delay_s=0.03   # check a fast-tier fidelity option

Drives a fixed script of discrete actions on an empty arena (`--repeats` times) and measures, per control
step k, the yaw increment dyaw_k and travelled distance ds_k of the pose the env reports. A pure command
delay d (the command for step k only takes effect d seconds after the step's observation) gives

    dyaw_k = w_k * (dt - d) + w_{k-1} * d          (likewise ds_k with v)

so a least-squares fit of dyaw_k on (w_k*dt, w_{k-1}*dt) gives the "current" and "previous" gains
(a, b) and d = b / (a + b) * dt. A first-order lag would instead also leak into w_{k-2}; the fit reports that
coefficient too. In Gazebo the 30 Hz /odom twist is logged as well (the wheel-odometry step response).
Writes results/transfer_diag/dynamics_<env>[_<tag>].npz and prints the fit.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from envs import make_env
from envs.common import wrap_angle
from envs.layout import Layout
from utils import apply_overrides, load_config, results_dir

# hard-left .. hard-right = 0..4. Holds of 8 steps; every action follows every other at least once.
SCRIPT = ([2] * 10 + [0] * 8 + [2] * 8 + [4] * 8 + [0] * 8 + [1] * 8 + [3] * 8 + [2] * 8 + [4] * 8 + [1] * 8
          + [0] * 8 + [3] * 8 + [2] * 8)


def fit(dy: np.ndarray, cur: np.ndarray, prev: np.ndarray, prev2: np.ndarray) -> dict[str, float]:
    X = np.stack([cur, prev, prev2], axis=1)
    coef, *_ = np.linalg.lstsq(X, dy, rcond=None)
    res = dy - X @ coef
    return {"a_cur": float(coef[0]), "b_prev": float(coef[1]), "c_prev2": float(coef[2]),
            "rmse": float(np.sqrt(np.mean(res ** 2)))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", choices=["fast", "gz"], default="fast")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_config(tier=args.env)
    apply_overrides(cfg, args.overrides)
    env = make_env(args.env, cfg)
    twist: list[tuple[float, float, float]] = []          # gz only: (sim stamp, vx, wz) from /odom
    if args.env == "gz":
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data

        def on_odom(msg: Odometry) -> None:
            st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            twist.append((st, msg.twist.twist.linear.x, msg.twist.twist.angular.z))
        env._node.create_subscription(Odometry, cfg["gz"]["odom_topic"], on_odom, qos_profile_sensor_data)
        env.check_connections(verbose=False)
    env.reset(seed=0)
    dt = env.dt
    w_tab = np.asarray(cfg["actions"]["discrete"]["w"])
    v_cmd = float(cfg["actions"]["discrete"]["v"])
    layout = Layout((), (-0.6, -0.6, 0.6), (1.5, 1.5))     # empty arena, far from walls for the whole script

    rows, traces = [], []
    for rep in range(args.repeats):
        _, info = env.reset(options={"layout": layout})
        pose = [info["pose"].copy()]
        stamps = [getattr(env, "_t_last", 0.0)]
        twist.clear()
        for a in SCRIPT:
            _, _, term, trunc, info = env.step(a)
            pose.append(info["pose"].copy())
            stamps.append(getattr(env, "_t_last", 0.0))
            if term or trunc:
                raise RuntimeError(f"episode ended ({info['outcome']}) during the script; move the start pose")
        pose = np.asarray(pose)
        w_cmd = w_tab[np.asarray(SCRIPT)]
        dyaw = np.array([wrap_angle(pose[k + 1, 2] - pose[k, 2]) for k in range(len(SCRIPT))])
        ds = np.hypot(np.diff(pose[:, 0]), np.diff(pose[:, 1]))
        rows.append((dyaw, ds, w_cmd))
        traces.append({"stamps": np.asarray(stamps), "pose": pose, "twist": np.asarray(twist).reshape(-1, 3)})

    # Fit on all steps after the start-up transient (k >= 3 so w_{k-2} exists and v has ramped up).
    dy = np.concatenate([r[0][3:] for r in rows])
    cur = np.concatenate([r[2][3:] for r in rows]) * dt
    prev = np.concatenate([r[2][2:-1] for r in rows]) * dt
    prev2 = np.concatenate([r[2][1:-2] for r in rows]) * dt
    fw = fit(dy, cur, prev, prev2)
    d_w = fw["b_prev"] / (fw["a_cur"] + fw["b_prev"]) * dt
    ds_all = np.stack([r[1] for r in rows])
    print(f"[{args.env}{(' ' + args.tag) if args.tag else ''}] yaw fit over {len(dy)} steps: "
          f"gain(w_k)={fw['a_cur']:.3f}  gain(w_k-1)={fw['b_prev']:.3f}  gain(w_k-2)={fw['c_prev2']:.3f}  "
          f"total={fw['a_cur'] + fw['b_prev'] + fw['c_prev2']:.3f}  rmse={fw['rmse']:.4f} rad")
    print(f"    -> effective command delay d = {1000 * d_w:.1f} ms  ({100 * d_w / dt:.0f}% of the {1000 * dt:.0f} ms period)")
    print(f"    start-up distance per step (mean of {args.repeats}): "
          + " ".join(f"{x:.4f}" for x in ds_all[:, :5].mean(0)) + f"  (steady state {v_cmd * dt:.4f} m)")
    print(f"    steady-state ds/(v dt) = {ds_all[:, 5:].mean() / (v_cmd * dt):.3f}")
    # Per-transition view: yaw increment of the first step after each change of w, as a fraction of w_k*dt.
    fr = []
    for dyaw, _, w_cmd in rows:
        for k in range(1, len(w_cmd)):
            if w_cmd[k] != w_cmd[k - 1] and abs(w_cmd[k] - w_cmd[k - 1]) >= 1.4:
                fr.append((dyaw[k] - w_cmd[k - 1] * dt) / ((w_cmd[k] - w_cmd[k - 1]) * dt))
    print(f"    first step after a |dw|>=1.5 change reaches {100 * np.mean(fr):.0f}% +- {100 * np.std(fr):.0f}% "
          f"of the commanded change (n={len(fr)})")

    if args.env == "gz" and traces[0]["twist"].size:
        # odom twist step response: time from the step's scan stamp until |wz - w_new| < 10% of the change.
        lat = []
        for tr in traces:
            tw, st = tr["twist"], tr["stamps"]
            w_cmd = w_tab[np.asarray(SCRIPT)]
            for k in range(1, len(SCRIPT)):
                dwk = w_cmd[k] - w_cmd[k - 1]
                if abs(dwk) < 1.4:
                    continue
                m = (tw[:, 0] > st[k]) & (tw[:, 0] <= st[k] + 3 * dt)
                hit = np.nonzero(np.abs(tw[m, 2] - w_cmd[k]) < 0.1 * abs(dwk))[0]
                if hit.size:
                    lat.append(tw[m][hit[0], 0] - st[k])
        if lat:
            print(f"    /odom twist reaches 90% of a |dw|>=1.5 step {1000 * np.mean(lat):.0f} ms (median "
                  f"{1000 * np.median(lat):.0f} ms, n={len(lat)}) after the step's scan stamp (odom is 30 Hz: +-17 ms)")
        sim_dts = np.asarray(getattr(env, "_sim_dts", []))
        if sim_dts.size:
            print(f"    control period (sim time): median {1000 * np.median(sim_dts):.1f} ms, "
                  f"p5-p95 {1000 * np.percentile(sim_dts, 5):.1f}-{1000 * np.percentile(sim_dts, 95):.1f} ms")

    out = results_dir(cfg) / "transfer_diag"
    out.mkdir(parents=True, exist_ok=True)
    fn = out / f"dynamics_{args.env}{('_' + args.tag) if args.tag else ''}.npz"
    np.savez(fn, script=np.asarray(SCRIPT), dt=dt, **{f"rep{i}_{k}": v for i, t in enumerate(traces) for k, v in t.items()},
             **{f"rep{i}_dyaw": r[0] for i, r in enumerate(rows)}, **{f"rep{i}_ds": r[1] for i, r in enumerate(rows)},
             fit_yaw=np.array([fw["a_cur"], fw["b_prev"], fw["c_prev2"]]), delay_s=d_w)
    print(f"    saved {fn}")
    if args.env == "gz":
        env.close()
        os._exit(0)          # gz-transport teardown can hang


if __name__ == "__main__":
    main()
