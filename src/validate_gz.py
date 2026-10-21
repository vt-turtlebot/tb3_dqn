"""Sim-to-sim sanity checks for the Gazebo tier (needs ./start_gz.sh running).

    python validate_gz.py [--layouts 50] [--oracle-layouts 10]

 1. Reset check      no phantom collisions at reset on every fixed layout; goal/obstacles placed
 2. LiDAR fidelity   gz bins vs the FAST tier's bins for the same pose + layout (noise is on in both)
 3. Pose consistency gz LiDAR vs the scan rendered at the env's pose (gz.pose_topic) during a scripted drive
 4. Dynamics         the same action script in FAST and Gazebo: pose divergence
 5. Oracle           a heading-only controller (no avoidance) in both tiers: per-layout outcome agreement
 6. Post-turn coast  ground-truth (gz physics pose) vs wheel-odometry yaw after hard turns: the robot must not
                     keep spinning once the wheels go straight (README section 6)
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

from envs import make_env
from envs.common import wrap_angle
from envs.layout import Layout, eval_layouts
from utils import load_config


def oracle_action(obs: np.ndarray) -> int:
    """Heading-only controller: turn toward the goal, ignore obstacles. 0=hard left .. 4=hard right."""
    h = obs[25]
    if abs(h) < 0.2:
        return 2
    if abs(h) < 0.8:
        return 1 if h > 0 else 3
    return 0 if h > 0 else 4


def run_oracle(env, layout: Layout) -> tuple[str, int]:
    obs, _ = env.reset(options={"layout": layout})
    while True:
        obs, _, term, trunc, info = env.step(oracle_action(obs))
        if term or trunc:
            return info["outcome"], int(env._steps)


def coast_check(gz, cfg: dict, holds: tuple[int, ...] = (3, 10), straight: int = 20) -> float:
    """Ground truth vs wheel odometry after a hard turn: hold hard-right for N steps, then drive straight.

    The true yaw comes from the gz physics pose topic (not odometry). Returns the worst |true - odom| yaw change
    over the straight phase (rad). Found by the transfer-gap diagnosis: with the stock Burger SDF the body keeps
    spinning after the wheels stop (README section 6)."""
    import subprocess
    import sys
    import threading
    latest: dict[str, float] = {}
    # The subscriber runs in a child process: a gz-transport Python subscription in this process deadlocks the
    # env's blocking service calls (measured: set_pose_vector times out as soon as a Pose_V callback is live).
    code = (
        "import math,sys,time\n"
        "from gz.transport13 import Node\nfrom gz.msgs10.pose_v_pb2 import Pose_V\n"
        f"name={cfg['gz']['robot_name']!r}\n"
        "def cb(m):\n"
        "  for p in m.pose:\n"
        "    if p.name==name:\n"
        "      q=p.orientation; t=m.header.stamp.sec+m.header.stamp.nsec*1e-9\n"
        "      print(t, math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)), flush=True)\n"
        "n=Node()\n"
        f"n.subscribe(Pose_V, '/world/{cfg['gz']['world_name']}/dynamic_pose/info', cb)\n"
        "while True: time.sleep(1)\n")
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)

    def reader() -> None:
        for line in proc.stdout:              # type: ignore[union-attr]
            t, y = map(float, line.split())
            latest["t"], latest["yaw"] = t, y
    threading.Thread(target=reader, daemon=True).start()
    t0 = time.time()
    while "yaw" not in latest and time.time() - t0 < 10:
        time.sleep(0.1)
    if "yaw" not in latest:
        proc.kill()
        raise RuntimeError("no robot pose on dynamic_pose/info")
    # wheel odometry (the env's own pose may be the ground truth, gz.pose_topic), via ROS: no gz-transport deadlock
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data
    wheel: dict[str, float] = {}

    def on_odom(msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        wheel["yaw"] = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    gz._node.create_subscription(Odometry, cfg["gz"]["odom_topic"], on_odom, qos_profile_sensor_data)
    empty = Layout((), (-1.0, -1.0, 0.8), (1.5, 1.5))
    worst = 0.0
    for n in holds:
        gz.reset(options={"layout": empty})
        time.sleep(0.2)
        for _ in range(n):
            gz.step(4)
        y_true0, y_odom0 = latest["yaw"], wheel["yaw"]
        true_d = odom_d = 0.0
        for _ in range(straight):
            gz.step(2)
            true_d += wrap_angle(latest["yaw"] - y_true0)
            odom_d += wrap_angle(wheel["yaw"] - y_odom0)
            y_true0, y_odom0 = latest["yaw"], wheel["yaw"]
        worst = max(worst, abs(true_d - odom_d))
        print(f"    hard-right x{n:>2} then straight x{straight}: true yaw change {true_d:+.3f} rad, "
              f"odom {odom_d:+.3f} rad (commanded 0)")
    proc.kill()
    return worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", type=int, default=50, help="layouts for the reset + LiDAR checks")
    ap.add_argument("--oracle-layouts", type=int, default=10, help="layouts for the oracle comparison (slow: real time)")
    args = ap.parse_args()

    cfg = load_config(tier="gz")
    layouts = eval_layouts(cfg)
    gz = make_env("gz", cfg)
    gz.check_connections(verbose=True)
    fast = make_env("fast", load_config(tier="fast"))
    fast.reset(seed=0)
    ok = True

    # 1 + 2: reset and LiDAR fidelity
    phantom, diffs = 0, []
    for lay in layouts[: args.layouts]:
        o_gz, _ = gz.reset(options={"layout": lay})
        o_fa, _ = fast.reset(options={"layout": lay})
        phantom += int(o_gz[:24].min() * cfg["observation"]["lidar_clip"] < cfg["task"]["collision_dist"])
        diffs.append(np.abs(o_gz[:24] - o_fa[:24]) * cfg["observation"]["lidar_clip"])
    d = np.concatenate(diffs)
    print(f"\n[1] reset: {phantom}/{args.layouts} layouts start in 'collision' (want 0)")
    print(f"[2] LiDAR gz vs fast over {args.layouts} layouts x 24 bins: mean |diff| = {d.mean():.3f} m, "
          f"p95 = {np.percentile(d, 95):.3f} m, max = {d.max():.3f} m")
    ok &= phantom == 0 and d.mean() < 0.05

    # 3 + 4: pose consistency + dynamics with a scripted drive on an empty arena
    empty = Layout((), (-1.0, -1.0, 0.0), (1.8, -1.8))
    script = [2] * 15 + [0] * 10 + [2] * 10 + [3] * 10 + [1] * 10 + [2] * 10
    gz.reset(options={"layout": empty})
    fast.reset(options={"layout": empty})
    clip = cfg["observation"]["lidar_clip"]
    lidar_err, dyn_err = [], []
    for a in script:
        og, _, te, tr, ig = gz.step(a)
        _, _, _, _, ifa = fast.step(a)
        # Is the odom-derived pose consistent with what the gz LiDAR actually sees? Render the fast-tier scan
        # at the pose gz /odom reports and compare it with the real gz scan (independent of any pose topic).
        fast._state = np.asarray(ig["pose"], dtype=np.float64).copy()
        lidar_err.append(np.abs(og[:24] - np.minimum(fast._lidar(), clip) / clip).mean() * clip)
        fast._state = np.asarray(ifa["pose"], dtype=np.float64).copy()   # restore the fast tier's own state
        dyn_err.append(math.hypot(ig["pose"][0] - ifa["pose"][0], ig["pose"][1] - ifa["pose"][1]))
        if te or tr:
            break
    yaw_err = abs(wrap_angle(ig["pose"][2] - ifa["pose"][2]))
    print(f"[3] gz LiDAR vs scan rendered at the env pose ({cfg['gz']['pose_topic']}), over {len(lidar_err)} drive steps: "
          f"mean |diff| = {np.mean(lidar_err):.3f} m, max = {np.max(lidar_err):.3f} m")
    print(f"[4] same actions, gz vs fast: position divergence mean {np.mean(dyn_err):.3f} m, final {dyn_err[-1]:.3f} m, "
          f"final yaw diff {yaw_err:.2f} rad")
    ok &= np.mean(lidar_err) < 0.05 and dyn_err[-1] < 0.25

    # 5: heading-only oracle in both tiers
    agree, n = 0, args.oracle_layouts
    rows = []
    for i, lay in enumerate(layouts[:n]):
        og, sg = run_oracle(gz, lay)
        of, sf = run_oracle(fast, lay)
        agree += og == of
        rows.append(f"    layout {i:>2}: gz {og:<9} ({sg:>3} steps)   fast {of:<9} ({sf:>3} steps)")
    print(f"[5] heading-only oracle: outcome agreement {agree}/{n} layouts")
    print("\n".join(rows))
    print("[6] post-turn coast (true vs odom yaw while driving straight after a hard turn):")
    coast = coast_check(gz, cfg)
    print(f"    worst |true - odom| = {coast:.3f} rad (want < 0.3)")
    ok &= coast < 0.3
    print("\nOVERALL:", "PASS" if ok else "CHECK ABOVE (thresholds: LiDAR mean<0.05 m, odom/LiDAR mean<0.05 m, "
          "dyn div<0.25 m, coast<0.3 rad)")
    gz.close()
    print(flush=True)
    os._exit(0)          # gz-transport's teardown can hang the interpreter; everything is already printed


if __name__ == "__main__":
    main()
