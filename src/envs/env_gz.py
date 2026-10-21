"""GAZEBO tier: a ROS 2 (Jazzy) node wrapping gz sim (Harmonic) behind the same Gym interface.

Data path
    publish  /cmd_vel   geometry_msgs/TwistStamped   (the Jazzy TurtleBot3 bridge yaml maps TwistStamped ->
                                                      gz.msgs.Twist; set gz.cmd_vel_type: Twist if you change it)
    subscribe /scan     sensor_msgs/LaserScan        (360 samples -> 24 bins via the shared ScanBinner)
    subscribe /odom_gt  nav_msgs/Odometry            (gz.pose_topic: ground-truth pose by default; /odom = wheel
                                                      odometry, which drifts after hard turns -- README section 6)
    reset    gz services via gz.transport: WorldControl.reset(all) + /world/<w>/set_pose_vector

Timing: one env step = one control period measured in *simulation* time. The step returns on the
first /scan whose stamp is >= previous stamp + dt (the generated Burger publishes one scan per control
period). With gz.lockstep the sim is paused and advanced by exactly dt per step (WorldControl.multi_step).

Reset: the robot is NEVER teleported. Measured in this project: after a set_pose teleport the Burger's
dynamics are corrupted (yaw lags the wheels, then keeps spinning for ~1 s after a stop command), which
makes the sim useless for control. Instead each reset (1) does a gz world reset, which restores the robot
at the spawn pose (world origin) with fresh joint/plugin state, and (2) moves the *static* models (walls,
obstacles, goal marker) by the rigid transform inv(layout.start), so the robot sits at layout.start in the
arena frame by construction. Moving static models is harmless (also measured).

Pose: report arena-frame pose = T_start * inv(T_odom0) * T_odom_now (odom0 is recorded after the reset).
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any

import numpy as np

from envs.common import ScanBinner, TB3BaseEnv, wrap_angle
from envs.layout import Layout


def _se2_compose(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap_angle(a[2] + b[2]))


def _se2_inverse(a: tuple[float, float, float]) -> tuple[float, float, float]:
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), wrap_angle(-a[2]))


class GazeboEnv(TB3BaseEnv):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        import rclpy
        from geometry_msgs.msg import Twist, TwistStamped
        from nav_msgs.msg import Odometry
        from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan

        self.gc = cfg["gz"]
        self._rclpy = rclpy
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self._node = rclpy.create_node(f"tb3_dqn_env_{int(time.time()) % 100000}")
        self._binner = ScanBinner(cfg)
        self._cv = threading.Condition()
        self._scan: tuple[float, np.ndarray] | None = None     # (sim stamp, bin ranges in m)
        self._odom: deque[tuple[float, float, float, float]] = deque(maxlen=64)   # stamp, x, y, yaw
        self._n_scans = 0
        self._t_last = 0.0                                      # stamp of the last consumed scan
        self._odom0 = (0.0, 0.0, 0.0)
        self._t_start = (0.0, 0.0, 0.0)
        self._sim_dts: list[float] = []

        self._twist_stamped = self.gc["cmd_vel_type"] == "TwistStamped"
        msg_cls = TwistStamped if self._twist_stamped else Twist
        self._msg_cls = msg_cls
        self._cmd_pub = self._node.create_publisher(msg_cls, self.gc["cmd_vel_topic"],
                                                    QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self._node.create_subscription(LaserScan, self.gc["scan_topic"], self._on_scan, qos_profile_sensor_data)
        # pose source: ground truth (/odom_gt, default) or wheel odometry (/odom); see gz.pose_topic
        self._node.create_subscription(Odometry, self.gc["pose_topic"], self._on_odom, qos_profile_sensor_data)

        from rclpy.executors import SingleThreadedExecutor
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        from gz.transport13 import Node as GzNode
        self._gz = GzNode()
        self._world = self.gc["world_name"]
        self._lockstep_iters = int(round(self.dt / self.gc["max_step_size"]))
        if self.gc["lockstep"] and abs(self._lockstep_iters * self.gc["max_step_size"] - self.dt) > 1e-9:
            raise ValueError("control period must be a multiple of gz.max_step_size for lockstep")
        if self.gc["lockstep"]:
            self._world_control(pause=True)

    # ---- ROS callbacks ----------------------------------------------------------------
    def _on_scan(self, msg: Any) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        bins = self._binner(np.asarray(msg.ranges, dtype=np.float64), msg.angle_min, msg.angle_increment)
        with self._cv:
            self._scan = (stamp, bins)
            self._n_scans += 1
            self._cv.notify_all()

    def _on_odom(self, msg: Any) -> None:
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._odom.append((stamp, msg.pose.pose.position.x, msg.pose.pose.position.y, yaw))

    # ---- gz services ------------------------------------------------------------------
    def _world_control(self, pause: bool | None = None, multi_step: int = 0) -> None:
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.world_control_pb2 import WorldControl
        req = WorldControl()
        if pause is not None:
            req.pause = pause
        if multi_step:
            req.pause = True                                # step N iterations, then stay paused
            req.multi_step = multi_step
        ok, rep = self._gz.request(f"/world/{self._world}/control", req, WorldControl, Boolean, 3000)
        if not ok or not rep.data:
            raise RuntimeError("gz world control service failed (is gz sim running? ./start_gz.sh)")

    def _set_poses(self, poses: list[tuple[str, float, float, float, float]]) -> None:
        """poses: (model name, x, y, z, yaw). One blocking service call for the whole layout."""
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.pose_v_pb2 import Pose_V
        req = Pose_V()
        for name, x, y, z, yaw in poses:
            p = req.pose.add()
            p.name = name
            p.position.x, p.position.y, p.position.z = x, y, z
            p.orientation.z, p.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        ok, rep = self._gz.request(f"/world/{self._world}/set_pose_vector/blocking", req, Pose_V, Boolean, 5000)
        if not ok or not rep.data:
            raise RuntimeError(f"gz set_pose_vector failed (call ok={ok}, reply={rep.data}): wrong world_name, "
                               "gz sim not running (./start_gz.sh), or the sim is too overloaded to answer in 5 s")

    # ---- stepping ---------------------------------------------------------------------
    def _publish_cmd(self, v: float, w: float) -> None:
        msg = self._msg_cls()
        if self._twist_stamped:
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            twist = msg.twist
        else:
            twist = msg
        twist.linear.x, twist.angular.z = float(v), float(w)
        self._cmd_pub.publish(msg)

    def _tick(self, v: float, w: float) -> tuple[float, np.ndarray]:
        """Command (v, w) for one control period; return (scan stamp, bins) of the first scan after it."""
        self._publish_cmd(v, w)
        target = self._t_last + self.dt - 1e-4
        if self.gc["lockstep"]:
            time.sleep(self.gc["cmd_settle_s"])           # let the bridge deliver /cmd_vel to gz
            self._world_control(multi_step=self._lockstep_iters)
        with self._cv:
            ok = self._cv.wait_for(lambda: self._scan is not None and self._scan[0] >= target,
                                   timeout=self.gc["step_timeout_s"])
            if not ok:
                raise TimeoutError(
                    f"no /scan newer than sim t={target:.3f}s within {self.gc['step_timeout_s']}s. Is gz sim running "
                    "and unpaused, and is the ros_gz bridge up (./start_gz.sh)? Same ROS_DOMAIN_ID?")
            stamp, bins = self._scan
        self._sim_dts.append(stamp - self._t_last)
        self._t_last = stamp
        return stamp, bins

    def _odom_pose_at(self, stamp: float) -> tuple[float, float, float]:
        snap = list(self._odom)
        if not snap:
            raise RuntimeError("no /odom messages received")
        _, x, y, yaw = min(snap, key=lambda o: abs(o[0] - stamp))
        return (x, y, yaw)

    def _world_pose(self, stamp: float) -> np.ndarray:
        t_world_odom = _se2_compose(self._t_start, _se2_inverse(self._odom0))
        return np.asarray(_se2_compose(t_world_odom, self._odom_pose_at(stamp)))

    def _advance(self, v: float, w: float) -> tuple[np.ndarray, np.ndarray]:
        stamp, bins = self._tick(v, w)
        return self._world_pose(stamp), bins

    def _settle(self, n: int) -> tuple[float, np.ndarray]:
        stamp, bins = self._t_last, np.zeros(self.n_bins)
        for _ in range(n):
            stamp, bins = self._tick(0.0, 0.0)
        return stamp, bins

    def _world_reset(self) -> None:
        """gz world reset: robot back at the spawn pose with fresh joints/odometry; sim time restarts at ~0."""
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.world_control_pb2 import WorldControl
        pre_t = self._t_last
        req = WorldControl()
        req.reset.all = True
        # NB (measured): the reset is applied at the next sim iteration and unpauses the sim. A request that
        # also sets `pause`, or a pause sent before the reset ran, cancels it. So: reset alone, wait until sim
        # time has visibly gone backwards, and only then re-pause (lockstep).
        ok, rep = self._gz.request(f"/world/{self._world}/control", req, WorldControl, Boolean, 5000)
        if not ok or not rep.data:
            raise RuntimeError("gz world reset failed (is gz sim running? ./start_gz.sh)")
        with self._cv:                                       # drop pre-reset samples
            self._scan = None
            self._odom.clear()
            ok = self._cv.wait_for(lambda: self._scan is not None and self._scan[0] < pre_t - 1e-6,
                                   timeout=self.gc["step_timeout_s"] * 2)
            if not ok:
                raise TimeoutError("sim time did not restart after the gz world reset")
        if self.gc["lockstep"]:
            self._world_control(pause=True)
            with self._cv:                                   # discard the scans produced while it was running
                self._scan = None
                self._odom.clear()
            for _ in range(20):                              # first post-reset scan ~0.4 s of sim time in
                self._world_control(multi_step=self._lockstep_iters)
                with self._cv:
                    if self._cv.wait_for(lambda: self._scan is not None, timeout=0.3):
                        break
        with self._cv:
            if not self._cv.wait_for(lambda: self._scan is not None, timeout=self.gc["step_timeout_s"]):
                raise TimeoutError("no /scan after the gz world reset")
            self._t_last = self._scan[0]

    def _reset_world(self, layout: Layout) -> tuple[np.ndarray, np.ndarray]:
        n = int(self.gc["reset_settle_steps"])
        self._world_reset()
        # The robot is at the world origin. Express the arena in that frame: world = inv(start) * arena.
        inv = _se2_inverse(layout.start)
        poses = [("walls", inv[0], inv[1], 0.0, inv[2])]
        gx, gy, gyaw = _se2_compose(inv, (layout.goal[0], layout.goal[1], 0.0))
        poses.append(("goal_marker", gx, gy, self.gc["marker_z"], gyaw))
        for o in layout.obstacles:                           # inactive slots stay parked (restored by the reset)
            ox, oy, oyaw = _se2_compose(inv, (o.x, o.y, o.yaw))
            poses.append((f"obs_{o.slot}", ox, oy, 0.2, oyaw))
        self._set_poses(poses)
        stamp, bins = self._settle(n)                        # let physics + sensors see the new scene
        self._t_start = tuple(layout.start)
        self._odom0 = self._odom_pose_at(stamp)
        return np.asarray(layout.start, dtype=np.float64), bins

    # ---- diagnostics ------------------------------------------------------------------
    def check_connections(self, verbose: bool = True) -> None:
        """Fail fast (with hints) if /scan, /odom, /cmd_vel or the gz services are not wired up."""
        gc = self.gc
        if gc["lockstep"]:                                   # a paused sim publishes nothing: advance until it does
            for _ in range(10):
                self._world_control(multi_step=self._lockstep_iters)
                with self._cv:
                    if self._cv.wait_for(lambda: self._scan is not None, timeout=0.3):
                        break
        deadline = time.time() + 15.0
        while time.time() < deadline and (self._scan is None or not self._odom):
            time.sleep(0.1)
        checks = {
            f"{gc['scan_topic']} publishers": (self._node.count_publishers(gc["scan_topic"]), True),
            f"{gc['pose_topic']} publishers": (self._node.count_publishers(gc["pose_topic"]), True),
            f"{gc['cmd_vel_topic']} subscribers": (self._node.count_subscribers(gc["cmd_vel_topic"]), True),
            "scan received": (int(self._scan is not None), True),
            "odom received": (int(bool(self._odom)), True),
        }
        bad = [k for k, (n, _) in checks.items() if n < 1]
        if verbose:
            print(f"[gz check] cmd_vel type={gc['cmd_vel_type']}, control {1 / self.dt:g} Hz, "
                  f"lockstep={gc['lockstep']}")
            for k, (n, _) in checks.items():
                print(f"  {'OK ' if n >= 1 else 'FAIL'} {k}: {n}")
        if bad:
            raise RuntimeError(f"Gazebo tier not connected ({', '.join(bad)}). Start the sim with ./start_gz.sh in "
                               "another terminal (same ROS_DOMAIN_ID), and check that gz.cmd_vel_type matches the bridge.")
        # Timing: measure the sim-time period of a few control steps (robot stationary).
        self._t_last = self._scan[0]
        self._sim_dts.clear()
        for _ in range(5):
            self._tick(0.0, 0.0)
        med = float(np.median(self._sim_dts))
        if verbose:
            print(f"  OK  control period: median sim dt = {med:.3f}s (expected {self.dt:.3f}s)")
        if abs(med - self.dt) > 0.2 * self.dt:
            raise RuntimeError(f"measured control period {med:.3f}s != {self.dt:.3f}s: gz.lidar_rate_hz / "
                               "control.rate_hz mismatch (restart ./start_gz.sh after editing config.yaml)")
        self._sim_dts.clear()

    def close(self) -> None:
        try:
            self._publish_cmd(0.0, 0.0)
            if self.gc["lockstep"]:
                self._world_control(pause=False)
        except Exception:  # noqa: BLE001 - best effort on shutdown
            pass
        self._executor.shutdown()
        self._node.destroy_node()
        if self._owns_rclpy and self._rclpy.ok():
            self._rclpy.shutdown()
