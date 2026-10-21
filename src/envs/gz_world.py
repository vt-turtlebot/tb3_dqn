"""Generate the Gazebo (gz sim / Harmonic) assets for the GAZEBO tier from config.yaml.

Writes into `gz.generated_dir` (default ../gz_generated):
    world.sdf                     arena walls, obstacle pool, goal marker, robot
    bridge.yaml                   ros_gz_bridge config (/clock /odom /scan /cmd_vel)
    models/tb3_burger_rl/         the stock Burger SDF patched for RL (lidar rate, no ray viz)

Generating from config guarantees the arena size, obstacle sizes, control/lidar rate and the
cmd_vel message type agree with the FAST tier. Run via `./start_gz.sh` or:
    python -m envs.gz_world
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from envs.layout import slot_kind

MODEL_NAME = "tb3_burger_rl"


def find_tb3_gazebo_share() -> Path:
    """Locate the turtlebot3_gazebo share dir (workspace build or apt install)."""
    env = os.environ.get("TB3_GAZEBO_SHARE")
    if env and Path(env).is_dir():
        return Path(env)
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("turtlebot3_gazebo"))
    except Exception:  # noqa: BLE001 - ament not sourced
        pass
    for cand in (Path.home() / "turtlebot3_ws/install/turtlebot3_gazebo/share/turtlebot3_gazebo",
                 Path("/opt/ros/jazzy/share/turtlebot3_gazebo")):
        if cand.is_dir():
            return cand
    raise FileNotFoundError("turtlebot3_gazebo not found: source ROS and the colcon workspace "
                            "(see setup.sh) or set TB3_GAZEBO_SHARE")


def park_pose(slot: int, cfg: dict[str, Any]) -> tuple[float, float]:
    px, py = cfg["gz"]["park_xy"]
    return px + 2.0 * slot, py


def _patched_burger(cfg: dict[str, Any], share: Path) -> str:
    sdf = (share / "models/turtlebot3_burger/model.sdf").read_text()
    sdf, n1 = re.subn(r'<model name="turtlebot3_burger">', f'<model name="{MODEL_NAME}">', sdf)
    # lidar: one scan per control step; the IMU keeps its own 200 Hz rate
    sdf, n2 = re.subn(r"<update_rate>5</update_rate>", f"<update_rate>{cfg['gz']['lidar_rate_hz']:g}</update_rate>", sdf)
    sdf, n3 = re.subn(r"(<sensor name=\"hls_lfcd_lds\"[^>]*>\s*<always_on>true</always_on>\s*)<visualize>true</visualize>",
                      r"\1<visualize>false</visualize>", sdf)
    sdf, n4 = re.subn(r"(<noise>.*?<stddev>)[0-9.eE+-]+(</stddev>)",
                      rf"\g<1>{cfg['lidar']['noise_std']:g}\g<2>", sdf, flags=re.S)
    if (n1, n2, n3) != (1, 1, 1) or n4 < 1:
        raise RuntimeError(f"Burger SDF layout changed (patch counts {n1,n2,n3,n4}); update envs/gz_world.py")
    if cfg["gz"]["fix_base_footprint_inertia"]:
        # The stock SDF declares `<link name="base_footprint"/>` with no <inertial>, so SDF's default (1 kg, I = 1 kg m^2)
        # applies and the fixed base_joint merges it into the chassis: yaw inertia ~1.02 kg m^2 instead of ~0.02.
        # After a hard turn the body then keeps spinning on its wheels while wheel odometry reports a stop
        # (README section 6, transfer gap). Replace it with a negligible inertial.
        sdf, n5 = re.subn(r'<link name="base_footprint"\s*/>',
                          '<link name="base_footprint"><inertial><mass>0.001</mass><inertia><ixx>1e-7</ixx><ixy>0</ixy>'
                          '<ixz>0</ixz><iyy>1e-7</iyy><iyz>0</iyz><izz>1e-7</izz></inertia></inertial></link>', sdf)
        if n5 != 1:
            raise RuntimeError("Burger SDF: empty base_footprint link not found; update envs/gz_world.py")
    # Ground-truth pose publisher (gz topic `odom_gt`, sim-time stamps): the env's pose source by default
    # (gz.pose_topic), because wheel odometry drifts by tenths of a radian per episode (README section 6).
    gt = ('<plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">'
          f'<odom_frame>odom_gt</odom_frame><robot_base_frame>base_footprint</robot_base_frame>'
          f'<odom_topic>{cfg["gz"]["gt_odom_topic"].lstrip("/")}</odom_topic>'
          '<odom_publish_frequency>50</odom_publish_frequency><dimensions>2</dimensions></plugin>\n  </model>')
    sdf, n6 = re.subn(r"</model>\s*</sdf>", gt + "\n</sdf>", sdf)
    if n6 != 1:
        raise RuntimeError("Burger SDF: closing </model></sdf> not found; update envs/gz_world.py")
    return sdf


def _box_link(name: str, x: float, y: float, sx: float, sy: float, sz: float, rgba: str) -> str:
    return f"""
      <link name="{name}"><pose>{x:.4f} {y:.4f} {sz / 2:.4f} 0 0 0</pose>
        <collision name="c"><geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material></visual>
      </link>"""


def _world_sdf(cfg: dict[str, Any]) -> str:
    g, ac, oc = cfg["gz"], cfg["arena"], cfg["obstacles"]
    half, t = ac["size"] / 2.0, ac["wall_thickness"]
    wall_h = 0.3
    c, ln = half + t / 2.0, ac["size"] + 2 * t
    walls = "".join([
        _box_link("wall_e", c, 0, t, ln, wall_h, "0.5 0.5 0.5 1"),
        _box_link("wall_w", -c, 0, t, ln, wall_h, "0.5 0.5 0.5 1"),
        _box_link("wall_n", 0, c, ln, t, wall_h, "0.5 0.5 0.5 1"),
        _box_link("wall_s", 0, -c, ln, t, wall_h, "0.5 0.5 0.5 1"),
    ])
    obstacles = []
    for slot in range(oc["max_count"]):
        px, py = park_pose(slot, cfg)
        if slot_kind(slot) == "cylinder":
            geom = f"<cylinder><radius>{oc['cylinder_radius']}</radius><length>0.4</length></cylinder>"
            z, colour = 0.2, "0.8 0.3 0.2 1"
        else:
            s = oc["box_size"]
            geom = f"<box><size>{s} {s} 0.4</size></box>"
            z, colour = 0.2, "0.2 0.4 0.8 1"
        obstacles.append(f"""
    <model name="obs_{slot}"><static>true</static><pose>{px} {py} {z} 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry>{geom}</geometry></collision>
        <visual name="v"><geometry>{geom}</geometry>
          <material><ambient>{colour}</ambient><diffuse>{colour}</diffuse></material></visual>
      </link>
    </model>""")
    rtf = g["real_time_factor"]
    # DART (gz-physics' default engine) collision detector; "" keeps the engine default (FCL). README section 6.
    dart = (f"\n      <dart><collision_detector>{g['dart_collision_detector']}</collision_detector></dart>"
            if g["dart_collision_detector"] else "")
    return f"""<?xml version="1.0"?>
<!-- GENERATED by envs/gz_world.py from config.yaml -- do not edit by hand -->
<sdf version="1.9">
  <world name="{g['world_name']}">
    <!-- same physics profile as turtlebot3_gazebo/worlds/turtlebot3_world.world -->
    <physics name="rl_physics" type="ode">
      <real_time_update_rate>{1.0 / g['max_step_size']:g}</real_time_update_rate>
      <max_step_size>{g['max_step_size']:g}</max_step_size>
      <real_time_factor>{rtf:g}</real_time_factor>
      <ode>
        <solver><type>quick</type><iters>150</iters><precon_iters>0</precon_iters><sor>1.4</sor>
          <use_dynamic_moi_rescaling>1</use_dynamic_moi_rescaling></solver>
        <constraints><cfm>0.00001</cfm><erp>0.2</erp><contact_max_correcting_vel>2000</contact_max_correcting_vel>
          <contact_surface_layer>0.01</contact_surface_layer></constraints>
      </ode>{dart}
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <gravity>0 0 -9.8</gravity>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows><pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground"><static>true</static>
      <link name="link">
        <!-- friction copied from the Fuel "Ground Plane" the official TurtleBot3 worlds use (mu=100, mu2=50);
             with the default mu=1 the Burger slips badly during hard turns -->
        <collision name="c"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>
          <surface><friction><ode><mu>100</mu><mu2>50</mu2></ode></friction></surface></collision>
        <visual name="v"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>
          <material><ambient>0.7 0.7 0.7 1</ambient><diffuse>0.7 0.7 0.7 1</diffuse></material></visual>
      </link>
    </model>

    <model name="walls"><static>true</static>{walls}
    </model>
{''.join(obstacles)}

    <!-- goal marker: visual only (no collision) so the LiDAR and robot ignore it -->
    <model name="goal_marker"><static>true</static><pose>0 0 {g['marker_z']} 0 0 0</pose>
      <link name="link">
        <visual name="v"><geometry><cylinder><radius>{cfg['task']['goal_radius']}</radius><length>0.02</length></cylinder></geometry>
          <material><ambient>0 1 0 1</ambient><diffuse>0 1 0 1</diffuse><emissive>0 0.6 0 1</emissive></material></visual>
      </link>
    </model>

    <include>
      <uri>model://{MODEL_NAME}</uri>
      <name>{g['robot_name']}</name>
      <pose>0 0 0.01 0 0 0</pose>
    </include>
  </world>
</sdf>
"""


def _bridge_yaml(cfg: dict[str, Any]) -> str:
    g = cfg["gz"]
    ros_type = {"TwistStamped": "geometry_msgs/msg/TwistStamped", "Twist": "geometry_msgs/msg/Twist"}[g["cmd_vel_type"]]
    name = lambda topic: topic.lstrip("/")   # noqa: E731
    return f"""# GENERATED by envs/gz_world.py -- same layout as turtlebot3_gazebo/params/turtlebot3_burger_bridge.yaml
- ros_topic_name: "clock"
  gz_topic_name: "clock"
  ros_type_name: "rosgraph_msgs/msg/Clock"
  gz_type_name: "gz.msgs.Clock"
  direction: GZ_TO_ROS
- ros_topic_name: "{name(g['odom_topic'])}"
  gz_topic_name: "odom"
  ros_type_name: "nav_msgs/msg/Odometry"
  gz_type_name: "gz.msgs.Odometry"
  direction: GZ_TO_ROS
- ros_topic_name: "{name(g['gt_odom_topic'])}"
  gz_topic_name: "{name(g['gt_odom_topic'])}"
  ros_type_name: "nav_msgs/msg/Odometry"
  gz_type_name: "gz.msgs.Odometry"
  direction: GZ_TO_ROS
- ros_topic_name: "{name(g['scan_topic'])}"
  gz_topic_name: "scan"
  ros_type_name: "sensor_msgs/msg/LaserScan"
  gz_type_name: "gz.msgs.LaserScan"
  direction: GZ_TO_ROS
- ros_topic_name: "{name(g['cmd_vel_topic'])}"
  gz_topic_name: "cmd_vel"
  ros_type_name: "{ros_type}"
  gz_type_name: "gz.msgs.Twist"
  direction: ROS_TO_GZ
"""


def generate(cfg: dict[str, Any], src_dir: Path) -> dict[str, Path]:
    out = (src_dir / cfg["gz"]["generated_dir"]).resolve()
    share = find_tb3_gazebo_share()
    model_dir = out / "models" / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.sdf").write_text(_patched_burger(cfg, share))
    (model_dir / "model.config").write_text(
        f'<?xml version="1.0"?>\n<model><name>{MODEL_NAME}</name><version>1.0</version>'
        f'<sdf version="1.9">model.sdf</sdf><description>Burger patched for RL</description></model>\n')
    world, bridge = out / "world.sdf", out / "bridge.yaml"
    world.write_text(_world_sdf(cfg))
    bridge.write_text(_bridge_yaml(cfg))
    return {"dir": out, "world": world, "bridge": bridge, "models": out / "models",
            "tb3_models": share / "models"}


if __name__ == "__main__":
    import argparse

    from utils import SRC_DIR, apply_overrides, load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL",
                    help="config override, e.g. --set gz.generated_dir=../gz_generated_test (start_gz.sh: GZ_WORLD_ARGS)")
    cfg = load_config(tier="gz")
    apply_overrides(cfg, ap.parse_args().overrides)
    paths = generate(cfg, SRC_DIR)
    print("\n".join(f"{k}={v}" for k, v in paths.items()))
    sys.exit(0)
