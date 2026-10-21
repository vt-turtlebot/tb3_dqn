#!/usr/bin/env bash
# Start the Gazebo tier backend: headless gz sim (Harmonic) + ros_gz bridge.
#
#   ./start_gz.sh            # headless (what training uses)
#   ./start_gz.sh --gui      # additionally open the gz GUI (WSLg) to watch; slower
#
# Leave it running in one terminal; run `python train.py --env gz ...` in another.
# Regenerates world/bridge/model from config.yaml on every start.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
TB3_WS="${TB3_WS:-$HOME/turtlebot3_ws}"
GUI=0; [[ "${1:-}" == "--gui" ]] && GUI=1

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
[[ -f "$TB3_WS/install/setup.bash" ]] && source "$TB3_WS/install/setup.bash"
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"          # isolate from other ROS 2 traffic on the LAN / WSL
PY="${PYTHON:-$([[ -x "$PROJECT/venv/bin/python" ]] && echo "$PROJECT/venv/bin/python" || echo python3)}"

# Generate the world, bridge yaml and patched Burger model from config.yaml.
# GZ_WORLD_ARGS: optional generator overrides for a variant sim, e.g.
#   GZ_WORLD_ARGS="--set gz.generated_dir=../gz_generated_v --set gz.dart_collision_detector=bullet"
# shellcheck disable=SC2086
GEN="$(cd "$HERE" && "$PY" -m envs.gz_world ${GZ_WORLD_ARGS:-})"
eval "$(echo "$GEN" | sed 's/^/GEN_/')"               # GEN_dir, GEN_world, GEN_bridge, GEN_models, GEN_tb3_models
export GZ_SIM_RESOURCE_PATH="${GEN_models}:${GEN_tb3_models}${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
echo "[start_gz] ROS_DOMAIN_ID=$ROS_DOMAIN_ID  world=$GEN_world"

EXTRA=()
[[ "${GZ_HEADLESS_RENDERING:-0}" == "1" ]] && EXTRA+=(--headless-rendering)

pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Server only (-s), running (-r): no GUI, no window. Needed even for LiDAR (rendered off-screen).
gz sim -s -r -v2 "${EXTRA[@]}" "$GEN_world" &
pids+=($!)
sleep 3
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:="$GEN_bridge" &
pids+=($!)
if [[ $GUI -eq 1 ]]; then gz sim -g -v2 & pids+=($!); fi

echo "[start_gz] up. Ctrl-C to stop."
wait -n "${pids[@]}"
