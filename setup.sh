#!/usr/bin/env bash
# One-shot, idempotent setup for WSL2 / Ubuntu 24.04 / ROS 2 Jazzy / Gazebo Harmonic.
#
#   bash setup.sh              # everything
#   bash setup.sh --skip-apt   # skip the apt stage (e.g. no sudo available)
#   bash setup.sh --no-bashrc  # do not append the environment block to ~/.bashrc
#
# Stages (each is skipped when already satisfied):
#   1. ROS 2 Jazzy apt repo + packages: ros-jazzy-desktop ros-jazzy-ros-gz ros-jazzy-turtlebot3
#      ros-jazzy-turtlebot3-msgs (+ colcon, rosdep, venv tooling)
#   2. colcon workspace with turtlebot3_simulations built FROM SOURCE (branch `jazzy`)
#   3. Python venv (--system-site-packages so rclpy imports) with torch (CPU), numpy, pandas,
#      matplotlib, pyyaml, gymnasium
#   4. Environment block in ~/.bashrc (TURTLEBOT3_MODEL, GZ_SIM_RESOURCE_PATH, ROS_DOMAIN_ID, ws overlay)
#   5. Verification
#
# Verified while writing this script (Oct 2026): all four apt packages exist for noble/jazzy, and
# https://github.com/ROBOTIS-GIT/turtlebot3_simulations has a `jazzy` branch (the Gazebo-Harmonic port).
# NOTE: apt also ships binary ros-jazzy-turtlebot3-gazebo / -simulations. We build from source as
# required so the world/model files can be inspected and patched (envs/gz_world.py reads them).
set -eo pipefail

SKIP_APT=0; NO_BASHRC=0
for a in "$@"; do
  case "$a" in --skip-apt) SKIP_APT=1;; --no-bashrc) NO_BASHRC=1;; *) echo "unknown arg $a" >&2; exit 2;; esac
done
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TB3_WS="${TB3_WS:-$HOME/turtlebot3_ws}"
SIM_BRANCH="${SIM_BRANCH:-jazzy}"
log() { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 0. sanity -------------------------------------------------------------------------------------
. /etc/os-release
[[ "$VERSION_ID" == "24.04" ]] || die "Ubuntu 24.04 required for ROS 2 Jazzy (found $VERSION_ID)"
case "$PROJECT" in /mnt/*) die "Project is on the Windows filesystem ($PROJECT). Move it to ~/tb3_dqn (ext4); /mnt/c is very slow and breaks file watching.";; esac
grep -qi microsoft /proc/version && log "WSL2 detected" || log "Not WSL (fine, continuing)"

# ---- 1. apt ----------------------------------------------------------------------------------------
APT_PKGS=(ros-jazzy-desktop ros-jazzy-ros-gz ros-jazzy-turtlebot3 ros-jazzy-turtlebot3-msgs
          python3-colcon-common-extensions python3-rosdep python3-venv python3-pip git curl)
missing=()
for p in "${APT_PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
if [[ $SKIP_APT -eq 0 && ${#missing[@]} -gt 0 ]]; then
  log "Installing apt packages: ${missing[*]}"
  if [[ ! -f /opt/ros/jazzy/setup.bash && ! -f /etc/apt/sources.list.d/ros2.sources ]]; then
    sudo apt-get update && sudo apt-get install -y software-properties-common curl
    sudo add-apt-repository -y universe
    ver="$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F tag_name | awk -F\" '{print $4}')"
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ver}/ros2-apt-source_${ver}.$(. /etc/os-release && echo "$VERSION_CODENAME")_all.deb"
    sudo dpkg -i /tmp/ros2-apt-source.deb
  fi
  sudo apt-get update
  for p in "${missing[@]}"; do apt-cache policy "$p" | grep -q 'Candidate: (none)' && die "apt package $p has no candidate"; done
  sudo apt-get install -y "${missing[@]}"
else
  log "apt stage: nothing to install (or --skip-apt)"
fi
[[ -f /opt/ros/jazzy/setup.bash ]] || die "/opt/ros/jazzy not found"
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash

# ---- 2. colcon workspace: turtlebot3_simulations from source ---------------------------------------
mkdir -p "$TB3_WS/src"
if [[ ! -d "$TB3_WS/src/turtlebot3_simulations" ]]; then
  log "Checking branch '$SIM_BRANCH' exists upstream"
  git ls-remote --exit-code --heads https://github.com/ROBOTIS-GIT/turtlebot3_simulations "$SIM_BRANCH" >/dev/null \
    || die "branch $SIM_BRANCH not found on ROBOTIS-GIT/turtlebot3_simulations (set SIM_BRANCH=...)"
  git clone -b "$SIM_BRANCH" https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git "$TB3_WS/src/turtlebot3_simulations"
fi
if [[ ! -f "$TB3_WS/install/turtlebot3_gazebo/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf" ]]; then
  log "Building turtlebot3_simulations (colcon)"
  [[ -f /etc/ros/rosdep/sources.list.d/20-default.list ]] || sudo rosdep init || true
  rosdep update || true
  (cd "$TB3_WS" && rosdep install --from-paths src/turtlebot3_simulations -yir --rosdistro jazzy || true)
  (cd "$TB3_WS" && colcon build --symlink-install --packages-select turtlebot3_gazebo turtlebot3_fake_node turtlebot3_simulations)
else
  log "workspace already built: $TB3_WS"
fi
# shellcheck disable=SC1091
source "$TB3_WS/install/setup.bash"

# ---- 3. Python venv --------------------------------------------------------------------------------
VENV="$PROJECT/venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  log "Creating venv (with --system-site-packages so rclpy is importable)"
  python3 -m venv --system-site-packages "$VENV"
fi
if ! "$VENV/bin/python" -c "import torch, gymnasium, pandas, matplotlib, yaml, numpy" 2>/dev/null; then
  log "Installing Python deps (PyTorch CPU wheel)"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu
  "$VENV/bin/pip" install numpy pandas matplotlib pyyaml gymnasium
else
  log "venv already has the Python deps"
fi

# ---- 4. environment block --------------------------------------------------------------------------
BLOCK_BEGIN="# >>> tb3_dqn >>>"; BLOCK_END="# <<< tb3_dqn <<<"
if [[ $NO_BASHRC -eq 0 ]] && ! grep -qF "$BLOCK_BEGIN" "$HOME/.bashrc"; then
  log "Adding environment block to ~/.bashrc"
  cat >> "$HOME/.bashrc" <<EOF
$BLOCK_BEGIN
source /opt/ros/jazzy/setup.bash
[ -f "$TB3_WS/install/setup.bash" ] && source "$TB3_WS/install/setup.bash"
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-42}
export GZ_SIM_RESOURCE_PATH="$TB3_WS/install/turtlebot3_gazebo/share/turtlebot3_gazebo/models\${GZ_SIM_RESOURCE_PATH:+:\$GZ_SIM_RESOURCE_PATH}"
$BLOCK_END
EOF
fi
export TURTLEBOT3_MODEL=burger

# ---- 5. verification -------------------------------------------------------------------------------
log "Verifying"
ros2 pkg prefix turtlebot3_gazebo
ros2 pkg prefix turtlebot3_msgs
ros2 pkg prefix ros_gz_sim
gz sim --versions | head -1
"$VENV/bin/python" - <<'EOF'
import rclpy, torch, gymnasium, numpy, pandas, matplotlib, yaml
print("rclpy OK | torch", torch.__version__, "| numpy", numpy.__version__, "| gymnasium", gymnasium.__version__)
EOF
cat <<EOF

[setup] done. Next:
  source $VENV/bin/activate && cd $PROJECT/src
  python train.py --agent dqn --env fast --seed 0 --smoke
  ./start_gz.sh               # in a second terminal, for the Gazebo tier
To check that TurtleBot3 teleop works in Gazebo (build-order step 1):
  ros2 launch turtlebot3_gazebo empty_world.launch.py     # GUI via WSLg
  ros2 run turtlebot3_teleop teleop_keyboard              # (TwistStamped on /cmd_vel in Jazzy)
EOF
