#!/usr/bin/env bash
# Final protocol (README section 6, "Evaluation protocol"), for DQN (default) or TD3 (AGENT=td3). Everything selected
# beforehand on the TUNING layouts; this script only trains the chosen configuration and reports on the held-out
# REPORT layouts.
#
#   ./run_final.sh train          # fast tier: SEEDS seeds x 200k steps, in parallel (1 thread each)
#   ./run_final.sh report-fast    # final.pt of every fast seed + random policy on the report layouts, fast tier
#   ./run_final.sh report-gz      # same in Gazebo (fast->gz transfer, gz-trained runs, random); starts its own sims
#   ./run_final.sh plots          # plot.py for both tiers (headline numbers = report-layout evaluations)
#
# AGENT (dqn | td3, default dqn), SEEDS (default "0 1 2 3 4 5 6 7 8 9"), GZ_SIMS (parallel sims for report-gz, default 5).
#   AGENT=td3 ./run_final.sh train    # etc.: run names, checkpoints, the random baseline (random_td3 = uniform over the
#                                     # continuous box) and the plot file names all follow AGENT
# Gazebo sims are isolated by GZ_PARTITION=tb3_r<i> / ROS_DOMAIN_ID=80+i (README section 2).
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="${PYTHON:-../venv/bin/python}"
AGENT="${AGENT:-dqn}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
GZ_SIMS="${GZ_SIMS:-5}"
LOG=../results/logs
mkdir -p "$LOG" ../results/report

ros_env() {   # $1 = sim index
  source /opt/ros/jazzy/setup.bash
  source "${TB3_WS:-$HOME/turtlebot3_ws}/install/setup.bash"
  export TURTLEBOT3_MODEL=burger GZ_PARTITION="tb3_r$1" ROS_DOMAIN_ID=$((80 + $1))
}

start_sim() {  # $1 = sim index; returns after the sim is up. The sim's PIDs go to $LOG/final_sim_$1.pids
  ( ros_env "$1"
    GZ_WORLD_ARGS="--set gz.generated_dir=../gz_generated_r$1" setsid ./start_gz.sh > "$LOG/final_sim_$1.log" 2>&1 &
    echo $! > "$LOG/final_sim_$1.pids" )
  sleep 15
}

stop_sim() {   # kill by PID (never pkill -f: the pattern would match this script's own command line)
  local root; root=$(cat "$LOG/final_sim_$1.pids")
  for p in $(ps -eo pid,pgid | awk -v g="$root" '$2 == g {print $1}'); do kill -9 "$p" 2>/dev/null || true; done
}

case "${1:-}" in
  train)
    for s in $SEEDS; do
      $PY train.py --agent "$AGENT" --env fast --seed "$s" > "$LOG/fast_${AGENT}_seed$s.log" 2>&1 &
    done
    wait ;;
  report-fast)
    ckpts=(); for s in $SEEDS; do ckpts+=("../results/runs/fast_${AGENT}_seed$s/final.pt"); done
    $PY evaluate.py --agent "$AGENT" --checkpoint "${ckpts[@]}" --env fast --layouts report --out-dir ../results/report
    $PY evaluate.py --agent "$AGENT" --random --env fast --layouts report --out-dir ../results/report ;;
  report-gz)
    jobs_=()
    for s in $SEEDS; do jobs_+=("../results/runs/fast_${AGENT}_seed$s/final.pt"); done
    for f in ../results/runs/gz_${AGENT}_seed*/final.pt; do [[ -f $f ]] && jobs_+=("$f"); done
    jobs_+=("random")
    for i in $(seq 0 $((GZ_SIMS - 1))); do start_sim "$i"; done
    for i in $(seq 0 $((GZ_SIMS - 1))); do      # job k runs on sim k mod GZ_SIMS, sequentially per sim
      ( ros_env "$i"
        for k in "${!jobs_[@]}"; do
          (( k % GZ_SIMS == i )) || continue
          j="${jobs_[$k]}"
          if [[ $j == random ]]; then $PY evaluate.py --agent "$AGENT" --random --env gz --layouts report --out-dir ../results/report
          else $PY evaluate.py --agent "$AGENT" --checkpoint "$j" --env gz --layouts report --out-dir ../results/report; fi
        done ) > "$LOG/final_report_gz_$i.log" 2>&1 &
    done
    wait
    for i in $(seq 0 $((GZ_SIMS - 1))); do stop_sim "$i"; done ;;
  plots)
    $PY plot.py --agent "$AGENT"
    $PY plot.py --agent "$AGENT" --env gz ;;
  *) sed -n '2,14p' "$0"; exit 1 ;;
esac
