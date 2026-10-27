# TurtleBot3 DQN — Gazebo (discrete actions)

A reproducible **vanilla DQN** experiment: a TurtleBot3 Burger learns, inside Gazebo Harmonic (`gz sim`), to drive to random
goals among obstacles using 24 LiDAR bins + goal geometry and 5 discrete steering actions. The env (`envs/env_gz.py`) is a
ROS 2 Jazzy node behind a Gymnasium interface; the agent, logging and plotting are agent-agnostic.

## Results at a glance

Success/collision/timeout are **greedy (ε=0) evaluations of the last checkpoint on 200 held-out layouts** that were never used
for any decision (§5); ± is the 95 % CI over seeds. Raw CSVs are in `results/`, plots in `results/*_gz.png`.

| Condition | Seeds | Success % | Collision % | Timeout % | Mean return |
|---|---|---|---|---|---|
| Gazebo-trained (30k steps), Gazebo eval | 3 | 60.2 ± 13.0 | 39.8 ± 13.0 | 0.0 | 76.6 |
| Random policy, Gazebo | - | 3.5 | 96.5 | 0.0 | -107.7 |

Per-seed success: 55.0 / 65.5 / 60.0 %. Details and caveats are in [Results & analysis](#5-results--analysis).

---

## 1. Install (WSL2 + Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic)

Target: Windows 11, WSL2, Ubuntu 24.04. The project must live on the Linux filesystem (`~/tb3_dqn`), **not** `/mnt/c`
(`setup.sh` refuses otherwise).

```bash
cd ~/tb3_dqn
bash setup.sh                 # idempotent; add --skip-apt (no sudo) and/or --no-bashrc if you prefer
source venv/bin/activate
```

`setup.sh` does, skipping whatever is already satisfied:

1. adds the ROS 2 apt source and installs `ros-jazzy-desktop ros-jazzy-ros-gz ros-jazzy-turtlebot3 ros-jazzy-turtlebot3-msgs`
   (+ colcon, rosdep, venv tooling);
2. clones `ROBOTIS-GIT/turtlebot3_simulations` **branch `jazzy`** into a colcon workspace (default `~/turtlebot3_ws`,
   override with `TB3_WS=`) and builds it from source;
3. creates `venv/` with `--system-site-packages` (so `rclpy` imports) and pip-installs PyTorch (CPU wheel index),
   numpy, pandas, matplotlib, pyyaml, gymnasium — *only if they are not already importable*;
4. appends an environment block to `~/.bashrc` (`TURTLEBOT3_MODEL=burger`, `GZ_SIM_RESOURCE_PATH`, `ROS_DOMAIN_ID`,
   workspace overlay);
5. verifies the install.

**Verified before relying on it** (Oct 2026): all four apt packages have installation candidates on noble/jazzy, and the
`jazzy` branch exists upstream. apt also ships binary `ros-jazzy-turtlebot3-gazebo`/`-simulations`; the brief asks for a
source build, so that is what `setup.sh` does (the generator in `envs/gz_world.py` reads the Burger SDF from the built package).

> **What was and was not exercised.** Developed on a machine that already had ROS/Gazebo/the workspace, running
> `setup.sh --skip-apt --no-bashrc` there. The apt-repo bootstrap, `rosdep`, the fresh `colcon build` and the `~/.bashrc`
> block were **not** exercised end-to-end on a clean machine.

To check that TurtleBot3 teleop works in Gazebo with the stock package:

```bash
ros2 launch turtlebot3_gazebo empty_world.launch.py     # GUI through WSLg
ros2 run turtlebot3_teleop teleop_keyboard              # Jazzy publishes TwistStamped on /cmd_vel
```

## 2. Running

All commands run from `src/` with the venv active.

```bash
./start_gz.sh                                                  # terminal 1: headless gz sim + ros_gz bridge
python train.py --agent dqn --env gz --seed 0 --smoke          # terminal 2: checks /scan /odom /cmd_vel, control period
python train.py --agent dqn --env gz --seed 0                  # 30k steps, ~2.5 h incl. evals (real time)

python evaluate.py --checkpoint ../results/runs/gz_dqn_seed0/final.pt --env gz
python evaluate.py --checkpoint <ckpt> --env gz --layouts report   # held-out report layouts (final numbers only)
python evaluate.py --random --env gz                           # random-policy baseline
python validate_gz.py                                          # sim-to-sim sanity checks (see §4)
python plot.py --env gz                                        # regenerate plots from the CSVs
```

* Every run prints a summary table at the end (`env tier | success % | collision % | timeout % | mean return`).
* Re-running a `train.py` command **resumes** from `ckpt_latest.pt` (written after every evaluation, includes the replay
  buffer); use `--fresh` to restart. After a WSL restart just run the same command again.
* `--set section.key=value` overrides any config value for a run; `--tag name` sends the run to `results/tuning/`.
* **Layout sets.** Periodic evaluations use the 50 *tuning* layouts (`eval.tuning_layout_seed`); the headline numbers use 200
  held-out *report* layouts (`eval.report_layout_seed`, `evaluate.py --layouts report`).
* **Gazebo variants.** `GZ_WORLD_ARGS="--set gz.generated_dir=../gz_generated_x --set key=val" ./start_gz.sh` generates a
  world from config overrides into its own directory.
* **Parallel Gazebo runs.** One sim cannot be shared between runs. For N concurrent runs start N isolated sims:
  `GZ_PARTITION=tb3_s$i ROS_DOMAIN_ID=$((50+i)) ./start_gz.sh` and use the same two variables for that run's `train.py`.
  That is how the three Gazebo seeds here were run (each sim held a real-time factor of 1.00).

### Outputs

```
results/runs/gz_dqn_seed<N>/   train_episodes.csv  eval.csv  final.pt  final_eval_episodes.csv  final_eval_traj.npz
                               ckpt_latest.pt(+.replay.npz) state.json config_snapshot.json env_info.json
results/report/                held-out report-layout evaluations (headline numbers) + per-episode CSVs
results/tuning/                hyper-parameter tuning runs (same format)
results/*_gz.png, summary_gz.csv   plots / summaries written by plot.py
```

`eval.csv` has, per evaluation (every 5k steps, 50 episodes, ε=0): success / collision / timeout rate, mean return,
mean steps-to-goal (successful episodes only), mean |Δw| (path smoothness), plus mean training-episode return since the
previous evaluation, ε, and the cumulative training-episode count. It also logs the interval mean and max of the TD loss,
Q(s,a), max_a Q(s,a), TD target and pre-clip gradient norm, for stability analysis. Nothing feeds back into learning.

## 3. Design choices

**Task.** Square 4 m arena, 3–6 obstacles (cylinders r=0.15 m and 0.30 m boxes), random start pose and goal
(1–3 m apart). Episode ends on goal (<0.2 m), collision (min LiDAR bin <0.13 m) or 500 steps. Control rate 10 Hz.
**Observation** (28-d): 24 LiDAR bins (min over a 15° sector of the 360-sample scan, `range/3.5`), distance to goal
(`/5.66`), heading to goal wrapped to [−π, π], last v and last w (normalised). **Actions** (5): v = 0.15 m/s constant,
w ∈ {+1.5, +0.75, 0, −0.75, −1.5} rad/s (hard-left … hard-right), all within the Burger limits (0–0.22 m/s, ±2.84 rad/s).
**Reward:** +100 goal, −100 collision, `50 × (previous − current goal distance)` progress shaping, −0.05 per step.
Everything is in `src/config.yaml`.

* *Progress shaping is potential-based* (difference of distance-to-goal), so circling or wall-hugging nets zero shaping and
  just pays the step penalty — the guard against reward hacking. Collision takes priority over goal if both trigger on one step.
* *Shared logic.* Action decoding, observation, reward, termination and the heading wrap live in `envs/common.py`; the env only
  implements "reset the world" and "advance one control period". The `ScanBinner` reduces 360 raw samples to 24 min-pooled bins.
* *Fixed evaluation layouts.* `eval.layout_seed` generates 50 layouts once; evaluation episode *i* uses layout *i*.

**DQN (`agents/dqn.py`)** — vanilla only. `y = r + γ·max_a' Q_target(s',a')·(1−done)`: the target net both selects and
evaluates (no Double/Dueling/PER/n-step). MLP 2×256 ReLU, Adam lr 5e-4, γ 0.99, batch 128, replay 100k, hard target copy every
1,000 steps, Huber loss, gradient-norm clip 10, ε linear 1.0→0.05 over 15k steps, 1k warm-up random steps. One gradient step per env step.

* `done` is **only** goal/collision. A 500-step *truncation* is not a terminal state (the observation has no clock), so it still
  bootstraps.
* The agent interface is `act / observe / update / save / load (+ info)` (`agents/base.py`); `train.py`, `evaluate.py` and
  `plot.py` only use that interface.

**Reproducibility.** Python, NumPy, torch and the env are seeded from `--seed`. Training is CPU-only and single-threaded per
process. Every run writes `env_info.json` (package versions, config hash, per-file source hashes). Gazebo itself is **not
deterministic** in free-running mode (timing depends on machine load), so results are reproducible only statistically.

### The Gazebo env

* **`/cmd_vel` message type.** The Jazzy TurtleBot3 bridge config
  (`turtlebot3_gazebo/params/turtlebot3_burger_bridge.yaml`) maps **`geometry_msgs/msg/TwistStamped`** ↔ `gz.msgs.Twist` (its
  own comment says "if you use Twist, you need to change the type"). `env_gz` therefore publishes `TwistStamped`;
  `gz.cmd_vel_type: Twist` switches both the publisher and the generated bridge. `--smoke` verifies that `/scan`, `/odom`
  publishers and a `/cmd_vel` subscriber exist, that messages actually arrive, and that the measured control period equals `1/rate`.
* **Generated assets.** `envs/gz_world.py` writes `gz_generated/{world.sdf,bridge.yaml,models/tb3_burger_rl}` from `config.yaml`
  (arena size, obstacle pool, goal marker, physics profile, cmd_vel type) and patches the stock Burger SDF so the LiDAR publishes
  at the control rate (stock: 5 Hz; at 10 Hz control that would give every other step a duplicate scan).
* **Pose source.** The env's pose (goal distance/heading, goal test) comes from `gz.pose_topic`: by default `/odom_gt`, the
  physics pose published by gz's `OdometryPublisher` (added to the generated model and bridged like `/odom`). `/odom` (wheel
  odometry) is still bridged and selectable, but it drifts badly (§4).
* **Physics.** The generated world sets DART's collision detector to `bullet` (`gz.dart_collision_detector`) and gives the
  Burger's empty `base_footprint` link a negligible inertial (`gz.fix_base_footprint_inertia`); see §4.
* **Timing.** One env step = the first `/scan` whose *simulation* stamp is ≥ previous + dt. The pose is matched to the scan stamp.
  Gazebo runs in real time, so ≈ 7–10 env steps/s.
* **Goal marker.** A visual-only green disc (no collision); visible with `./start_gz.sh --gui`.
* **Reset without teleporting the robot.** Each reset does a gz **world reset** (robot at the origin, fresh joints and
  odometry) and moves the walls, obstacles and goal by the rigid transform `inv(start)` so the robot sits at the layout's
  start pose in the arena frame (§4, bug 2).
* `gz.lockstep: true` pauses the sim and advances exactly one control period per step (`WorldControl.multi_step`). It was
  **not faster** here (4–7 steps/s vs 7–10); it is kept as an option for determinism. The reported runs use free-running mode.

## 4. Gazebo fidelity: what I checked, and four bugs it found

`python validate_gz.py` (needs `./start_gz.sh`) runs checks of the Gazebo env against ground truth; latest result:

| Check | Result |
|---|---|
| [1] Reset leaves the robot in collision | 0 / 50 layouts |
| [2] Reset-time LiDAR vs an analytic scan of the layout (50 layouts x 24 bins) | mean abs diff 0.015 m, p95 0.029 m (max 1.85 m: one bin on an obstacle edge) |
| [3] Uncommanded yaw after a hard turn (ground truth vs wheel odometry, 20 straight steps) | ≤ 0.18 rad (stock physics: up to 2.3 rad) |
| [4] Heading-only oracle, outcome agreement with an analytic reference | 9 / 10 layouts |

Four real problems were found *only* by checking ground truth against what the controller believes:

1. **Ground friction.** A bare ground plane (default μ = 1) let the Burger slip during hard turns: yaw rate reached only
   roughly 40–65 % of the commanded rate while `/odom` reported the full rate. The official worlds use the Fuel "Ground Plane"
   (μ = 100, μ₂ = 50); copying that, and the official physics profile, fixed it.
2. **`set_pose` teleport.** Teleporting the Burger with `set_pose` appeared to corrupt its dynamics. Resets now use the
   world-reset + move-the-world procedure (§3). Respawning the model (`remove`+`create`) was flaky (the new LiDAR sometimes never
   published). In hindsight the spinning was most likely bug 3; the procedure is kept because it is correct either way.
3. **Wheel contact in DART's default collision detector.** With the default (FCL) detector, and also ODE, the Burger loses
   wheel traction after a hard turn held for a few steps: the wheels go straight while the body keeps yawing at ~1.4 rad/s for
   1–2 s and then re-grips abruptly (up to 2.3 rad of uncommanded yaw in 20 steps). Setting
   `<dart><collision_detector>bullet</collision_detector></dart>` reduces it to ≤ 0.18 rad. The stock Burger SDF also declares
   `<link name="base_footprint"/>` without an inertial, so SDF's default (1 kg, I = 1 kg m²) is merged into the chassis; fixing
   it alone did not stop the spin-out but is kept as a correction.
4. **Wheel odometry as the pose.** The env originally used `/odom` for the goal geometry and success test. Because of bug 3,
   odometry yaw was off by 1–2 rad in many episodes. Scan matching the recorded LiDAR against the known layout showed that
   only 7 % of the stock-physics "goals" were real (within 0.2 m). The env now uses the ground-truth pose (`/odom_gt`).

Remaining known difference: a **47–56 ms effective command delay** in free-running Gazebo (16 ms in lockstep), measured with
`diag_dynamics.py --env gz`. The heading-only oracle still disagrees with the reference on 1 of 10 layouts (a corner graze).

## 5. Results & analysis

### Summary (vanilla DQN, ε = 0, final checkpoints, 200 held-out report layouts)

Every decision was made on the 50 tuning layouts; the numbers come from 200 **report** layouts (`eval.report_layout_seed`
987654), one episode per layout, using the **last** checkpoint of each run (never a best checkpoint). ± is the 95 % CI of the
mean over seeds (Student t).

| Condition | Seeds | Success % | Collision % | Timeout % | Mean return |
|---|---|---|---|---|---|
| Gazebo-trained (30k), Gazebo eval | 3 | 60.2 ± 13.0 | 39.8 ± 13.0 | 0.0 | 76.6 ± 34.6 |
| Random policy, Gazebo | - | 3.5 | 96.5 | 0.0 | -107.7 |

Per-seed success: 55.0 / 65.5 / 60.0. Files: `results/report/summary.csv` (+ per-episode CSVs), `results/summary_gz.csv`,
`results/*_gz.png`.

### Learning speed

Gazebo runs near real time (≈ 7–10 steps/s), so the budget is 30k steps (~2.5 h per seed). With ε decaying over 15k steps,
success on the tuning layouts reaches 48–72 % by 10–15k steps and 64 / 76 / 68 % at 30k (mean 69 %). On the held-out layouts the
same checkpoints score 60.2 %; the ~9-point drop is consistent with numbers on the tuning layouts being optimistic by selection.

### Final success and collisions

Failures are collisions, not timeouts (0 % timeouts), so the agent keeps driving toward the goal and fails by clipping
obstacles or walls; there is no sign of reward hacking (circling or wall-hugging would show up as timeouts). The random
baseline is 3.5 %. With 3 seeds the CI is wide (±13 points).

### Hyperparameters

`progress_weight = 50` was chosen from a sweep over 10–100 (success rose with the weight up to ≈ 50 and was flat at 100; lr,
MSE vs Huber, reward scale and LiDAR clip changes did not help). The sweep and its caveats are in this README's git history
and `results/tuning/`. Network size, batch, buffer, γ and the target-update interval are fixed by the brief.

### Caveats

* One task, one robot model, one reward; `progress_weight` was tuned on the tuning layouts.
* 3 seeds and 30k steps (real-time limited).
* Gazebo success uses the ground-truth pose, which a real robot would not have; on hardware, localisation error is an extra
  source of failure that is not modelled.
* Gazebo is non-deterministic in free-running mode; numbers are reproducible statistically only (s.e. ~3 points at 200 episodes).

## 6. WSL troubleshooting

* **Gazebo/LiDAR renders black, crashes with an OpenGL/ogre2 error, or `gz sim -s` exits at start-up.** The LiDAR is a
  `gpu_lidar` and needs an off-screen GL context even with `-s`. First try the software rasteriser:
  `export LIBGL_ALWAYS_SOFTWARE=1` before `./start_gz.sh`. You can also try `GZ_HEADLESS_RENDERING=1 ./start_gz.sh`
  (adds `--headless-rendering`). (On the machine used here neither was needed, with `/dev/dxg` present.)
* **Memory/CPU limits — `C:\Users\<you>\.wslconfig`:**
  ```ini
  [wsl2]
  memory=12GB        # leave headroom for Windows; each gz sim measured ~0.5 GB RSS here, each trainer ~0.7 GB
  processors=8
  swap=4GB
  ```
  then `wsl --shutdown` from PowerShell and reopen the distro. Without a cap WSL2 may take half the host RAM and starve Windows.
  Many concurrent sims can crash the VM; keep it to roughly 6.
* **`ROS_DOMAIN_ID` isolation.** `start_gz.sh` and the env use `ROS_DOMAIN_ID` (default 42 if unset). Use a distinct value
  per concurrent sim (and a distinct `GZ_PARTITION` — the gz transport has its own discovery, which `ROS_DOMAIN_ID` does not
  isolate). If `train.py --env gz` says *"Gazebo tier not connected"*, the first suspect is a domain mismatch between the two
  terminals — `echo $ROS_DOMAIN_ID` in both.
* **`no /scan newer than sim t=…`.** The sim is paused (e.g. a lockstep run was killed), the bridge is down, or
  `gz.lidar_rate_hz` ≠ `control.rate_hz` after editing `config.yaml` (restart `./start_gz.sh`).
* **After editing `config.yaml`, restart `./start_gz.sh`** — the world, bridge and patched model are regenerated at start-up.
* **Orphaned processes.** `gz sim` and `ros_gz_bridge` can outlive a killed terminal; find them with
  `ps -eo pid,args | grep -E "gz sim|parameter_bridge"` and `kill -9` them by PID before restarting (`pkill -f` with a pattern
  that also appears in your own command line will kill your shell).
* **Slow file access / weird permission errors** → the project is on `/mnt/c`; move it to `~`.

## 7. Known pitfalls (and how this repo addresses them)

| pitfall | status |
|---|---|
| heading-angle wrap bug | `wrap_angle` in `envs/common.py`; checked: goal behind the robot → ±π, goal to the left → +π/2 |
| phantom crashes / bad Gazebo resets | `validate_gz.py` [1]: 0 / 50 layouts start in collision; world-reset procedure (§4). Apparent phantom collisions were real collisions at poses the env did not know about (wheel slip + odometry pose, §4 bugs 3–4) |
| mismatched control rates | one `control.rate_hz`; Gazebo LiDAR patched to the control rate; `--smoke` measures the real period |
| reward hacking | potential-based shaping + step penalty; evaluation reports |Δw|, steps-to-goal and timeouts |
| Gazebo is slow | ≈ 7–10 steps/s; budget is 30k steps per seed, one isolated sim per concurrent run |

## 8. Repository layout

```
tb3_dqn/
├── README.md  setup.sh
├── results/                      runs/ report/ tuning/ logs/ + plots
├── gz_generated/                 (generated) world / bridge / patched model
└── src/
    ├── config.yaml               every tunable value
    ├── envs/  common.py layout.py env_gz.py gz_world.py
    ├── agents/ base.py dqn.py
    ├── train.py  evaluate.py  plot.py
    ├── start_gz.sh  validate_gz.py  diag_dynamics.py
    └── utils.py
```
