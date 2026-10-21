# Fast-to-Gazebo transfer gap: diagnosis

Executed as specified in `TRANSFER_GAP_TEST.md`, plus three additions that turned out to be decisive: a
**noise-floor control** (fast vs fast with a different LiDAR-noise seed), **scan matching** (the pose that explains each
recorded Gazebo scan, against the known layout), and **ground-truth checks** against the Gazebo physics pose.

All recordings: the three fast-trained checkpoints `results/runs/fast_dqn_seed{0,1,2}/final.pt` (unchanged, not retrained),
50 *tuning* layouts (`eval.tuning_layout_seed` 12345), eps = 0, LiDAR noise on in both tiers.
Reproduce with `evaluate.py --record-steps`, `diagnose_transfer.py {analyse,drift,replay}`, `diag_dynamics.py`,
`validate_gz.py` (commands at the end). Observation is 28-d (the test plan says 29; the code has always been 28).

## Summary

The 23-point gap was **not** a policy-robustness problem. It came from two defects in the Gazebo tier, both now fixed:

1. **Wheel-ground contact in DART's default collision detector (FCL, also ODE).** After a hard turn held for a few
   steps, the Burger's wheels go straight (as commanded, per `joint_states`) but the body keeps yawing at ~1.4 rad/s for
   1-2 s and then re-grips abruptly. Ground truth (gz physics pose): up to **2.3 rad** of uncommanded yaw in 20
   steps. Switching DART's collision detector to **bullet** removes it (worst 0.11-0.18 rad). A second, minor stock-model
   bug was fixed on the way: `<link name="base_footprint"/>` has no inertial, so SDF's default (1 kg, I = 1 kg m^2) is
   merged into the chassis.
2. **The Gazebo env used wheel odometry as its pose**, for the goal distance/heading observation *and* for the
   success test. Odometry cannot see the slip: its yaw was off by up to 1-2 rad in the stock sim and still by a median
   0.41 rad per episode with bullet. Scan matching shows that **only 7% (stock) / 13% (bullet) of the Gazebo "goals" were
   real** (within 0.2 m of the goal); every Gazebo success number reported before this diagnosis is affected. The env now
   uses a ground-truth pose topic (`/odom_gt`, gz `OdometryPublisher`), matching the fast tier, which has always used
   the true pose.

| Gazebo condition (3 ckpts x 50 layouts) | success | collision | outcome flips vs fast | net gap (episodes) | real goals |
|---|---|---|---|---|---|
| stock physics, wheel-odometry pose (as before) | 57.3% | 42.7% | 41% | 36 | 7% |
| stock physics, true pose | 74.0% | 26.0% | 17% | 12 | (true pose) |
| bullet physics, wheel-odometry pose | 78.0% | 22.0% | 21% | 6 | 13% |
| **bullet physics, true pose (new default)** | **80.0%** | **20.0%** | **5%** | **3** | 81%* |
| *control:* fast vs fast, other LiDAR-noise seed | 81.3% | 18.7% | 3% | 1 | 93%* |
| fast tier (reference) | 82.0% | 18.0% | - | - | - |

\* scan-matched final distance < 0.2 m. Goals trigger at 0.2 m, so true final distances sit at 0.19-0.20 m and the
matcher's ~1 cm error moves some just over the line (median 0.194 m for both rows); these two rows are equivalent.

The remaining gap (82.0 -> 80.0, 3 net episodes of 150) is the size of the noise floor (fast vs fast: 3% flips).
With 3 checkpoints x 50 episodes it is not distinguishable from zero. **Confirmed on the held-out protocol** (10 seeds x 200
report layouts, README section 6): fast 73.7%, Gazebo 72.0%, paired gap 1.6 +- 1.6 points, 5.9% of outcomes differ.

## 1. Outcome table (stock Gazebo, as originally reported)

| cell | seed0 | seed1 | seed2 | total |
|---|---|---|---|---|
| both goal | 26 | 27 | 21 | 74 |
| both collision | 6 | 5 | 4 | 15 |
| **fast goal, gz collision** | 14 | 14 | 20 | **48** |
| fast collision, gz goal | 4 | 3 | 5 | 12 |
| timeout | 0 | 1 | 0 | 1 |

After both fixes the "fast goal, gz collision" cell is 5 (2/2/1) and the reverse cell 2. Full tables per condition:
`report_fast_vs_<set>.md`; per-pair data `pairs_fast_vs_<set>.csv`.

## 2. Divergence onset

| condition | pairs diverging > 5 cm | > 20 cm | median step (> 20 cm) |
|---|---|---|---|
| fast vs fast (noise control) | 2/150 | 0/150 | - |
| stock gz | 99/150 | 23/150 | 89 |
| stock gz, true pose | 140/150 | 53/150 | 30 |
| bullet gz, true pose | 102/150 | 2/150 | 34 |

Small (5 cm) separations are common in every Gazebo condition (the 47-56 ms command delay, below), but they stay small
once the physics is fixed. Plots: `divergence_fast_vs_<set>.png`, `paths_fast_vs_<set>.png` (overlaid paths on gap
layouts with the divergence point and first action disagreement marked).

## 3. First action disagreement and Q-margins

Every pair disagrees on some action, typically by step ~10 and while the rollouts are < 1-3 cm apart. The Q-margin
(best minus second-best Q) is tiny for this policy everywhere (median 0.02 over all fast-tier steps), and first
disagreements are only mildly concentrated at small margins (30-37% below the all-steps 25th percentile vs 25% if
unrelated). So the argmax flips easily, but the noise control shows those flips are almost always harmless (3% outcome
flips). **The policy is not brittle in a way that explains the gap**, so domain randomisation (plan item A3.3) was not
pursued.

## 4. Observation difference at the first disagreement (stock Gazebo)

Median |obs_fast - obs_gz| at the first disagreement: heading 0.078 rad, LiDAR mean 0.065 (x 3.5 m = 0.23 m),
goal distance 0.0014 (x 5.66 m = 8 mm), last v/w 0. The heading difference matches one control step of command delay
at |w| = 1.5 rad/s (1.5 x 0.056 s = 0.084 rad), and the LiDAR difference is the same rotation re-binned. This is the
command delay, which (section 7) does not change outcomes.

## 5. Collision geometry (stock Gazebo, fast-goal & gz-collision, n = 48)

Objects hit: 21 cylinders, 20 boxes, 7 walls; closest bin front-right 27, front-left 17, straight ahead 4; median
distance to goal at the collision 0.72 m; 30 distinct layouts (all three checkpoints fail on 5, 7, 16).
**At the collision step the odometry pose puts the LiDAR 0.51 m (median) from the nearest object while the scan reads
0.125 m.** Rendering the fast-tier LiDAR at the Gazebo env pose differs from the real Gazebo scan by 0.56 m on average
(noise control: 0.0085 m). The robot was not where the env believed it to be. With the true pose this is 0.014 m.

## 6. Counterfactual open-loop replay

| replay | stock physics | bullet physics |
|---|---|---|
| Gazebo's colliding action sequence, replayed in fast | 10/48 collide (38 do not) | 3/5 collide |
| fast's goal-reaching sequence, replayed in Gazebo (true pose) | 11/19 collide | 2/5 collide |

With stock physics most gap cases are dynamics: the same commands are safe in the 2D sim and crash in Gazebo. With the
fix, the few remaining cases are marginal paths that also fail in the fast tier. (Open-loop replays accumulate the
command-delay difference over 100+ steps, so some open-loop divergence is expected even with matching physics.)
Files: `replay_*.csv` / `*_traj.npz`.

## 7. Dynamics and geometry (plan item A2)

* **Command delay.** Least-squares fit of per-step yaw increments on (w_k, w_k-1, w_k-2) (`diag_dynamics.py`):
  fast 0 ms; Gazebo free-running **56 ms** (stock) / **47 ms** (bullet), a pure delay (w_k-2 gain ~0.01); Gazebo
  lockstep 16 ms. So ~40 ms is wall-clock pipeline latency (scan -> bridge -> Python -> bridge -> DiffDrive) and ~16 ms is
  intrinsic. Linear velocity matches (both tiers: 1 m/s^2 ramp, then exact). **Injecting 0-90 ms of delay into the fast
  tier does not change the old checkpoints' success (81-83%, `delay_sweep/`)**, so the delay is not the cause of the gap.
* **Geometry.** Collision is the same rule in both tiers (min LiDAR bin < 0.13 m) on the same LiDAR model (360 samples,
  0.12-3.5 m, 3.2 cm offset; Gazebo below-min-range returns map to 0.12 m in the shared `ScanBinner`). The Burger's
  collision box (0.14 m, centred under the LiDAR; corner 0.10 m from the sensor) and wheels (0.11 m) lie inside the 0.13 m
  threshold, so an episode always ends before physical contact. No mismatch was found, so plan item A3.2 was not needed.

## Caveats

* 3 checkpoints x 50 layouts per condition; layout-level statements are on 50 layouts.
* The scan-matching "true pose" is a local search against the known layout; its residual (0.008-0.009 m) equals the
  LiDAR noise floor except for stock-physics collisions (0.36 m), where it loses track, so the stock drift numbers are
  lower bounds.
* Bullet does not remove all slip: ~0.15-0.2 rad of uncommanded yaw remains after a long hard turn (validate_gz check 6).
  With the true pose this no longer corrupts the observation, but it is a residual dynamics difference.
* The fixes change the Gazebo tier, so all Gazebo numbers were re-measured (old ones archived in
  `results/archive/gz_stock_physics/`).

## Commands

```bash
python evaluate.py --checkpoint ../results/runs/fast_dqn_seed{0,1,2}/final.pt --env fast --record-steps --out-dir ../results/transfer_diag
python evaluate.py --checkpoint ../results/runs/fast_dqn_seed0/final.pt --env gz --record-steps --out-dir ../results/transfer_diag/gz_gt   # one sim per checkpoint
python diagnose_transfer.py analyse --b gz_gt            # also: fast_noise2, gz_stock, gz_bullet, gz_stock_gt
python diagnose_transfer.py drift --set gz_stock
python diagnose_transfer.py replay --src gz_stock --env fast --pairs pairs_fast_vs_gz_stock.csv --tag from_gz_stock
python diag_dynamics.py --env gz [--set gz.lockstep=true]
python validate_gz.py
```
Stock-physics sims for the before/after rows: `GZ_WORLD_ARGS='--set gz.dart_collision_detector="" --set gz.fix_base_footprint_inertia=false' ./start_gz.sh`
plus `--set gz.pose_topic=/odom` on the evaluate.py side for the odometry-pose rows.
