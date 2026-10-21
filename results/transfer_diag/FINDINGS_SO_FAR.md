> Superseded by report.md (final write-up). Kept as the working log of the diagnosis.

# Transfer-gap diagnosis: findings so far (working notes, 2026-10-19)

Written so a WSL restart cannot lose them. The final write-up goes into README section 6 and report.md.

1. **Recordings** (tuning layouts 12345, 3 fast checkpoints x 50 layouts, eps=0): `fast_*` here, stock-physics Gazebo in
   `gz_stock/`, fast with another LiDAR-noise seed in `fast_noise2/`. Analysis: `diagnose_transfer.py analyse --b <set>`
   -> `report_fast_vs_<set>.md`, `pairs_*.csv`, `paths_*.png`, `divergence_*.png`.
2. **Noise floor** (fast vs fast_noise2): 3% outcome flips, 2/150 pairs diverge > 5 cm. The policy is robust to LiDAR noise.
3. **Stock Gazebo**: 41% flips, 48 fast-goal/gz-collision vs 12 the other way (net 36 episodes = the gap).
   Gazebo LiDAR vs fast-tier LiDAR rendered at the gz *odom* pose: mean 0.56 m (noise floor 0.0085 m). At the gap
   collisions the odom pose is 0.51 m (median) from any obstacle although the LiDAR reads 0.125 m.
4. **Scan matching** (pose search that explains each gz scan, residual 0.008 m): the odom yaw is off by 1.7-2.1 rad after
   step ~20-40 in the gap episodes. Step by step: after a held hard turn the wheels go straight (joint_states) but the
   body keeps yawing at ~-1.4 rad/s, decaying ~0.45 rad/s^2, then stops abruptly -> wheel-ground contact is lost/slides.
5. **Ground truth (gz physics pose)**, hard-right x N then straight x 20, worst |true - odom| yaw change:
   FCL(default)+stock SDF 2.22 rad; FCL+base_footprint inertia fix 2.29; ODE detector 2.21; bullet+stock 0.16;
   bullet+fix 0.11. => cause = DART's default (FCL) / ODE collision detection of the wheel cylinders; bullet fixes it.
   Also fixed: stock `<link name="base_footprint"/>` has no inertial -> SDF default 1 kg, I = 1 kg m^2 (minor effect).
6. **Command delay**: gz free-running 56 ms (stock) / 47 ms (bullet), lockstep 16 ms, fast 0 ms (fit of per-step yaw
   increments, `diag_dynamics.py`). Injecting 0-90 ms delay in the fast tier does NOT change the old checkpoints'
   success (81-83%, `delay_sweep/`), so the delay is not the cause of the gap.
7. validate_gz with bullet: PASS (0/50 phantom resets, same-action divergence 0.010 m, coast 0.10 rad, oracle 8/10).
8. Code: evaluate.py --record-steps/--layouts/--set; diagnose_transfer.py; diag_dynamics.py; validate_gz check [6];
   env_fast act_delay_s/act_delay_jitter_s/act_lag_tau_s (default 0, bit-identical default path verified);
   gz.dart_collision_detector (bullet), gz.fix_base_footprint_inertia; train.py exact resume + env_info.json + upd_* logs.

Still to do: transfer with bullet (gz_bullet/), A3 delay candidate (tuning/a3_delay50_*), repro comparison
(repro/), gz-trained re-run (runs/gz_dqn_seed*), final 10-seed protocol (run_final.sh), README.
Stock-physics Gazebo results are archived in results/archive/gz_stock_physics/.

## Update 2026-10-19 ~11:00
9. Transfer with bullet physics, pose = wheel odometry (gz_bullet/): 78.0% vs fast 82.0% (stock 57.3%); flips 21%.
10. BUT Gazebo goal success was judged in the drifting odometry frame. Scan matching (`diagnose_transfer.py drift`,
    `logs/true_goal.py`): only 13% (bullet) / 7% (stock) of recorded gz "goals" are really within 0.2 m (fast control 93%);
    true final distance median 0.36 m. Odometry yaw drift: median max 0.41 rad (bullet), 0.55+ (stock), 0.005 (fast).
    => all earlier Gazebo success numbers (incl. 59.3% / 54.0% in the README) overstate real goal reaching.
11. Fix: gz OdometryPublisher (ground-truth physics pose) bridged as /odom_gt; `gz.pose_topic: /odom_gt` (default).
    validate_gz with it: [3] LiDAR at env pose 0.010 m (was 0.048), oracle 9/10, coast check (true vs wheel odom) 0.18 rad.
12. B3 reproducibility: fresh seed-0 run == original seed-0 run (old code) bit-for-bit on all shared columns (41 evals);
    runs resumed after real WSL crashes (b twice) bit-identical to uninterrupted, incl. final weights. (results/repro/)
13. Running: true-pose transfer gz_gt/ (bullet) and gz_stock_gt/ (stock physics); A3 tuning/a3_delay50_seed*.
