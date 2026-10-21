# fast vs gz_bullet: per-layout comparison (3 checkpoints x 50 layouts)

## 1. Outcome table

| cell | fast_dqn_seed0 | fast_dqn_seed1 | fast_dqn_seed2 | total |
|---|---|---|---|---|
| both goal | 33 | 39 | 32 | 104 |
| both collision | 7 | 3 | 4 | 14 |
| fast goal, gz_bullet collision | 7 | 3 | 9 | 19 |
| fast collision, gz_bullet goal | 3 | 5 | 5 | 13 |
| other (timeouts) | 0 | 0 | 0 | 0 |
| success fast / gz_bullet | 80 / 72% | 84 / 88% | 82 / 74% | 82.0 / 78.0% |

Outcome flips: 32 / 150 pairs (21%); net gap = 6 episodes.

## 2. Divergence onset (first step with xy distance > threshold)

| threshold | diverged pairs | median step | p25-p75 step | median clearance at onset (m) | gap pairs diverged |
|---|---|---|---|---|---|
| 0.05 m | 77/150 | 47 | 32-71 | 0.407 | 17/19 |
| 0.10 m | 27/150 | 72 | 47-92 | 0.317 | 9/19 |
| 0.20 m | 10/150 | 62 | 41-94 | 0.352 | 5/19 |

## 3. First action disagreement

Pairs with any disagreement: 150/150; median first step 10; xy distance between the rollouts at that step: median 0.013 m.

Q-margin (best - second best, fast) at the first disagreement: median 0.02, p75 0.03; over all fast steps: median 0.02, p25 0.01. Share of first disagreements with margin below the all-steps p25: 31% (25% if unrelated).

## 4. |obs_a - obs_b| at the first disagreement step (normalised obs units; LiDAR x 3.5 = metres)

| pairs | lidar_mean | lidar_max | dist | heading | v | w |
|---|---|---|---|---|---|---|
| all with a disagreement (n=150) | 0.1338 | 0.5405 | 0.0022 | 0.1259 | 0.0000 | 0.0000 |
| fast goal & gz_bullet collision (n=19) | 0.1338 | 0.5757 | 0.0023 | 0.1250 | 0.0000 | 0.0000 |

Values are medians. heading is in radians, dist in units of 5.66 m.

## Sensor consistency of gz_bullet: its LiDAR vs the noise-free fast-tier LiDAR rendered at its own pose

mean |diff| over all steps and bins: 0.3762 m; mean per-episode p99: 2.043 m. At the final step, (recorded min bin) - (rendered min bin): median -0.0686 m, collisions only -0.0892 m.

## 5. Collision geometry of the fast-goal & gz_bullet-collision episodes (n=19)

Object hit: cylinder 13, box 5, wall 1
Closest-bin direction at the collision: front-right 11, front-left 6, side/rear 1, front (+-22 deg) 1
Distance to goal at collision: median 1.07 m (p25-p75 0.79-1.50).
Geometric clearance of the lidar origin at the collision step (from pose + layout): median 0.249 m; recorded min range 0.124 m.
Layouts: 18 distinct; hit by all 3 checkpoints: []; by 2: [0].

