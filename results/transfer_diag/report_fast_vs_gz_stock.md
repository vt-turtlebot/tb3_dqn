# fast vs gz: per-layout comparison (3 checkpoints x 50 layouts)

## 1. Outcome table

| cell | fast_dqn_seed0 | fast_dqn_seed1 | fast_dqn_seed2 | total |
|---|---|---|---|---|
| both goal | 26 | 27 | 21 | 74 |
| both collision | 6 | 5 | 4 | 15 |
| fast goal, gz collision | 14 | 14 | 20 | 48 |
| fast collision, gz goal | 4 | 3 | 5 | 12 |
| other (timeouts) | 0 | 1 | 0 | 1 |
| success fast / gz | 80 / 60% | 84 / 60% | 82 / 52% | 82.0 / 57.3% |

Outcome flips: 61 / 150 pairs (41%); net gap = 36 episodes.

## 2. Divergence onset (first step with xy distance > threshold)

| threshold | diverged pairs | median step | p25-p75 step | median clearance at onset (m) | gap pairs diverged |
|---|---|---|---|---|---|
| 0.05 m | 99/150 | 51 | 38-70 | 0.471 | 46/48 |
| 0.10 m | 56/150 | 72 | 53-96 | 0.470 | 34/48 |
| 0.20 m | 23/150 | 89 | 74-118 | 0.542 | 19/48 |

## 3. First action disagreement

Pairs with any disagreement: 150/150; median first step 10; xy distance between the rollouts at that step: median 0.008 m.

Q-margin (best - second best, fast) at the first disagreement: median 0.02, p75 0.03; over all fast steps: median 0.02, p25 0.01. Share of first disagreements with margin below the all-steps p25: 37% (25% if unrelated).

## 4. |obs_a - obs_b| at the first disagreement step (normalised obs units; LiDAR x 3.5 = metres)

| pairs | lidar_mean | lidar_max | dist | heading | v | w |
|---|---|---|---|---|---|---|
| all with a disagreement (n=150) | 0.0650 | 0.4263 | 0.0014 | 0.0779 | 0.0000 | 0.0000 |
| fast goal & gz collision (n=48) | 0.0656 | 0.4503 | 0.0014 | 0.0782 | 0.0000 | 0.0000 |

Values are medians. heading is in radians, dist in units of 5.66 m.

## Sensor consistency of gz: its LiDAR vs the noise-free fast-tier LiDAR rendered at its own pose

mean |diff| over all steps and bins: 0.5599 m; mean per-episode p99: 2.200 m. At the final step, (recorded min bin) - (rendered min bin): median -0.1246 m, collisions only -0.3359 m.

## 5. Collision geometry of the fast-goal & gz-collision episodes (n=48)

Object hit: cylinder 21, box 20, wall 7
Closest-bin direction at the collision: front-right 27, front-left 17, front (+-22 deg) 4
Distance to goal at collision: median 0.72 m (p25-p75 0.46-1.15).
Geometric clearance of the lidar origin at the collision step (from pose + layout): median 0.507 m; recorded min range 0.125 m.
Layouts: 30 distinct; hit by all 3 checkpoints: [16, 5, 7]; by 2: [0, 40, 39, 33, 32, 27, 2, 20, 26, 12, 11, 13].

