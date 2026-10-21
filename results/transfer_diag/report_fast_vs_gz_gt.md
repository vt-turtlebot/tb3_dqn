# fast vs gz_gt: per-layout comparison (3 checkpoints x 50 layouts)

## 1. Outcome table

| cell | fast_dqn_seed0 | fast_dqn_seed1 | fast_dqn_seed2 | total |
|---|---|---|---|---|
| both goal | 38 | 40 | 40 | 118 |
| both collision | 9 | 7 | 9 | 25 |
| fast goal, gz_gt collision | 2 | 2 | 1 | 5 |
| fast collision, gz_gt goal | 1 | 1 | 0 | 2 |
| other (timeouts) | 0 | 0 | 0 | 0 |
| success fast / gz_gt | 80 / 78% | 84 / 82% | 82 / 80% | 82.0 / 80.0% |

Outcome flips: 7 / 150 pairs (5%); net gap = 3 episodes.

## 2. Divergence onset (first step with xy distance > threshold)

| threshold | diverged pairs | median step | p25-p75 step | median clearance at onset (m) | gap pairs diverged |
|---|---|---|---|---|---|
| 0.05 m | 102/150 | 20 | 13-44 | 0.547 | 2/5 |
| 0.10 m | 11/150 | 31 | 22-50 | 0.425 | 0/5 |
| 0.20 m | 2/150 | 34 | 25-44 | 0.348 | 0/5 |

## 3. First action disagreement

Pairs with any disagreement: 150/150; median first step 9; xy distance between the rollouts at that step: median 0.033 m.

Q-margin (best - second best, fast) at the first disagreement: median 0.02, p75 0.03; over all fast steps: median 0.02, p25 0.01. Share of first disagreements with margin below the all-steps p25: 30% (25% if unrelated).

## 4. |obs_a - obs_b| at the first disagreement step (normalised obs units; LiDAR x 3.5 = metres)

| pairs | lidar_mean | lidar_max | dist | heading | v | w |
|---|---|---|---|---|---|---|
| all with a disagreement (n=150) | 0.1372 | 0.5407 | 0.0054 | 0.3535 | 0.0000 | 0.0000 |
| fast goal & gz_gt collision (n=5) | 0.1471 | 0.5908 | 0.0031 | 0.2881 | 0.0000 | 0.0000 |

Values are medians. heading is in radians, dist in units of 5.66 m.

## Sensor consistency of gz_gt: its LiDAR vs the noise-free fast-tier LiDAR rendered at its own pose

mean |diff| over all steps and bins: 0.0141 m; mean per-episode p99: 0.048 m. At the final step, (recorded min bin) - (rendered min bin): median -0.0133 m, collisions only -0.0195 m.

## 5. Collision geometry of the fast-goal & gz_gt-collision episodes (n=5)

Object hit: cylinder 4, box 1
Closest-bin direction at the collision: front-right 2, front-left 2, side/rear 1
Distance to goal at collision: median 1.42 m (p25-p75 1.05-1.42).
Geometric clearance of the lidar origin at the collision step (from pose + layout): median 0.150 m; recorded min range 0.123 m.
Layouts: 4 distinct; hit by all 3 checkpoints: []; by 2: [12].

