# fast vs gz_stock_gt: per-layout comparison (3 checkpoints x 50 layouts)

## 1. Outcome table

| cell | fast_dqn_seed0 | fast_dqn_seed1 | fast_dqn_seed2 | total |
|---|---|---|---|---|
| both goal | 34 | 34 | 36 | 104 |
| both collision | 8 | 5 | 7 | 20 |
| fast goal, gz_stock_gt collision | 6 | 8 | 5 | 19 |
| fast collision, gz_stock_gt goal | 2 | 3 | 2 | 7 |
| other (timeouts) | 0 | 0 | 0 | 0 |
| success fast / gz_stock_gt | 80 / 72% | 84 / 74% | 82 / 76% | 82.0 / 74.0% |

Outcome flips: 26 / 150 pairs (17%); net gap = 12 episodes.

## 2. Divergence onset (first step with xy distance > threshold)

| threshold | diverged pairs | median step | p25-p75 step | median clearance at onset (m) | gap pairs diverged |
|---|---|---|---|---|---|
| 0.05 m | 140/150 | 21 | 17-27 | 0.563 | 19/19 |
| 0.10 m | 75/150 | 24 | 22-30 | 0.501 | 17/19 |
| 0.20 m | 53/150 | 30 | 28-35 | 0.489 | 14/19 |

## 3. First action disagreement

Pairs with any disagreement: 150/150; median first step 9; xy distance between the rollouts at that step: median 0.027 m.

Q-margin (best - second best, fast) at the first disagreement: median 0.02, p75 0.03; over all fast steps: median 0.02, p25 0.01. Share of first disagreements with margin below the all-steps p25: 31% (25% if unrelated).

## 4. |obs_a - obs_b| at the first disagreement step (normalised obs units; LiDAR x 3.5 = metres)

| pairs | lidar_mean | lidar_max | dist | heading | v | w |
|---|---|---|---|---|---|---|
| all with a disagreement (n=150) | 0.1011 | 0.5082 | 0.0044 | 0.1979 | 0.0000 | 0.0000 |
| fast goal & gz_stock_gt collision (n=19) | 0.1045 | 0.5418 | 0.0041 | 0.1733 | 0.0000 | 0.0000 |

Values are medians. heading is in radians, dist in units of 5.66 m.

## Sensor consistency of gz_stock_gt: its LiDAR vs the noise-free fast-tier LiDAR rendered at its own pose

mean |diff| over all steps and bins: 0.0136 m; mean per-episode p99: 0.034 m. At the final step, (recorded min bin) - (rendered min bin): median -0.0133 m, collisions only -0.0179 m.

## 5. Collision geometry of the fast-goal & gz_stock_gt-collision episodes (n=19)

Object hit: cylinder 11, box 8
Closest-bin direction at the collision: front-right 12, front-left 5, side/rear 1, front (+-22 deg) 1
Distance to goal at collision: median 1.32 m (p25-p75 1.01-1.65).
Geometric clearance of the lidar origin at the collision step (from pose + layout): median 0.142 m; recorded min range 0.122 m.
Layouts: 11 distinct; hit by all 3 checkpoints: [5, 12, 28]; by 2: [16, 45].

