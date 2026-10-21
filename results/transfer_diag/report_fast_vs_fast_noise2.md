# fast vs fast_noise2: per-layout comparison (3 checkpoints x 50 layouts)

## 1. Outcome table

| cell | fast_dqn_seed0 | fast_dqn_seed1 | fast_dqn_seed2 | total |
|---|---|---|---|---|
| both goal | 39 | 41 | 40 | 120 |
| both collision | 9 | 7 | 9 | 25 |
| fast goal, fast_noise2 collision | 1 | 1 | 1 | 3 |
| fast collision, fast_noise2 goal | 1 | 1 | 0 | 2 |
| other (timeouts) | 0 | 0 | 0 | 0 |
| success fast / fast_noise2 | 80 / 80% | 84 / 84% | 82 / 80% | 82.0 / 81.3% |

Outcome flips: 5 / 150 pairs (3%); net gap = 1 episodes.

## 2. Divergence onset (first step with xy distance > threshold)

| threshold | diverged pairs | median step | p25-p75 step | median clearance at onset (m) | gap pairs diverged |
|---|---|---|---|---|---|
| 0.05 m | 2/150 | 27 | 21-33 | 0.579 | 0/3 |
| 0.10 m | 2/150 | 33 | 26-40 | 0.502 | 0/3 |
| 0.20 m | 0/150 | nan | nan-nan | nan | 0/3 |

## 3. First action disagreement

Pairs with any disagreement: 108/150; median first step 38; xy distance between the rollouts at that step: median 0.000 m.

Q-margin (best - second best, fast) at the first disagreement: median 0.00, p75 0.00; over all fast steps: median 0.02, p25 0.01. Share of first disagreements with margin below the all-steps p25: 92% (25% if unrelated).

## 4. |obs_a - obs_b| at the first disagreement step (normalised obs units; LiDAR x 3.5 = metres)

| pairs | lidar_mean | lidar_max | dist | heading | v | w |
|---|---|---|---|---|---|---|
| all with a disagreement (n=108) | 0.0027 | 0.0074 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fast goal & fast_noise2 collision (n=3) | 0.0024 | 0.0074 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Values are medians. heading is in radians, dist in units of 5.66 m.

## Sensor consistency of fast_noise2: its LiDAR vs the noise-free fast-tier LiDAR rendered at its own pose

mean |diff| over all steps and bins: 0.0085 m; mean per-episode p99: 0.026 m. At the final step, (recorded min bin) - (rendered min bin): median -0.0133 m, collisions only -0.0201 m.

## 5. Collision geometry of the fast-goal & fast_noise2-collision episodes (n=3)

Object hit: cylinder 2, box 1
Closest-bin direction at the collision: front-right 3
Distance to goal at collision: median 0.58 m (p25-p75 0.51-1.11).
Geometric clearance of the lidar origin at the collision step (from pose + layout): median 0.151 m; recorded min range 0.129 m.
Layouts: 3 distinct; hit by all 3 checkpoints: []; by 2: [].

