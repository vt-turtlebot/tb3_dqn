# Understanding DQN with the TurtleBot3 in this project

A guided tour of how vanilla DQN works, mapped onto the code in `src/`. Read top to bottom once, then use it as a lookup.
Every number below comes from `src/config.yaml`; every code pointer is a real file in this repo.

---

## 1. The problem in one paragraph

A TurtleBot3 Burger sits in a 4 m x 4 m arena with 3-6 obstacles. A goal is placed somewhere. The robot must drive to
the goal (< 0.2 m) without touching anything (collision = a LiDAR reading < 0.13 m) within 500 steps. It decides what to do
10 times per second. It is never told a map: it only sees LiDAR distances and where the goal is relative to itself.

Reinforcement learning (RL) frames this as a loop:

```
          action a_t (turn left / straight / ...)
   +-----------------------------------------------+
   |                                               v
 Agent (the Q-network)                         Environment (fast sim or Gazebo)
   ^                                               |
   +-----------------------------------------------+
        observation s_{t+1}, reward r_t, done
```

The agent's job: pick actions that maximise the **total discounted reward** over an episode.

---

## 2. The pieces (state, action, reward)

### State (what the robot sees): 28 numbers
Built in `src/envs/common.py` (`obs_dim = n_bins + 4`).

| Part | Count | Meaning |
|---|---|---|
| LiDAR bins | 24 | Closest obstacle in each 15 degree sector, divided by 3.5 m, so 0 = touching, 1 = nothing in range |
| Distance to goal | 1 | Divided by 5.66 (arena diagonal) |
| Heading to goal | 1 | Angle between where the robot faces and the goal, wrapped to [-pi, pi]. 0 = goal dead ahead |
| Last v | 1 | Previous linear speed command |
| Last w | 1 | Previous angular speed command |

(Some older notes say 29 dims; the code gives 24 + 4 = 28. Check `observation_space` if it matters.)

The robot has no memory beyond "last v and w". It decides from a single snapshot.

### Actions (what it can do): 5 discrete choices
Constant forward speed `v = 0.15 m/s`, and one of five turn rates `w`:

| Index | Action | w (rad/s) |
|---|---|---|
| 0 | hard left | +1.5 |
| 1 | left | +0.75 |
| 2 | forward | 0 |
| 3 | right | -0.75 |
| 4 | hard right | -1.5 |

The robot never stops or reverses. "Discrete" is exactly why DQN works here: DQN needs a finite list of actions it can score.
Phase 2 (TD3) will instead output a continuous `(v, w)`.

### Reward (what "good" means)
From `reward:` in the config:

| Event | Reward |
|---|---|
| Reach goal | +100 |
| Collide | -100 |
| Every step | -0.05 (nudges it to hurry) |
| Progress | `50 * (previous_distance - current_distance)` |

The progress term is **shaping**: without it the only signal is at the very end of a long episode (sparse reward) and
learning is slow. It is potential-based, so driving in circles nets zero: you cannot farm it. Tuning this weight
(10 to 50) was the one change that fixed the unstable first run.

---

## 3. The core idea: the Q-function

**Q(s, a)** = "if I'm in state `s`, take action `a` now, and act well afterwards, how much total future reward do I expect?"

If you knew Q exactly, acting would be trivial: in each state, pick `argmax_a Q(s, a)`.

For this robot, the network takes the 28 numbers and outputs 5 numbers, one Q-value per action:

```
28 inputs --> [256] --ReLU--> [256] --ReLU--> 5 outputs (Q for each of the 5 turns)
```

That is `QNetwork` in `src/agents/dqn.py`. A "good" network, for a state with an obstacle just ahead-left,
outputs high Q for "right" and low Q for "left".

### Discounting
Future reward is worth slightly less: `gamma = 0.99`. A reward 100 steps away counts as 0.99^100 ~ 0.37 of its face value.
This is what lets the agent trade a short detour for a safe path.

---

## 4. How the network learns: the Bellman target

We don't know the true Q. We learn it by making it **self-consistent**. If the robot takes action `a` in state `s`,
gets reward `r`, and lands in `s'`, then Q should satisfy:

```
Q(s, a)  ~=  r + gamma * max_a' Q(s', a')          (and just r if the episode ended)
```

The right-hand side is the **target** `y`. Training minimises the gap between `Q(s, a)` and `y`.
This is the key line, in `DQNAgent.update` (`src/agents/dqn.py`):

```python
y = rew * reward_scale + gamma * q_target(next_obs).max(dim=1).values * (1.0 - done)
loss = smooth_l1_loss(q_sa, y)
```

- `(1 - done)`: if the robot just hit the goal or a wall, there is no "future", so the target is only `r`.
- **Vanilla** DQN: the same target network both *picks* the best next action and *evaluates* it. Double DQN splits those
  two jobs. This project deliberately does not use Double DQN, Dueling, or prioritised replay.
- A timeout (500 steps) is **not** treated as terminal, because the observation has no clock. The agent still bootstraps.

---

## 5. The three tricks that make it work

Neural networks trained on a stream of correlated, moving-target data tend to blow up. DQN adds three stabilisers:

### 5.1 Replay buffer (`ReplayBuffer`)
Every step `(s, a, r, s', done)` goes into a buffer holding the last 100,000 transitions. Each learning step samples
128 **random** ones. Why: consecutive frames are nearly identical (the robot barely moved), and training on them in order would
overfit to the last few seconds. Random sampling breaks that correlation and reuses data many times.

### 5.2 Target network (`q_target`)
A frozen copy of the Q-network used only to compute `y`. It is overwritten with the live weights every 1,000 steps (hard update).
Why: if the target moved every gradient step, you'd be chasing your own tail.

### 5.3 Epsilon-greedy exploration (`DQNAgent.act`)
- With probability epsilon, take a random action; otherwise take `argmax Q`.
- Epsilon decays linearly from 1.0 to 0.05 over 100,000 steps (fast tier).
- The first 5,000 steps are pure random warmup so the buffer has something to learn from.

Early on the robot flails randomly and mostly crashes; as epsilon falls it increasingly trusts its Q-values.
At **evaluation time epsilon = 0**: pure greedy, no randomness.

Extras: Huber loss (robust to large errors, such as the +/-100 terminal rewards) and gradient-norm clipping at 10.

---

## 6. One training step, end to end

Per environment step (`src/train.py` drives this):

1. **Observe** `s` (28 numbers).
2. **Act**: random with prob. epsilon, else `argmax Q(s, .)`.
3. **Step the world** for 0.1 s with that `(v, w)`.
4. **Receive** `s'`, reward `r`, and `done` (goal / collision).
5. **Store** `(s, a, r, s', done)` in the replay buffer.
6. **Learn** (after warmup): sample 128 transitions, compute `y`, take one Adam step (lr 5e-4) on the loss.
7. Every 1,000 steps: copy Q-network weights into the target network.
8. Every 5,000 steps: pause and **evaluate** 50 episodes greedily, log success / collision / timeout rates.

Repeat 200,000 times (fast tier) and the Q-network goes from random garbage to a usable obstacle-avoiding navigator.

---

## 7. Why two simulators?

| Tier | What it is | Used for |
|---|---|---|
| **fast** (`env_fast.py`) | Pure NumPy 2D sim, emulates a 24-beam LiDAR | Full 200k-step training in minutes |
| **gz** (`env_gz.py`) | ROS 2 + Gazebo Harmonic, real physics | Short 30k-step training, and testing the fast-trained policy |

Both expose the same Gymnasium interface, so the agent code is identical. Gazebo runs near real time (about 7-10 steps/s),
so 200k steps there would take most of a day. The plan: train fast, then check the policy still works in Gazebo
(**sim-to-sim transfer**).

---

## 8. What the results mean

(Held-out layouts, greedy, last checkpoint. See README "Results at a glance".)

| Condition | Success |
|---|---|
| Random policy | ~4% |
| Heading-only controller (ignores LiDAR; just steers at the goal) | 68% |
| **Fast-trained DQN, fast sim** | **73.7%** |
| Fast-trained DQN, Gazebo | 72.0% |

Read this honestly:
- DQN went from ~4% to ~74%: it clearly learned.
- But a controller that ignores obstacles already gets 68%, so DQN's real gain over "just steer at the goal" is about 6 points.
  Many layouts are easy; the learned obstacle avoidance helps on the harder ones.
- Transfer to Gazebo works (gap 1.6 points) after two Gazebo bugs were fixed: wheels losing traction after hard turns, and
  using drifting wheel odometry as the pose.

---

## 9. Common failure modes (and where they show up here)

| Symptom | Cause | Where to look |
|---|---|---|
| Robot spins in place forever | Heading angle not wrapped to [-pi, pi] | `envs/common.py` |
| Learning stalls near 0% | Sparse reward, too little shaping | `reward.progress_weight` |
| Success collapses mid-training | Q-values diverge, unstable targets | `results/q_stability.png` |
| Circling or wall-hugging for reward | Reward hacking | Progress term is potential-based to prevent it |
| Great in fast sim, bad in Gazebo | Sim-to-sim gap (dynamics, contact, pose source) | `results/transfer_diag/report.md` |

---

## 10. Try it yourself

```bash
cd ~/tb3_dqn/src
../venv/bin/python train.py --agent dqn --env fast --smoke      # ~2k steps, sanity check
../venv/bin/python train.py --agent dqn --env fast --seed 0     # full run
../venv/bin/python evaluate.py --checkpoint ../results/runs/fast_dqn_seed0/final.pt --env fast
../venv/bin/python plot.py                                      # regenerate figures
```

Good experiments to build intuition (change one value in `src/config.yaml` at a time):
- `reward.progress_weight`: 5 vs 50. See how reward shaping changes learning speed.
- `dqn.target_update_every`: 100 vs 5000. Watch stability change.
- `dqn.gamma`: 0.9 vs 0.99. Shorter vs longer foresight.
- `tiers.fast.eps_decay_steps`: how long to explore.

---

## 11. Glossary

- **Episode**: one attempt from reset to goal / collision / timeout.
- **Policy**: the rule mapping state to action. Here: `argmax Q`.
- **Bootstrapping**: using the network's own estimate of the future (`max Q(s', .)`) as part of the learning target.
- **Off-policy**: DQN can learn from old, random transitions in the buffer, not only from its current behaviour.
- **Greedy**: always pick the highest-Q action (epsilon = 0).
- **Sim-to-sim transfer**: train in one simulator, test in another.
- **Vanilla DQN**: the 2015 recipe (replay + target network + epsilon-greedy), with no later improvements.

## 12. Where it goes next (phase 2)

DQN can only choose from 5 fixed turns. The next phase adds **TD3**, an actor-critic method that outputs a continuous
`(v, w)`, so the robot could slow down near obstacles or turn exactly as much as needed. The envs, logging and plotting
were kept interface-stable so the comparison is apples to apples.
