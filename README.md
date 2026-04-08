# 🏭 Adaptive Warehouse Dispatch System — v3

> A real-time, self-learning multi-agent warehouse simulator with live Pygame visualisation.
> Inspired by **Amazon Robotics (Kiva)**, **Locus Robotics**, and **6 River Systems Chuck**.

---

## Demo

<!-- Drag your screen recording into a GitHub Issue comment to get a shareable URL, then paste it here -->
https://github.com/AdityaSohal/Adaptive-Warehouse-Dispatch-System/blob/main/demo.mp4

> **How to embed your video:** Open any GitHub Issue, drag `demo.mp4` into the comment box, wait for the upload link to appear, copy it, and paste it above. GitHub renders `.mp4` files inline in READMEs.

---

## Quick Start

```bash
pip install -r requirements.txt
python simulation.py
```

### Controls

| Key / Action | Effect |
|---|---|
| `SPACE` | Pause / resume |
| `+` / `-` | Speed up / slow down |
| `R` | Reset simulation |
| `Q` / `ESC` | Quit |
| Click an agent | Expand agent detail in sidebar |

---

## What Is This?

This system simulates a warehouse floor where a fleet of autonomous mobile robots (AMRs) fulfil orders in real time. Every robot independently plans its path, manages its battery, and learns from experience — while a central dispatcher coordinates assignments, resolves deadlocks, and continuously trains shared neural policies across episodes.

The stack combines classical robotics algorithms (A\*, CBS, ORCA) with modern reinforcement learning (Q-learning, UCB1 bandit, MAPPO) in a single cohesive loop that becomes measurably smarter with each run.

---

## Architecture

```
warehouse/
├── simulation.py         ← Pygame live viewer (entry point)
├── dispatcher.py         ← Central brain: assignment, charging, MAPPO
├── agent.py              ← A* pathfinding, battery, MAPPO cost overlay
├── environment.py        ← Grid, shelves, CBS replanning, deadlocks
├── cbs_planner.py        ← Conflict-Based Search multi-agent planner
├── collision_deadlock.py ← ORCA avoidance + wait-for-graph resolution
├── scheduler.py          ← Hungarian/auction assignment + predictive charging
├── strategies.py         ← 5 UCB1 bandit strategies
├── rl_engine.py          ← Q-learner, UCB1 bandit, MAPPO, heatmap, model store
├── task.py               ← Order lifecycle + priority queue
├── config.py             ← All tunable constants in one place
├── train_and_benchmark.py← Headless training + chart generation
└── requirements.txt
```

---

## Fleet

| Type | Count | Speed | Capacity | Battery Drain |
|---|---|---|---|---|
| FAST | 4 | 2.0 | 15 kg | 1.4 / step |
| HEAVY | 2 | 1.0 | 40 kg | 0.9 / step |

Agents earn **specialisations** after completing 5+ tasks in a category: `runner` (short routes), `hauler` (long routes), `lifter` (heavy loads).

---

## Algorithms

### Path Planning — Conflict-Based Search (CBS)

A two-level search that finds collision-free paths for all agents simultaneously.

- **High level:** Constraint tree (CT) — detects agent conflicts, branches by adding constraints
- **Low level:** Space-time A\* per agent under its current constraint set
- **Edge cost:** base distance + heatmap penalty + MAPPO learned overlay
- **Fallback:** Prioritised planning when CBS budget (`CBS_MAX_ITERATIONS = 200`) is exceeded
- **Global replan** triggered every `CBS_REPLAN_INTERVAL = 10` ticks

### Collision Avoidance — ORCA

Each agent computes Optimal Reciprocal Collision Avoidance half-planes in velocity space, projecting its preferred velocity onto the feasible region. An emergency stop fires if time-to-collision drops below `TTC_EMERGENCY_THRESH = 2.0` ticks.

### Deadlock Detection & Resolution — Three Layers

1. **Space-time reservations** — agents claim future cells before moving
2. **Wait-for graph cycle detection** — DFS over the directed "A waits for B" graph; cycles = deadlocks
3. **Push-and-swap resolution** — lowest-battery agent in a cycle is BFS-pushed to the nearest free cell

Long-wait fallback: any agent stuck for ≥ 6 ticks without a formal cycle is yielded and replanned.

### Task Assignment

| Fleet size | Method | Complexity |
|---|---|---|
| ≤ 20 agents | **Hungarian algorithm** (scipy) | O(n³), globally optimal |
| > 20 agents | **Auction bidding** | O(n), scalable |

Both methods minimise a composite cost covering travel time, congestion, battery risk, capacity mismatch, priority, and urgency.

### Charging — Predictive + Q-Learning

1. **Predictive model:** estimate `battery_after_task = battery − (path_length × drain)`. If below `BATTERY_PREDICTIVE = 35%`, charge first.
2. **Q-learner consulted** when battery ≤ 28%
3. **Forced divert** regardless of queue when battery ≤ 12%
4. **Queue management:** max 2 agents per station; best station chosen by (queue depth, distance)
5. Interrupted tasks return to the priority queue at their original priority level

### Reinforcement Learning

#### Per-Agent Tabular Q-Learner

Decides **charge vs. work** from a 48-state encoding of `(battery_bucket, queue_pressure, has_critical, own_load)`.

| Parameter | Value |
|---|---|
| Algorithm | Q-learning |
| Learning rate α | 0.15 |
| Discount γ | 0.92 |
| Exploration | ε-greedy, 50% → 5% |

#### UCB1 Strategy Bandit

Selects among 5 assignment strategies. Near-optimal without a hand-tuned ε:

```
score_i = avg_reward_i + C × sqrt( ln(total_pulls) / pulls_i )
```

| Strategy | Description |
|---|---|
| `nearest` | Closest robot to pickup — minimises deadhead |
| `fastest` | Highest-speed robot — good for CRITICAL orders |
| `balanced` | Fewest completed tasks — fleet utilisation |
| `random` | Exploration baseline |
| `specialized` | Role-aware + cost-minimising |

#### MAPPO — Multi-Agent PPO

A shared neural policy (actor + critic) that **directly shapes A\* routing** via a learned cost overlay:

- After each rollout update, the actor's **action-distribution entropy** is sampled across the 22×22 grid
- High entropy → policy is uncertain here (congestion / obstacle-adjacent) → added as A\* edge cost
- Low entropy → policy is confident (clear corridor) → zero extra cost
- Agents naturally learn to prefer unobstructed corridors without explicit path constraints

This closes the loop between neural training and classical planning: MAPPO shapes the cost landscape; A\* finds the optimal path within it.

#### Congestion Heatmap

EMA-learned per-cell cost map fed into CBS edge weights:

```
heatmap[cell] = (1 − α) × heatmap[cell] + α × observed_cost
```

Collisions and waits are recorded as high-cost events (5.0 and 1.5 respectively), so the map learns which corridors are genuinely dangerous.

---

## Reward Function

| Event | Value |
|---|---|
| Successful delivery | +5.0 + priority × 1.5 |
| Full charge completed | +2.0 |
| Flat battery | −15.0 |
| Collision | −8.0 |
| Deadlock tick | −4.0 |
| Idle with tasks pending | −0.05 / tick |
| Congested step | −0.3 |

---

## Visual Design

| Element | Detail |
|---|---|
| Shelf obstacles | Brown racks with stacked colour-coded boxes |
| Pickup zones | Live inventory count; boxes visually deplete as tasks are assigned |
| Drop zones | Crosshair target marker |
| Chargers | Lightning-bolt icon; pulses when occupied |
| Agents | Colour ring = state; battery arc = charge level; floating package = carrying |
| Paths | Dashed fade lines showing planned A\* route per agent |
| Task markers | Filled diamond = pickup target; open diamond = drop target |
| Sidebar | KPI strip, efficiency / queue chart, UCB1 panel, fleet detail, order queue, event log |

---

## Order System

| Priority | Deadline |
|---|---|
| CRITICAL | 14 s |
| URGENT | 28 s |
| HIGH | 45 s |
| NORMAL | 65 s |
| LOW | 95 s |

Expired tasks → `FAILED`, surfaced in the UI event log. Each task has a live urgency bar in the sidebar that turns red as the deadline approaches.

---

## Model Persistence

All learned state is saved to `models/` at the end of each episode:

| File | Contents |
|---|---|
| `state.json` | Q-tables, UCB1 bandit counts, heatmap, episode metadata |
| `mappo.pkl` | MAPPO actor + critic weights (all agents) |

On the next run, models reload automatically so performance improves across sessions.

---

## Headless Training & Benchmarking

```bash
python train_and_benchmark.py --episodes 20 --ticks 300
```

Produces three charts in `benchmark_results/`:

| Chart | Contents |
|---|---|
| `learning_curves.png` | Efficiency, throughput, collisions, ε-decay over episodes |
| `strategy_convergence.png` | UCB1 arm pull distribution evolving over training |
| `kpi_summary.png` | Radar: first-quarter vs last-quarter episode performance |

---

## Extending

| Goal | Where |
|---|---|
| Add shelf rows | `NUM_SHELF_ROWS` / `SHELF_ROW_LEN` in `config.py` |
| Larger fleet | `NUM_FAST_AGENTS` / `NUM_HEAVY_AGENTS` in `config.py` |
| Larger grid | `GRID_SIZE` in `config.py` (window width auto-adjusts) |
| Live order feed | Call `dispatcher.add_task(task)` from a webhook thread |
| PyTorch MAPPO | Replace `LinearLayer` / `MAPPOActor` / `MAPPOCritic` with `torch.nn.Module` subclasses; the `forward / backward_and_update / to_dict / from_dict` interface is unchanged |
| Metrics export | `dispatcher.summary_dict()` returns a JSON-serialisable dict |
| 50–200 robots | Switch to `auction_assign`, set `CBS_MAX_ITERATIONS=50` |
| 200+ robots | Consider Rolling-Horizon CBS (RHCR) or decentralised ORCA only |

---

## Requirements

```
pygame>=2.1.0
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.9.0
torch>=2.0.0
```

PyTorch is optional for the current NumPy-based MAPPO. Install it when upgrading to a full `torch.nn` backend.

---

## References

- Sharon et al., *Conflict-Based Search for Optimal Multi-Agent Path Finding*, AAAI 2012
- van den Berg et al., *Reciprocal n-Body Collision Avoidance*, ISRR 2011
- Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv 2017
- Auer et al., *Finite-time Analysis of the Multiarmed Bandit Problem*, Machine Learning 2002