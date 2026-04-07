# Adaptive Warehouse Dispatch System  v2

A real-time, self-learning multi-agent warehouse simulator with live Pygame
visualisation — inspired by **Amazon Robotics (Kiva)**, **Locus Robotics**, and
**6 River Systems Chuck**.

---

## Quick start

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

## Architecture

```
warehouse/
├── simulation.py    ← Pygame live viewer (entry point)
├── dispatcher.py    ← Central brain: queue, bandit, charging decisions
├── agent.py         ← A* pathfinding, battery, RL integration
├── environment.py   ← Grid, shelves, collision/deadlock, chargers
├── strategies.py    ← 5 assignment strategies
├── task.py          ← Order + priority queue
├── rl_engine.py     ← Q-learner (per-agent) + UCB1 bandit
├── config.py        ← Single place for all tunable constants
└── requirements.txt
```

---

## What each file does

### `config.py`
Single source of truth for every constant in the system — grid size, fleet
composition, RL hyper-parameters, colour palette, window dimensions.  Edit
here; every other module imports from it.

### `rl_engine.py`  ← NEW
Two independent RL components.

#### `AgentQLearner`
Per-robot tabular Q-learner that decides **when to charge vs. keep working**.

| Element | Detail |
|---|---|
| State | `(battery_bucket, queue_pressure, has_critical, own_load)` — 48 possible states |
| Actions | `{0: work, 1: charge}` |
| Algorithm | Q-learning  (α=0.15, γ=0.92) |
| Exploration | ε-greedy decaying from 50 % → 5 % |
| Reward shaping | +priority×1.8 for delivery, −12 for flat battery, +3 for full charge |

Agents start exploring (randomly choosing between work and charge) and
gradually converge: high-battery agents almost always choose to work, while
sub-30 % agents learn to divert to a charger unless a CRITICAL order is
pending.

#### `StrategyBandit` (UCB1)
Replaces the original ε-greedy bandit with **Upper-Confidence-Bound (UCB1)**.

```
score_i = avg_reward_i + C × sqrt( ln(total_pulls) / pulls_i )
```

UCB1 is provably near-optimal without a hand-tuned ε and adapts to
non-stationary environments (e.g. congestion shifts every 22 ticks).

### `agent.py`
- Full **A\* pathfinding** with space-time reservation conflict avoidance
- **Local EMA memory map**: each agent privately caches cell congestion costs
- **Smooth pixel interpolation** for fluid rendering (lerp toward grid cell centre)
- **Battery management**: drain per step, charge at stations, RL-guided decisions
- **4 FAST** (speed=2, cap=15 kg) + **2 HEAVY** (speed=1, cap=40 kg)
- **Specialisation**: runner / hauler / lifter (unlocks after 5 category tasks)

### `environment.py`
- **22×22 grid** with pickup zones, drop zones, 3 chargers, congestion zones
- **Physical shelf obstacles** — A* plans routes around rows of warehouse shelves
- **Inventory tracking**: each pickup shelf holds 3–6 items; restocked every 25 ticks
- **Space-time reservation table**: prevents two agents occupying the same cell at the same tick
- **Deadlock detection**: agents waiting ≥ 6 ticks get yield-and-replan treatment
- **Dynamic congestion**: 25 % of zones shift every 22 ticks

### `dispatcher.py`
- **Priority queue**: orders sorted by Priority (LOW → CRITICAL) + deadline
- **UCB1 bandit**: selects among 5 strategies; converges to best performer
- **RL-guided charging manager**: consults each agent's Q-learner; overrides
  with force-charge below critical threshold
- **Per-tick orchestration**: expire → spawn → charge → assign → step → deadlock-check

### `strategies.py`

| Strategy | Description |
|---|---|
| `nearest` | Closest robot to pickup — minimises deadhead |
| `fastest` | Highest speed robot — good for CRITICAL orders |
| `balanced` | Agent with fewest completed tasks — fleet utilisation |
| `random` | Exploration baseline for the bandit |
| `specialized` | Role-aware (heavy→HEAVY, short→FAST) + cost-minimising |

### `task.py`
- 5 priority levels: LOW / NORMAL / HIGH / URGENT / CRITICAL
- Deadlines: CRITICAL=14 s, URGENT=28 s, HIGH=45 s, NORMAL=65 s, LOW=95 s
- Expired tasks → FAILED, surfaced in UI
- `urgency_ratio` property drives the per-task countdown bar in the sidebar

---

## Algorithms

### A\* with space-time reservations
Each agent runs A\* on the 22×22 grid with:
- **Heuristic**: Manhattan distance
- **Edge cost**: 1 + local memory congestion penalty
- **Space-time extension**: `reservations[(x,y,t)]` to avoid future conflicts
- **Swap conflict detection**: prevents two agents crossing each other head-on
- **Fallback**: greedy Manhattan if A\* finds no path

### Deadlock prevention (three-layer)
1. **Space-time reservations** — agents claim future cells before moving
2. **Wait threshold** (6 ticks) → yield to a free adjacent cell + fresh replan
3. **Charger re-queueing** — if a busy agent must charge, its task returns to
   the priority queue at original priority

### Charging (Quicktron-inspired + RL)
1. Battery ≤ 28 % → Q-learner consulted  
2. Battery ≤ 12 % → forced divert regardless of queue  
3. Station pre-reserved before agent arrives  
4. Task handed back to queue at original priority  
5. Positive reward (+3) fed to Q-learner on full charge completion  

### UCB1 Strategy Bandit
After a few warm-up pulls the bandit exploits the strategy with the best
upper-confidence bound.  Unlike ε-greedy it self-tunes to the current
environment without any fixed exploration rate.

---

## Visual design (Pygame)

Inspired by real AMR fleet dashboards:

| Element | Detail |
|---|---|
| Shelf obstacles | Drawn as brown racks with stacked colour-coded boxes |
| Pickup zones | Show live inventory count; boxes visually deplete as tasks are assigned |
| Drop zones | Crosshair target marker |
| Chargers | Lightning-bolt icon; pulses when occupied |
| Agents | Colour ring = state; battery arc = charge level; floating package = carrying |
| Paths | Dashed fade lines showing A\* route for each agent |
| Task markers | Diamond = pickup target; open diamond = drop target |
| Sidebar | KPI strip, rolling efficiency / queue chart, UCB1 strategy panel, fleet detail, order queue, event log |

---

## Extending

- **Add shelf rows**: increase `NUM_SHELF_ROWS` / `SHELF_ROW_LEN` in `config.py`
- **More agents**: bump `NUM_FAST_AGENTS` / `NUM_HEAVY_AGENTS`
- **Larger grid**: change `GRID_SIZE` (also adjusts `WINDOW_W` automatically)
- **Live order feed**: call `dispatcher.add_task(task)` from a webhook thread
- **DQN upgrade**: swap `AgentQLearner` for a neural-network Q-learner; the
  `encode / act / update` interface is identical
- **Metrics export**: `dispatcher.summary()` returns a JSON-serialisable dict
