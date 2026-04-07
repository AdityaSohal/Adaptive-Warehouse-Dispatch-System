# Adaptive Warehouse Dispatch System

A real-time, self-learning multi-agent warehouse simulator with live Pygame visualization.

---

## Quick start

```bash
pip install -r requirements.txt
python simulation.py
```

**Controls**

| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `+` / `-` | Speed up / slow down |
| `R` | Reset simulation |
| `Q` / `ESC` | Quit |
| Click agent | Inspect in sidebar |

---

## Architecture

```
warehouse/
├── simulation.py    ← Pygame live viewer (entry point)
├── dispatcher.py    ← Central brain: queue, bandit, charging decisions
├── agent.py         ← Agent with A* pathfinding + local memory
├── environment.py   ← Grid, collision detection, deadlock prevention
├── strategies.py    ← 5 assignment strategies
├── task.py          ← Order + priority queue
└── requirements.txt
```

---

## What each file does

### `agent.py`
- Full **A* pathfinding** with space-time reservation conflict avoidance
- **Local memory map**: each agent learns congestion costs via EMA
- **Battery management**: drain per step, charge at stations
- **Agent roles**: 4 FAST (speed=2, cap=15kg) + 2 HEAVY (speed=1, cap=40kg)
- **Specialization tracking**: agents evolve roles (speed_runner / long_hauler / heavy_lifter)

### `environment.py`
- 20×20 grid with pickup zones, drop zones, 3 charging stations, congestion zones
- **Space-time reservation table**: prevents two agents occupying the same cell at the same tick
- **Deadlock detection**: agents waiting > 5 ticks get yield-and-replan treatment
- **Dynamic congestion**: zones shift every 20 ticks (simulates real traffic)

### `dispatcher.py` — the main brain
- **Priority queue**: orders sorted by Priority (LOW → CRITICAL), with deadlines
- **Epsilon-greedy bandit**: selects among 5 assignment strategies, converges to best
- **Charging manager (Quicktron-style)**: sends lowest-battery agent to nearest free charger first; re-queues their unfinished task
- **Per-tick orchestration**: expire → spawn → charge → assign → step → deadlock-check

### `strategies.py`
Five strategies the bandit selects among:

| Strategy | Description |
|----------|-------------|
| `nearest_agent` | Closest robot to pickup |
| `fastest_agent` | Highest speed robot |
| `least_loaded` | Lowest current-load ratio |
| `random` | Baseline / exploration |
| `specialized` | Role-aware (heavy→HEAVY, short→FAST) |

### `task.py`
- 5 priority levels: LOW / NORMAL / HIGH / URGENT / CRITICAL
- Each priority has a time deadline (CRITICAL = 12s, LOW = 60s)
- Expired tasks are marked FAILED and removed from queue
- Task category (short / long / heavy) drives agent specialization

---

## Algorithms

### A* pathfinding
Each agent computes its own path using A* on the 20×20 grid.
- Heuristic: Manhattan distance
- Edge cost: 1 + local memory congestion penalty for that cell
- Space-time extension: checks `reservations[(x,y,t)]` to avoid future conflicts
- Fallback: greedy Manhattan if A* finds no path

### Deadlock prevention
1. **Space-time reservations**: agents claim cells at future ticks before moving
2. **Swap conflict detection**: prevents two agents crossing each other
3. **Wait threshold**: if an agent waits ≥ 5 ticks → yield to a free adjacent cell + replan
4. **Charger re-queueing**: if a busy agent must charge, its task is returned to queue

### Charging (Quicktron-inspired)
- Threshold: 25% battery → begin routing to nearest free station
- Priority: lowest battery goes first
- Station pre-reservation: station is marked occupied before agent arrives
- Task handoff: undelivered task returns to priority queue at original priority

---

## Hackathon pitch points

1. **Real-time decision making**: watch the system choose strategies and adapt live
2. **Observable cognition**: every agent's A* path is drawn; you see them think
3. **Emergent specialization**: agents drift away from "generalist" without being told to
4. **Graceful degradation**: deadlocks resolve themselves; expired orders are surfaced
5. **Energy awareness**: charging decisions compete with delivery urgency — just like real AMRs
6. **Demo-ready**: single `python simulation.py` — no server, no browser, no setup friction

---

## Extending for extra hackathon points

- **Add shelf obstacles**: populate `env.obstacles` with shelf coordinates → A* routes around them
- **Multi-floor**: stack two grids with elevator cells
- **Real order feed**: pipe live orders from a webhook into `dispatcher.add_task()`
- **Metrics export**: `dispatcher.summary()` outputs JSON for a live dashboard
- **Reinforcement learning**: replace the bandit with a DQN trained on the reward signal
