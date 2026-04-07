# 🏭 Self-Learning Warehouse Dispatch System

A Python simulation of a warehouse where multiple robots dynamically receive and complete tasks.
The system **improves over time** using:
- **Meta-Learning** via Epsilon-Greedy Multi-Armed Bandit (strategy selection)
- **Agent Specialization** (robots evolve roles based on performance)

---

## 🧩 Project Structure

```
warehouse_dispatch/
├── main.py          # Entry point (headless or Streamlit)
├── agent.py         # Agent/Robot class with specialization logic
├── task.py          # Task class + task generator
├── environment.py   # 2D grid warehouse, cost/reward functions
├── strategies.py    # 4 assignment strategies
├── learning.py      # Epsilon-Greedy Bandit meta-learner
├── visualization.py # All charts + Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2a. Headless mode (saves PNG charts to `output_charts/`)
```bash
python main.py
```

### 2b. Streamlit dashboard (interactive UI)
```bash
streamlit run main.py
```

---

## 🧠 How the Learning System Works

### Meta-Learning (Multi-Armed Bandit)

The system treats each assignment strategy as an "arm" of a bandit machine:

| Strategy        | Description                              |
|----------------|------------------------------------------|
| `nearest_agent` | Assign to the closest robot              |
| `fastest_agent` | Assign to the fastest robot              |
| `least_loaded`  | Assign to the least-loaded robot         |
| `random`        | Assign randomly (exploration baseline)   |

**Epsilon-Greedy Logic:**
- With probability ε (default 0.2): pick a **random** strategy (explore)
- Otherwise: pick the strategy with the **highest average reward** (exploit)
- After each round, update the strategy's running average reward

Over time, the bandit converges to the best-performing strategy.

### Agent Specialization

Each agent tracks performance across 3 task categories:
- `short` — short-distance deliveries
- `long` — long-distance deliveries
- `heavy` — heavy-load tasks

After 5+ tasks in a category, each agent is assigned a **role**:
- `fast_agent` → best at short tasks
- `long_distance_agent` → best at long routes
- `heavy_agent` → best at heavy loads
- `generalist` → not yet specialized

Roles are updated dynamically each round.

---

## 📊 Output Charts

| Chart | Description |
|-------|-------------|
| `grid_final.png` | Warehouse grid with agent positions, zones, congestion |
| `performance.png` | Average reward per round (shows improvement) |
| `strategy_usage.png` | How many times each strategy was chosen (shows convergence) |
| `role_distribution.png` | Agent role breakdown over time (shows specialization) |
| `demand_heatmap.png` | Which pickup zones were most used |

---

## ⚙️ Configuration

Edit `CONFIG` in `main.py`:

```python
CONFIG = {
    'grid_size':       15,   # NxN grid size
    'num_agents':       5,   # Number of robots
    'tasks_per_round':  4,   # Tasks per simulation round
    'num_rounds':      60,   # Total rounds
    'epsilon':         0.2,  # Exploration rate (0=pure exploit, 1=pure random)
    'dynamic_traffic': True, # Refresh congestion zones each round
}
```

---

## 🎯 What the System Demonstrates

1. **Starts with no knowledge** — all strategies equally likely at round 1
2. **Learns the best strategy** — bandit converges to lowest-cost strategy
3. **Agents evolve specialized roles** — role distribution shifts away from `generalist`
4. **Overall efficiency improves** — average reward increases over 60 rounds

---

## 🧪 Tech Stack

- Python 3.8+
- `matplotlib` — charts and grid visualization
- `numpy` — smoothing and heatmap
- `streamlit` — interactive dashboard
- `random` — simulation randomness
