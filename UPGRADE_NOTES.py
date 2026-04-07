"""
UPGRADE NOTES — Adaptive Warehouse Dispatch System  v2 → v3
============================================================

NEW FILES
─────────
  cbs_planner.py        Conflict-Based Search path planner
  collision_deadlock.py ORCA avoidance + wait-for-graph deadlock resolution
  scheduler.py          Hungarian/auction task assignment + predictive charging scheduler

CHANGED FILES
─────────────
  config.py     Many new constants (CBS, ORCA, MAPPO, heatmap, charging)
  rl_engine.py  + MAPPOPolicy, CongestionHeatmap, ModelStore  (Q-learner/bandit API unchanged)
  environment.py + CBS, ORCA, WaitForGraph, heatmap (environment interface unchanged)
  dispatcher.py + Hungarian, ChargingScheduler, MAPPO, ModelStore (tick() API unchanged)

UNCHANGED FILES (drop in as-is from v2)
────────────────────────────────────────
  agent.py       (A* fallback still used; CBS now primary)
  strategies.py  (still used by bandit for legacy tracking)
  task.py
  simulation.py  (Pygame viewer; add heatmap overlay drawing for full benefit)

MIGRATION STEPS
───────────────
1. pip install -r requirements.txt
2. Copy all new/changed files into your warehouse/ directory.
3. On first run a models/ folder is created automatically.
4. After each episode, call dispatcher.end_episode() — this triggers MAPPO
   update and saves all learned state.  On the next run, models are reloaded
   automatically and performance improves over time.

KEY ALGORITHMS
──────────────
Path planning    Conflict-Based Search (CBS) + space-time A* per agent
                 Fallback: prioritised planning (A* per agent, prior paths = constraints)
Collision avoid  ORCA velocity-obstacle half-planes + time-to-collision emergency stop
Deadlock detect  Wait-for-graph cycle detection (DFS) + long-wait fallback
Deadlock resolve Push-and-swap: lowest-battery agent in cycle moves to nearest free cell
Task assignment  Hungarian algorithm (n≤20 agents, globally optimal O(n³))
                 Auction bidding (n>20 agents, scalable)
Charging         ChargingScheduler: predictive (battery-after-task estimate),
                 queue-aware (max CHARGE_MAX_QUEUE per station), Q-learner guided
RL               Tabular Q-learner (charge/work), UCB1 bandit (strategy), MAPPO (nav)
Heatmap          EMA congestion learner fed into CBS A* edge weights
Persistence      ModelStore: JSON (Q-tables, bandit, heatmap) + pickle (MAPPO weights)

REWARD FUNCTION (MAPPO + Q-learner)
────────────────────────────────────
  +REWARD_DELIVERY  per successful delivery  (default +5.0)
  +priority × REWARD_PRIORITY_SCALE           (default ×1.5)
  +REWARD_CHARGE_COMPLETE  on full charge     (default +2.0)
  -PENALTY_COLLISION      per collision       (default −8.0)
  -PENALTY_DEADLOCK       per deadlock tick   (default −4.0)
  -PENALTY_IDLE           per idle tick       (default −0.05)
  -PENALTY_FLAT_BATTERY   battery hits 0      (default −15.0)
  -PENALTY_CONGESTION     per congested step  (default −0.3)

METRICS TO TRACK
────────────────
  throughput         deliveries per tick
  efficiency         delivered / (delivered + failed)
  collision_rate     collisions per tick
  deadlock_freq      deadlocks per 100 ticks
  charge_queue_time  avg ticks spent waiting for a charger
  battery_util       avg battery % across fleet at any tick
  heatmap_hot_cells  cells with learned cost > threshold (congestion awareness)

SCALING
───────
  10–50  robots  : CBS + Hungarian work well
  50–200 robots  : Enable auction_assign (auto-selected), switch CBS to
                   bounded-CBS with CBS_MAX_ITERATIONS=50; use prioritised
                   planning as primary and CBS only for dense corridors.
  200+   robots  : Consider RHCR (Rolling-Horizon CBS) or decentralised ORCA only.

PYTORCH UPGRADE PATH (for full MAPPO performance)
──────────────────────────────────────────────────
  1. pip install torch
  2. In rl_engine.py replace LinearLayer / MAPPOActor / MAPPOCritic with
     torch.nn.Module subclasses (same interface: forward(), to_dict(), from_dict()).
  3. The MAPPOPolicy.update() method already follows PPO structure;
     replace the NumPy gradient approximation with torch autograd.
  4. Everything else (ModelStore, Dispatcher, environment) stays unchanged.
"""
