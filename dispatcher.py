"""
dispatcher.py  —  Central Brain
Responsibilities:
  1. Maintain a priority-sorted order queue
  2. Select an assignment strategy via epsilon-greedy bandit
  3. Assign tasks to best-fit agents (role-aware + battery-aware)
  4. Decide which agents go to charge (Quicktron-style energy management)
  5. Expose metrics for visualization
"""

import heapq
import random
from collections import defaultdict
from agent import Agent, AgentRole, AgentState
from task import Task, Priority
from environment import WarehouseEnvironment
from strategies import STRATEGY_REGISTRY


class Dispatcher:
    def __init__(self, env: WarehouseEnvironment, epsilon: float = 0.2):
        self.env     = env
        self.epsilon = epsilon

        # Priority order queue (min-heap with negated priority for max-first)
        self._queue: list = []   # heap of (-priority, task)
        self.pending_tasks: list[Task] = []

        # Bandit
        self.num_strategies    = len(STRATEGY_REGISTRY)
        self.strategy_scores   = [0.0] * self.num_strategies
        self.strategy_counts   = [0]   * self.num_strategies
        self.last_strategy_idx = 0

        # History
        self.history      = []   # {round, strategy, reward}
        self.round_num    = 0
        self.round_rewards: list[float] = []
        self.role_history: list[dict]   = []
        self.all_tasks:    list[Task]   = []

        # Per-round tracking
        self._round_task_rewards: list[float] = []

    # ──────────────────────────────────────────
    # Order queue management
    # ──────────────────────────────────────────
    def add_task(self, task: Task):
        """Push task into priority queue."""
        # Negate priority so higher priority = lower heap value (max-first)
        heapq.heappush(self._queue, (-task.priority_score, task.id, task))
        self.pending_tasks.append(task)
        self.all_tasks.append(task)

    def add_tasks(self, tasks: list):
        for t in tasks:
            self.add_task(t)

    def pop_task(self) -> Task | None:
        while self._queue:
            _, _, task = heapq.heappop(self._queue)
            if task.status.name == 'PENDING' and not task.is_expired:
                return task
            elif task.is_expired:
                task.fail()
        return None

    def expire_old_tasks(self):
        expired = [t for t in self.pending_tasks
                   if t.is_expired and t.status.name == 'PENDING']
        for t in expired:
            t.fail()
        self.pending_tasks = [t for t in self.pending_tasks
                              if t.status.name in ('PENDING', 'ASSIGNED', 'IN_TRANSIT')]

    # ──────────────────────────────────────────
    # Bandit strategy selection
    # ──────────────────────────────────────────
    def select_strategy(self) -> int:
        if random.random() < self.epsilon:
            idx = random.randint(0, self.num_strategies - 1)
        else:
            avgs = [
                self.strategy_scores[i] / self.strategy_counts[i]
                if self.strategy_counts[i] > 0 else 0.0
                for i in range(self.num_strategies)
            ]
            idx = avgs.index(max(avgs))
        self.last_strategy_idx = idx
        return idx

    def update_bandit(self, strategy_idx: int, reward: float):
        self.strategy_counts[strategy_idx] += 1
        n = self.strategy_counts[strategy_idx]
        self.strategy_scores[strategy_idx] += (
            reward - self.strategy_scores[strategy_idx]) / n
        name = STRATEGY_REGISTRY[strategy_idx][0]
        self.history.append({
            'round':    self.round_num,
            'strategy': name,
            'reward':   reward,
        })

    def best_strategy(self) -> tuple:
        avgs = [
            self.strategy_scores[i] / self.strategy_counts[i]
            if self.strategy_counts[i] > 0 else float('-inf')
            for i in range(self.num_strategies)
        ]
        idx = avgs.index(max(avgs))
        return idx, STRATEGY_REGISTRY[idx][0]

    def get_usage_counts(self) -> dict:
        return {STRATEGY_REGISTRY[i][0]: self.strategy_counts[i]
                for i in range(self.num_strategies)}

    # ──────────────────────────────────────────
    # Task assignment
    # ──────────────────────────────────────────
    def assign_tasks(self, tasks: list) -> dict:
        """
        Assign tasks to idle agents using the bandit-selected strategy.
        Returns {task_id -> agent}.
        Battery-aware: skips agents that need charging.
        """
        idle_agents = [
            a for a in self.env.agents
            if a.state == AgentState.IDLE and not a.needs_charge()
        ]

        strategy_idx = self.select_strategy()
        strategy_fn  = STRATEGY_REGISTRY[strategy_idx][1]
        assignments  = strategy_fn(tasks, idle_agents, self.env)

        for task, agent in assignments.items() if isinstance(assignments, dict) else []:
            if isinstance(task, int):
                # assignments is {task_id -> agent}
                pass
            
        # Normalize: ensure we get {task_id: agent}
        if assignments and isinstance(list(assignments.keys())[0], Task):
            assignments = {t.id: a for t, a in assignments.items()}

        return assignments, strategy_idx

    # ──────────────────────────────────────────
    # Charging management (Quicktron-style)
    # ──────────────────────────────────────────
    def manage_charging(self):
        """
        Send low-battery agents to the nearest free charger.
        Priority: agents with lower battery go first.
        Agents already charging stay put.
        """
        low_battery = sorted(
            [a for a in self.env.agents
             if a.needs_charge() and a.state not in (AgentState.CHARGING, AgentState.TO_CHARGER)],
            key=lambda a: a.battery
        )

        for agent in low_battery:
            station = self.env.nearest_free_charger(agent)
            if station is None:
                continue  # All chargers busy — agent finishes current task first

            agent.state = AgentState.TO_CHARGER
            # Drop current task back into queue if it had one
            if agent.current_task and agent.current_task.status.name != 'DELIVERED':
                agent.current_task.status = type(agent.current_task.status).PENDING
                self.add_task(agent.current_task)
                agent.current_task = None

            agent.compute_path(
                agent.location, station.location,
                self.env.grid_size, self.env.obstacles,
                self.env.reservations
            )
            self.env.reserve_path(agent)
            station.occupied_by = agent.id   # pre-reserve station

    def tick_charger_movement(self):
        """Move agents heading to charger, start charging on arrival."""
        for agent in self.env.agents:
            if agent.state == AgentState.TO_CHARGER:
                station = next((s for s in self.env.charging_stations
                                if s.occupied_by == agent.id), None)
                if station is None:
                    agent.state = AgentState.IDLE
                    continue

                if agent.location == station.location:
                    self.env.occupy_charger(agent, station)
                else:
                    if not agent.path:
                        agent.compute_path(agent.location, station.location,
                                           self.env.grid_size, self.env.obstacles,
                                           self.env.reservations)
                    next_pos = agent.path[0] if agent.path else agent.location
                    if not self.env.detect_collision(agent, next_pos):
                        agent.step_path()

    # ──────────────────────────────────────────
    # Main tick (called every simulation step)
    # ──────────────────────────────────────────
    def tick(self, new_tasks: list = None) -> dict:
        """
        One simulation tick:
          1. Expire old tasks
          2. Add new tasks
          3. Manage charging
          4. Assign pending tasks to idle agents
          5. Step all agents
          6. Tick chargers
          7. Detect deadlocks
        Returns stats for this tick.
        """
        self.env.tick += 1

        # 1. Expire
        self.expire_old_tasks()

        # 2. New tasks
        if new_tasks:
            self.add_tasks(new_tasks)

        # 3. Charging management
        self.manage_charging()
        self.tick_charger_movement()

        # 4. Assign tasks to idle agents
        assignable = [t for t in self.pending_tasks
                      if t.status.name == 'PENDING' and not t.is_expired]
        # Sort by priority descending
        assignable.sort(key=lambda t: -t.priority_score)

        if assignable:
            assignments, strategy_idx = self.assign_tasks(assignable)
            for task in assignable:
                agent = assignments.get(task.id)
                if agent:
                    task.assign(agent)
                    agent.current_task = task
                    agent.state        = AgentState.TO_PICKUP
                    agent.path         = []
                    agent.compute_path(agent.location, task.pickup,
                                       self.env.grid_size, self.env.obstacles,
                                       self.env.reservations)
                    self.env.reserve_path(agent)

        # 5. Step each task-executing agent
        tick_rewards = []
        for agent in self.env.agents:
            if agent.state in (AgentState.TO_PICKUP, AgentState.TO_DROP,
                               AgentState.WAITING):
                reward = self.env.simulate_task_step(agent)
                if reward is not None:
                    tick_rewards.append(reward)
                    self._round_task_rewards.append(reward)

        # 6. Charger ticks
        self.env.tick_chargers()

        # 7. Deadlock check
        self.env.check_deadlocks()

        # 8. Dynamic congestion refresh every 20 ticks
        if self.env.dynamic_traffic and self.env.tick % 20 == 0:
            self.env.refresh_congestion()

        # Snapshot role distribution
        role_counts = defaultdict(int)
        for a in self.env.agents:
            role_counts[a.specialization] += 1
        self.role_history.append({
            'tick':       self.env.tick,
            'role_counts': dict(role_counts),
        })

        return {
            'tick':         self.env.tick,
            'rewards':      tick_rewards,
            'pending':      len([t for t in self.pending_tasks if t.status.name == 'PENDING']),
            'collisions':   self.env.collision_events,
            'deadlocks':    self.env.deadlocks_resolved,
        }

    # ──────────────────────────────────────────
    # Metrics
    # ──────────────────────────────────────────
    def summary(self) -> str:
        best_idx, best_name = self.best_strategy()
        lines = [
            "=== Dispatcher Summary ===",
            f"  Total ticks:         {self.env.tick}",
            f"  Tasks dispatched:    {len(self.all_tasks)}",
            f"  Collision events:    {self.env.collision_events}",
            f"  Deadlocks resolved:  {self.env.deadlocks_resolved}",
            f"  Charges completed:   {self.env.charges_completed}",
            f"  Best strategy:       {best_name}",
        ]
        return "\n".join(lines)
