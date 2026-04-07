"""
dispatcher.py — Central brain of the warehouse simulation.

Responsibilities
────────────────
1. Maintain a priority-sorted order queue (max-heap via negated score).
2. Select an assignment strategy via UCB1 bandit (StrategyBandit).
3. Assign tasks to idle, battery-sufficient agents (role-aware).
4. Decide which agents go to charge using per-agent Q-learners.
5. Expose per-tick stats and rolling metrics for the visualiser.
"""

from __future__ import annotations

import heapq
from collections import defaultdict

from agent       import Agent, AgentState
from task        import Task, Priority
from environment import WarehouseEnvironment
from strategies  import STRATEGY_REGISTRY, STRATEGY_NAMES
from rl_engine   import StrategyBandit
from config      import BATTERY_CRITICAL, BATTERY_LOW, BATTERY_RESUME


class Dispatcher:
    def __init__(self, env: WarehouseEnvironment) -> None:
        self.env = env

        # ── Priority order queue (max-heap via negated priority) ────────────────
        self._heap:        list              = []
        self.pending_tasks: list[Task]       = []
        self.all_tasks:     list[Task]       = []

        # ── UCB1 bandit over strategies ─────────────────────────────────────────
        self.bandit = StrategyBandit(
            num_arms  = len(STRATEGY_REGISTRY),
            arm_names = STRATEGY_NAMES,
        )
        self.last_strategy_idx = 0

        # ── History / metrics ────────────────────────────────────────────────────
        self.round_num   = 0
        self.history:     list[dict] = []   # per-assignment records
        self.tick_rewards: list[float] = []

    # ── Queue management ─────────────────────────────────────────────────────────
    def add_task(self, task: Task) -> None:
        heapq.heappush(self._heap, (-task.priority_score, task.id, task))
        self.pending_tasks.append(task)
        self.all_tasks.append(task)

    def add_tasks(self, tasks: list[Task]) -> None:
        for t in tasks:
            self.add_task(t)

    def expire_old_tasks(self) -> None:
        expired = [t for t in self.pending_tasks
                   if t.is_expired and t.status.name == 'PENDING']
        for t in expired:
            t.fail()
        self.pending_tasks = [
            t for t in self.pending_tasks
            if t.status.name in ('PENDING', 'ASSIGNED', 'IN_TRANSIT')
        ]

    def _replenish_queue(self) -> None:
        """Re-sync heap from pending_tasks (needed after re-queueing a task)."""
        self._heap = [(-t.priority_score, t.id, t)
                      for t in self.pending_tasks if t.status.name == 'PENDING']
        heapq.heapify(self._heap)

    # ── Charging management (RL-guided) ──────────────────────────────────────────
    def manage_charging(self) -> None:
        """
        For each idle or working agent, consult its Q-learner to decide whether
        to divert to a charger.  Critically-low agents are forced to charge.
        """
        pending_count = sum(1 for t in self.pending_tasks if t.status.name == 'PENDING')
        has_critical  = any(
            t.priority == Priority.CRITICAL and t.status.name == 'PENDING'
            for t in self.pending_tasks
        )

        for agent in self.env.agents:
            if agent.state in (AgentState.CHARGING, AgentState.TO_CHARGER):
                continue

            force = agent.critically_low()
            rl_says_charge = False

            if not force and agent.state == AgentState.IDLE:
                rl_says_charge = agent.ql.decide_charge(
                    battery      = agent.battery,
                    queue_len    = pending_count,
                    has_critical = has_critical,
                    is_working   = False,
                )

            if force or (agent.needs_charge() and rl_says_charge):
                station = self.env.nearest_free_charger(agent)
                if station is None:
                    continue
                self._divert_to_charge(agent, station)

    def _divert_to_charge(self, agent: Agent, station) -> None:
        """Route an agent to a charging station; re-queue its current task."""
        if agent.current_task and agent.current_task.status.name not in ('DELIVERED', 'FAILED'):
            t = agent.current_task
            t.status    = type(t.status).PENDING   # re-open
            t.assigned_agent = None
            self.pending_tasks.append(t)
            self._replenish_queue()
            agent.current_task = None
            agent.carrying     = False

        agent.state = AgentState.TO_CHARGER
        station.occupied_by = agent.id
        agent.path = agent.compute_path(
            agent.location, station.location,
            self.env.grid_size, self.env.obstacles,
            self.env.reservations,
        )
        self.env.reserve_path(agent)

    def tick_charger_movement(self) -> None:
        """Advance agents that are heading toward a charger."""
        for agent in self.env.agents:
            if agent.state != AgentState.TO_CHARGER:
                continue
            station = next(
                (s for s in self.env.charging_stations if s.occupied_by == agent.id),
                None,
            )
            if station is None:
                agent.state = AgentState.IDLE
                continue
            if agent.location == station.location:
                self.env.occupy_charger(agent, station)
            else:
                if not agent.path:
                    agent.path = agent.compute_path(
                        agent.location, station.location,
                        self.env.grid_size, self.env.obstacles,
                        self.env.reservations,
                    )
                if agent.path:
                    next_pos = agent.path[0]
                    if not self.env.detect_collision(agent, next_pos):
                        agent.step_path()

    # ── Task assignment ───────────────────────────────────────────────────────────
    def assign_tasks(self, tasks: list[Task]) -> dict[int, Agent]:
        idle = [
            a for a in self.env.agents
            if a.state == AgentState.IDLE and not a.critically_low()
        ]
        if not idle or not tasks:
            return {}

        arm_idx     = self.bandit.select()
        strategy_fn = STRATEGY_REGISTRY[arm_idx][1]
        self.last_strategy_idx = arm_idx

        assignments = strategy_fn(tasks, idle, self.env)
        return assignments

    # ── Main tick ─────────────────────────────────────────────────────────────────
    def tick(self, new_tasks: list[Task] | None = None) -> dict:
        """
        One simulation step (called every tick_ms):
          1. expire old tasks
          2. ingest new tasks
          3. manage charging (RL)
          4. assign pending tasks
          5. step all executing agents
          6. tick chargers
          7. detect deadlocks
          8. refresh congestion (every N ticks)
        Returns a stats dict for the UI.
        """
        self.env.tick += 1

        # 1. Expire
        self.expire_old_tasks()

        # 2. New tasks
        if new_tasks:
            self.add_tasks(new_tasks)

        # 3. Charging
        self.manage_charging()
        self.tick_charger_movement()

        # 4. Assign
        assignable = sorted(
            [t for t in self.pending_tasks if t.status.name == 'PENDING' and not t.is_expired],
            key=lambda t: (-t.priority_score, t.deadline),
        )
        if assignable:
            assignments = self.assign_tasks(assignable)
            for task in assignable:
                agent = assignments.get(task.id)
                if agent is None:
                    continue
                task.assign(agent)
                agent.current_task = task
                agent.state        = AgentState.TO_PICKUP
                agent.carrying     = False
                agent.path         = agent.compute_path(
                    agent.location, task.pickup,
                    self.env.grid_size, self.env.obstacles,
                    self.env.reservations,
                )
                self.env.reserve_path(agent)
                self.history.append({
                    'tick':     self.env.tick,
                    'task_id':  task.id,
                    'agent_id': agent.id,
                    'strategy': STRATEGY_REGISTRY[self.last_strategy_idx][0],
                    'priority': task.priority.name,
                })

        # 5. Step agents
        tick_rewards = []
        for agent in self.env.agents:
            if agent.state in (AgentState.TO_PICKUP, AgentState.TO_DROP, AgentState.WAITING):
                reward = self.env.simulate_task_step(agent)
                if reward is not None:
                    tick_rewards.append(reward)
                    self.tick_rewards.append(reward)
                    # Bandit update: use negative cost as reward (higher = better)
                    self.bandit.update(self.last_strategy_idx, reward, self.env.tick)

        # 6. Charger ticks
        self.env.tick_chargers()

        # 7. Deadlocks
        self.env.check_deadlocks()

        # 8. Congestion refresh
        if self.env.dynamic_traffic and self.env.tick % 22 == 0:
            self.env.refresh_congestion()

        return {
            'tick':       self.env.tick,
            'rewards':    tick_rewards,
            'pending':    sum(1 for t in self.pending_tasks if t.status.name == 'PENDING'),
            'collisions': self.env.collision_events,
            'deadlocks':  self.env.deadlocks_resolved,
            'charges':    self.env.charges_completed,
        }

    # ── Metrics ───────────────────────────────────────────────────────────────────
    def total_delivered(self) -> int:
        return sum(1 for t in self.all_tasks if t.status.name == 'DELIVERED')

    def total_failed(self) -> int:
        return sum(1 for t in self.all_tasks if t.status.name == 'FAILED')

    def efficiency(self) -> float:
        d = self.total_delivered()
        f = self.total_failed()
        total = d + f
        return d / total if total > 0 else 0.0

    def avg_reward(self) -> float:
        if not self.tick_rewards:
            return 0.0
        window = self.tick_rewards[-50:]
        return sum(window) / len(window)

    def summary(self) -> str:
        idx, name = self.bandit.best_arm()
        lines = [
            "=== Dispatcher Summary ===",
            f"  Total ticks:         {self.env.tick}",
            f"  Tasks dispatched:    {len(self.all_tasks)}",
            f"  Delivered:           {self.total_delivered()}",
            f"  Failed:              {self.total_failed()}",
            f"  Efficiency:          {self.efficiency()*100:.1f}%",
            f"  Collisions:          {self.env.collision_events}",
            f"  Deadlocks resolved:  {self.env.deadlocks_resolved}",
            f"  Charges completed:   {self.env.charges_completed}",
            f"  Best strategy:       {name}",
        ]
        return "\n".join(lines)
