"""
dispatcher.py v3 — Central orchestrator with full ML integration.

Upgrades over v2:
  • Hungarian / auction task assignment (replaces single UCB1 strategy)
  • ChargingScheduler with predictive charging and queue management
  • CBS global replanning triggered every CBS_REPLAN_INTERVAL ticks
  • MAPPO policy stored and loaded between episodes
  • ModelStore persistence (Q-tables, bandit, heatmap, MAPPO)
  • Episode-level metrics for RL training loop
"""

from __future__ import annotations

import heapq
import os
from collections import defaultdict

from agent        import Agent, AgentState
from environment  import WarehouseEnvironment
from rl_engine    import AgentQLearner, StrategyBandit, MAPPOPolicy, CongestionHeatmap, ModelStore
from scheduler    import hungarian_assign, auction_assign, ChargingScheduler, assignment_cost
from strategies   import STRATEGY_REGISTRY, STRATEGY_NAMES
from task         import Task, Priority
from config       import (
    BATTERY_CRITICAL, BATTERY_LOW, BATTERY_RESUME,
    CBS_REPLAN_INTERVAL, MAPPO_ENABLED, MODEL_SAVE_INTERVAL,
    NUM_FAST_AGENTS, NUM_HEAVY_AGENTS, MODEL_DIR,
)


# Globals used by ChargingScheduler (imported there via duck-typing)
_pending_count = 0
_has_critical  = False


class Dispatcher:
    """
    Central brain.  One instance per simulation run (episode).

    Call flow each tick:
      expire → ingest → restock → CBS replan (every N ticks)
      → charging decisions → assign tasks → step agents
      → tick chargers → deadlock check → congestion refresh
      → MAPPO buffer store → (end of episode: update + save)
    """

    def __init__(self, env: WarehouseEnvironment, episode: int = 0) -> None:
        self.env     = env
        self.episode = episode

        # ── Priority queue ──────────────────────────────────────────────────────
        self._heap:         list        = []
        self.pending_tasks: list[Task]  = []
        self.all_tasks:     list[Task]  = []

        # ── Strategy bandit ─────────────────────────────────────────────────────
        self.bandit = StrategyBandit(
            num_arms  = len(STRATEGY_REGISTRY),
            arm_names = STRATEGY_NAMES,
        )
        self.last_strategy_idx = 0

        # ── Charging scheduler ──────────────────────────────────────────────────
        self.charge_sched = ChargingScheduler(env.charging_stations)
        self.charge_sched.set_env(env)

        # ── MAPPO ───────────────────────────────────────────────────────────────
        n_agents      = len(env.agents)
        self.mappo     = MAPPOPolicy(n_agents) if MAPPO_ENABLED else None
        self._obs_prev: dict[int, object] = {}   # agent_id → last obs

        # ── Model store ─────────────────────────────────────────────────────────
        self.store   = ModelStore(MODEL_DIR)
        self._load_models()

        # ── Metrics ─────────────────────────────────────────────────────────────
        self.round_num       = 0
        self.history:         list[dict]  = []
        self.tick_rewards:    list[float] = []
        self.episode_metrics: dict        = defaultdict(float)

    def _load_models(self) -> None:
        qlearners = [a.ql for a in self.env.agents]
        meta = self.store.load_all(
            self.mappo, qlearners, self.bandit, self.env.heatmap
        )
        self.episode = meta.get("episode", 0)

    # ── Queue helpers ─────────────────────────────────────────────────────────────
    def add_task(self, task: Task) -> None:
        heapq.heappush(self._heap, (-task.priority_score, task.id, task))
        self.pending_tasks.append(task)
        self.all_tasks.append(task)

    def add_tasks(self, tasks: list[Task]) -> None:
        for t in tasks:
            self.add_task(t)

    def expire_old_tasks(self) -> None:
        expired = [t for t in self.pending_tasks if t.is_expired and t.status.name == "PENDING"]
        for t in expired:
            t.fail()
        self.pending_tasks = [
            t for t in self.pending_tasks
            if t.status.name in ("PENDING", "ASSIGNED", "IN_TRANSIT")
        ]

    def _replenish_queue(self) -> None:
        self._heap = [(-t.priority_score, t.id, t)
                      for t in self.pending_tasks if t.status.name == "PENDING"]
        heapq.heapify(self._heap)

    # ── Charging management ───────────────────────────────────────────────────────
    def manage_charging(self) -> None:
        global _pending_count, _has_critical
        _pending_count = sum(1 for t in self.pending_tasks if t.status.name == "PENDING")
        _has_critical  = any(
            t.priority == Priority.CRITICAL and t.status.name == "PENDING"
            for t in self.pending_tasks
        )
        self.env.pending_proxy = [t for t in self.pending_tasks if t.status.name == "PENDING"]

        self.charge_sched.expire_old_reservations(self.env.tick)

        for agent in self.env.agents:
            if agent.state in (AgentState.CHARGING, AgentState.TO_CHARGER):
                continue
            if self.charge_sched.should_charge_now(
                agent, self.env.agents, self.env.tick, _has_critical
            ):
                station = self.charge_sched.reserve_charger(agent, self.env.tick)
                if station:
                    self._divert_to_charge(agent, station)

    def _divert_to_charge(self, agent: Agent, station) -> None:
        if agent.current_task and agent.current_task.status.name not in ("DELIVERED", "FAILED"):
            t = agent.current_task
            t.status         = type(t.status).PENDING
            t.assigned_agent = None
            self.pending_tasks.append(t)
            self._replenish_queue()
            agent.current_task = None
            agent.carrying     = False

        agent.state         = AgentState.TO_CHARGER
        station.occupied_by = agent.id
        agent.path          = agent.compute_path(
            agent.location, station.location,
            self.env.grid_size, self.env.obstacles,
            self.env.reservations,
        )
        self.env.reserve_path(agent)

    def tick_charger_movement(self) -> None:
        for agent in self.env.agents:
            if agent.state != AgentState.TO_CHARGER:
                continue
            station = next(
                (s for s in self.env.charging_stations if s.occupied_by == agent.id), None
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

        n_agents = len(self.env.agents)
        if n_agents <= 20:
            # Hungarian: globally optimal
            assignments = hungarian_assign(tasks, idle, self.env)
        else:
            # Auction: scalable
            assignments = auction_assign(tasks, idle, self.env)

        # Track bandit reward for chosen combo
        arm_idx = self.bandit.select()
        self.last_strategy_idx = arm_idx
        return assignments

    # ── Main tick ─────────────────────────────────────────────────────────────────
    def tick(self, new_tasks: list[Task] | None = None) -> dict:
        self.env.tick += 1

        # 1. Expire + ingest
        self.expire_old_tasks()
        if new_tasks:
            self.add_tasks(new_tasks)

        # 2. Restock inventories
        self.env._restock_inventories()

        # 3. CBS global replan
        if self.env.tick % CBS_REPLAN_INTERVAL == 0:
            self.env.global_replan()

        # 4. Charging
        self.manage_charging()
        self.tick_charger_movement()

        # 5. Assign pending tasks
        assignable = sorted(
            [t for t in self.pending_tasks if t.status.name == "PENDING" and not t.is_expired],
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
                    "tick":     self.env.tick,
                    "task_id":  task.id,
                    "agent_id": agent.id,
                    "strategy": STRATEGY_NAMES[self.last_strategy_idx],
                    "priority": task.priority.name,
                })

        # 6. Step agents + collect MAPPO observations
        tick_rewards = []
        for i, agent in enumerate(self.env.agents):
            # MAPPO observation before step
            if self.mappo:
                obs = MAPPOPolicy.encode_obs(agent, self.env)
                self._obs_prev[agent.id] = obs

            if agent.state in (AgentState.TO_PICKUP, AgentState.TO_DROP, AgentState.WAITING):
                reward = self.env.simulate_task_step(agent)
                if reward is not None:
                    tick_rewards.append(reward)
                    self.tick_rewards.append(reward)
                    self.bandit.update(self.last_strategy_idx, reward, self.env.tick)
                    self.episode_metrics["total_reward"] += reward

                    # MAPPO transition
                    if self.mappo:
                        next_obs = MAPPOPolicy.encode_obs(agent, self.env)
                        self.mappo.store_transition(i, self._obs_prev[agent.id], 1, reward, next_obs, False)

        # 7. Charger + deadlock
        self.env.tick_chargers()
        self.env.check_deadlocks()
        self.episode_metrics["deadlocks"] = self.env.deadlocks_resolved
        self.episode_metrics["collisions"] = self.env.collision_events

        # 8. Congestion refresh
        if self.env.dynamic_traffic and self.env.tick % 22 == 0:
            self.env.refresh_congestion()

        return {
            "tick":       self.env.tick,
            "rewards":    tick_rewards,
            "pending":    sum(1 for t in self.pending_tasks if t.status.name == "PENDING"),
            "collisions": self.env.collision_events,
            "deadlocks":  self.env.deadlocks_resolved,
            "charges":    self.env.charges_completed,
        }

    # ── End-of-episode ────────────────────────────────────────────────────────────
    def end_episode(self) -> dict:
        """
        Call when a simulation run (episode) completes.
        Updates MAPPO, increments episode counter, saves all models.
        Returns a summary dict.
        """
        self.episode += 1

        # MAPPO update
        if self.mappo:
            self.mappo.update()

        summary = self.summary_dict()

        if self.episode % MODEL_SAVE_INTERVAL == 0:
            self.store.save_all(
                mappo     = self.mappo,
                qlearners = [a.ql for a in self.env.agents],
                bandit    = self.bandit,
                heatmap   = self.env.heatmap,
                episode   = self.episode,
                metrics   = summary,
            )

        return summary

    # ── Metrics ───────────────────────────────────────────────────────────────────
    def total_delivered(self) -> int:
        return sum(1 for t in self.all_tasks if t.status.name == "DELIVERED")

    def total_failed(self) -> int:
        return sum(1 for t in self.all_tasks if t.status.name == "FAILED")

    def efficiency(self) -> float:
        d = self.total_delivered()
        f = self.total_failed()
        return d / (d + f) if (d + f) > 0 else 0.0

    def avg_reward(self) -> float:
        w = self.tick_rewards[-50:]
        return sum(w) / len(w) if w else 0.0

    def throughput(self) -> float:
        """Deliveries per tick."""
        if self.env.tick == 0:
            return 0.0
        return self.total_delivered() / self.env.tick

    def summary_dict(self) -> dict:
        idx, name = self.bandit.best_arm()
        return {
            "episode":           self.episode,
            "ticks":             self.env.tick,
            "delivered":         self.total_delivered(),
            "failed":            self.total_failed(),
            "efficiency":        round(self.efficiency() * 100, 1),
            "throughput":        round(self.throughput(), 4),
            "collisions":        self.env.collision_events,
            "deadlocks":         self.env.deadlocks_resolved,
            "charges":           self.env.charges_completed,
            "best_strategy":     name,
            "avg_reward":        round(self.avg_reward(), 2),
            "heatmap_hot_cells": len(self.env.heatmap._map),
        }

    def summary(self) -> str:
        d = self.summary_dict()
        lines = [
            "=== Dispatcher Summary v3 ===",
            f"  Episode:            {d['episode']}",
            f"  Total ticks:        {d['ticks']}",
            f"  Delivered:          {d['delivered']}",
            f"  Failed:             {d['failed']}",
            f"  Efficiency:         {d['efficiency']}%",
            f"  Throughput:         {d['throughput']} del/tick",
            f"  Collisions:         {d['collisions']}",
            f"  Deadlocks resolved: {d['deadlocks']}",
            f"  Charges:            {d['charges']}",
            f"  Best strategy:      {d['best_strategy']}",
            f"  Avg reward:         {d['avg_reward']}",
            f"  Heatmap cells:      {d['heatmap_hot_cells']}",
        ]
        return "\n".join(lines)
