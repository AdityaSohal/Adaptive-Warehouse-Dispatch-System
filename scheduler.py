"""
scheduler.py — Advanced task assignment and predictive charging scheduler.

Task assignment
───────────────
1. Hungarian algorithm for globally optimal 1-to-1 assignment when
   fleet ≤ 20 agents (O(n³) so fine for typical fleets).
2. Auction-based bidding for larger fleets: each agent computes a bid
   (negative cost) for each task; tasks are auctioned to the lowest bidder.
3. Both methods factor in: distance, congestion, battery, queue depth,
   task priority, and travel-time estimation.

Charging scheduler
──────────────────
Predictive model:
  For each agent, estimate battery at task completion:
      batt_pred = battery - (path_length × drain_per_step)
  If batt_pred < BATTERY_PREDICTIVE → schedule charging before the task.
  Also avoids charging all robots simultaneously (queue management).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

from config import (
    BATTERY_PREDICTIVE, BATTERY_CRITICAL, BATTERY_FULL,
    CHARGE_RATE, CHARGE_RESERVE_WINDOW, CHARGE_MAX_QUEUE,
    CHARGE_PRIORITY_BONUS, HEAVY_WEIGHT_KG, SHORT_DIST_CELLS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Cost function
# ══════════════════════════════════════════════════════════════════════════════

def assignment_cost(agent, task, env) -> float:
    """
    Comprehensive cost for assigning `agent` to `task`.
    Lower = better assignment.

    Factors:
      - Travel time (distance / speed) to pickup + delivery distance
      - Congestion along both path segments
      - Battery sufficiency (prefer agents with more battery)
      - Task priority urgency (high priority = lower cost = prefer assigning)
      - Queue pressure at drop zone (avoids piling up)
    """
    pickup_dist = env.manhattan(agent.location, task.pickup)
    total_dist  = pickup_dist + task.distance

    # Time estimate
    time_est = total_dist / max(agent.speed, 0.1)

    # Congestion estimate along route
    cong = (
        env.congestion_penalty(agent.location, task.pickup)
        + env.congestion_penalty(task.pickup, task.drop)
    )

    # Battery penalty: how close would we be to empty after the task?
    batt_after = agent.battery - total_dist * agent.drain_per_step
    batt_risk  = max(0.0, -batt_after * 0.1)   # positive if would run flat

    # Weight/capacity match
    capacity_pen = 0.0 if task.weight <= agent.capacity else 5.0

    # Priority bonus (higher priority = reduce cost)
    priority_bonus = (task.priority_score - 1) * 1.2

    # Urgency: tasks near expiry cost more to leave unassigned → assign ASAP
    urgency = task.urgency_ratio * 3.0

    return max(0.01, time_est + cong + batt_risk + capacity_pen - priority_bonus + urgency)


# ══════════════════════════════════════════════════════════════════════════════
# Hungarian algorithm (Kuhn-Munkres via scipy if available, else pure Python)
# ══════════════════════════════════════════════════════════════════════════════

def _hungarian_pure(cost_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """
    Simple O(n³) Hungarian algorithm implementation.
    Returns list of (row, col) assignments.
    """
    n = len(cost_matrix)
    m = len(cost_matrix[0]) if n > 0 else 0
    sz = max(n, m)

    # Pad to square
    C = [[cost_matrix[i][j] if i < n and j < m else 1e9 for j in range(sz)] for i in range(sz)]

    u = [0.0] * (sz + 1)
    v = [0.0] * (sz + 1)
    p = [0] * (sz + 1)    # p[j] = row assigned to column j (1-indexed)
    way = [0] * (sz + 1)

    for i in range(1, sz + 1):
        p[0] = i
        j0   = 0
        minv = [1e18] * (sz + 1)
        used = [False] * (sz + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = 1e18
            j1    = -1

            for j in range(1, sz + 1):
                if not used[j]:
                    cur = C[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j]  = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1    = j

            for j in range(sz + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j]    -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while j0:
            p[j0] = p[way[j0]]
            j0    = way[j0]

    assignments = []
    for j in range(1, sz + 1):
        if p[j] != 0 and p[j] - 1 < n and j - 1 < m:
            assignments.append((p[j] - 1, j - 1))

    return assignments


def hungarian_assign(
    tasks: list,
    agents: list,
    env,
) -> dict[int, object]:
    """
    Globally optimal task-agent assignment via the Hungarian algorithm.
    Returns {task_id: agent}.
    """
    if not tasks or not agents:
        return {}

    cost_matrix = [
        [assignment_cost(agent, task, env) for agent in agents]
        for task in tasks
    ]

    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        row_ind, col_ind = linear_sum_assignment(np.array(cost_matrix))
        pairs = list(zip(row_ind, col_ind))
    except ImportError:
        pairs = _hungarian_pure(cost_matrix)

    assignments: dict[int, object] = {}
    assigned_agents: set[int] = set()

    for t_idx, a_idx in pairs:
        if t_idx < len(tasks) and a_idx < len(agents):
            agent = agents[a_idx]
            if agent.id not in assigned_agents:
                assignments[tasks[t_idx].id] = agent
                assigned_agents.add(agent.id)

    return assignments


# ══════════════════════════════════════════════════════════════════════════════
# Auction-based assignment (scalable, decentralised bids)
# ══════════════════════════════════════════════════════════════════════════════

def auction_assign(
    tasks: list,
    agents: list,
    env,
    rounds: int = 3,
) -> dict[int, object]:
    """
    Iterative auction: tasks broadcast, agents submit bids, highest-priority
    task is awarded to the lowest bidder. Repeat for remaining tasks.

    Each bid = assignment_cost (lower cost = better bid = preferred).
    """
    assignments: dict[int, object] = {}
    available_agents = list(agents)

    # Sort tasks by priority descending so critical tasks are auctioned first
    remaining_tasks = sorted(tasks, key=lambda t: -t.priority_score)

    for task in remaining_tasks:
        if not available_agents:
            break

        bids = [(assignment_cost(agent, task, env), agent) for agent in available_agents]
        bids.sort(key=lambda b: b[0])

        winner = bids[0][1]
        assignments[task.id] = winner
        available_agents.remove(winner)

    return assignments


# ══════════════════════════════════════════════════════════════════════════════
# Predictive charging scheduler
# ══════════════════════════════════════════════════════════════════════════════

class ChargingScheduler:
    """
    Manages charging reservations and predictive decisions.

    Decision logic per agent:
      1. Battery ≤ CRITICAL → force charge immediately.
      2. Predicted battery after current task < BATTERY_PREDICTIVE → schedule
         charge before accepting next task.
      3. Many agents already charging → defer unless critical.
      4. Reserve charger slot ahead of time (CHARGE_RESERVE_WINDOW ticks).
    """

    def __init__(self, charging_stations: list):
        self.stations = charging_stations

        # {station_id: [(agent_id, reserved_tick)]}
        self._reservations: dict[int, list[tuple[int, int]]] = defaultdict(list)

        # Environment reference (set later)
        self._env = None

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _count_charging(self, agents: list) -> int:
        """Count agents currently charging or heading to charge."""
        from agent import AgentState
        return sum(
            1 for a in agents
            if a.state in (AgentState.CHARGING, AgentState.TO_CHARGER)
        )

    def _station_queue_depth(self, station_id: int, tick: int) -> int:
        """How many agents are reserved for this station in the near future."""
        window = [
            r for r in self._reservations[station_id]
            if tick <= r[1] <= tick + CHARGE_RESERVE_WINDOW
        ]
        return len(window)

    def _best_available_station(self, agent, tick: int) -> Optional[object]:
        """
        Return the best station based on:
        - lowest queue depth
        - shortest distance
        - stable tie-break using station.id
        """
        env = self._env

        candidates = []
        for station in self.stations:
            qd = self._station_queue_depth(station.id, tick)
            if qd >= CHARGE_MAX_QUEUE:
                continue

            dist = env.manhattan(agent.location, station.location) if env else 0
            candidates.append((qd, dist, station))

        if not candidates:
            # All queues full; pick nearest anyway
            candidates = [
                (0, env.manhattan(agent.location, s.location) if env else 0, s)
                for s in self.stations
            ]

        # FIX: sort only by queue depth + distance + station.id
        # Avoid comparing ChargingStation objects directly
        candidates.sort(key=lambda x: (x[0], x[1], x[2].id))

        return candidates[0][2] if candidates else None

    # ── Main API ────────────────────────────────────────────────────────────────

    def set_env(self, env) -> None:
        self._env = env

    def predict_battery_after_task(self, agent, task) -> float:
        """Estimate remaining battery after completing the current task."""
        env = self._env
        if task is None or env is None:
            return agent.battery

        pickup_dist = env.manhattan(agent.location, task.pickup)
        total_dist  = pickup_dist + task.distance
        return agent.battery - total_dist * agent.drain_per_step

    def should_charge_now(
        self,
        agent,
        agents: list,
        tick: int,
        has_critical: bool,
    ) -> bool:
        """Return True if the agent should divert to a charger right now."""
        from agent import AgentState

        # Already charging
        if agent.state in (AgentState.CHARGING, AgentState.TO_CHARGER):
            return False

        # Force charge if critically low
        if agent.critically_low():
            return True

        # If a CRITICAL task is pending and robot has enough battery, keep working
        if has_critical and agent.battery > BATTERY_CRITICAL * 2:
            return False

        # Too many robots already charging (avoid charging paralysis)
        charging_count = self._count_charging(agents)
        max_simultaneous = max(1, len(self.stations))
        if charging_count >= max_simultaneous and not agent.critically_low():
            return False

        # Predictive: would we run out on the current task?
        if agent.current_task:
            pred = self.predict_battery_after_task(agent, agent.current_task)
            if pred < BATTERY_PREDICTIVE:
                return True

        # Q-learner decision
        if hasattr(agent, "ql"):
            return agent.ql.decide_charge(
                battery      = agent.battery,
                queue_len    = 0,
                has_critical = has_critical,
                is_working   = agent.current_task is not None,
            )

        return agent.needs_charge()

    def reserve_charger(self, agent, tick: int) -> Optional[object]:
        """
        Reserve a charger for `agent`, returning the station object.
        Records the reservation for queue management.
        """
        station = self._best_available_station(agent, tick)
        if station:
            self._reservations[station.id].append((agent.id, tick))
        return station

    def release_reservation(self, agent_id: int) -> None:
        """Remove all reservations for this agent."""
        for sid in list(self._reservations.keys()):
            self._reservations[sid] = [
                r for r in self._reservations[sid] if r[0] != agent_id
            ]

    def expire_old_reservations(self, tick: int) -> None:
        """Purge reservations older than the reserve window."""
        cutoff = tick - CHARGE_RESERVE_WINDOW
        for sid in list(self._reservations.keys()):
            self._reservations[sid] = [
                r for r in self._reservations[sid] if r[1] >= cutoff
            ]

    def charging_summary(self) -> dict:
        return {sid: len(rlist) for sid, rlist in self._reservations.items()}