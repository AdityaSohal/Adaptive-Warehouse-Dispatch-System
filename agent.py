"""
agent.py v4 — MAPPO-guided A* pathfinding.

Key change over v3:
  MAPPO now ACTUALLY AFFECTS MOVEMENT via cost-shaping.

  How it works:
    After each MAPPO update, the policy produces a "congestion bias vector"
    per agent — a learned 22×22 cell-cost overlay (mappo_cost_overlay).
    This overlay is added as extra edge weight in A*, on top of the static
    congestion_cost and heatmap penalties.

    Result: agents that have been penalised for entering a certain region
    (via PENALTY_CONGESTION rewards) will learn to route around it. The
    overlay is updated every rollout via MAPPOPolicy.compute_cost_overlay().

  This fixes the structural bug where MAPPO was training in a vacuum —
  its actions (directions 0-4) were ignored and A* made all movement
  decisions. Now the two systems are connected: MAPPO shapes the cost
  landscape, A* finds the optimal path within that landscape.

Other fixes:
  - Priority-aware yielding: LOW-priority agents yield to higher-priority ones.
  - wait_counter reset on successful step (was never reset before, causing
    false deadlock detections).
"""

from __future__ import annotations

import heapq
import random
from collections import defaultdict
from enum import Enum, auto

import numpy as np

from config import (
    FAST_SPEED, HEAVY_SPEED,
    FAST_CAPACITY, HEAVY_CAPACITY,
    FAST_DRAIN, HEAVY_DRAIN,
    CHARGE_RATE, BATTERY_FULL,
    BATTERY_LOW, BATTERY_CRITICAL, BATTERY_RESUME,
    SPEC_THRESHOLD, AGENT_LERP,
)
from rl_engine import AgentQLearner


# ── Enums ───────────────────────────────────────────────────────────────────────

class AgentRole(Enum):
    FAST  = "fast"
    HEAVY = "heavy"

class AgentState(Enum):
    IDLE       = auto()
    TO_PICKUP  = auto()
    TO_DROP    = auto()
    TO_CHARGER = auto()
    CHARGING   = auto()
    WAITING    = auto()


# ── Agent ───────────────────────────────────────────────────────────────────────

class Agent:

    def __init__(self, agent_id: int, x: int, y: int, role: AgentRole):
        self.id    = agent_id
        self.x     = x
        self.y     = y
        self.role  = role

        # Physics
        self.speed    = FAST_SPEED    if role == AgentRole.FAST else HEAVY_SPEED
        self.capacity = FAST_CAPACITY if role == AgentRole.FAST else HEAVY_CAPACITY
        self.drain    = FAST_DRAIN    if role == AgentRole.FAST else HEAVY_DRAIN

        # Pixel position for smooth rendering
        self.px: float = 0.0
        self.py: float = 0.0

        # Battery
        self.battery         = 65.0 + random.random() * 35.0
        self.drain_per_step  = self.drain

        # Task state
        self.state        = AgentState.IDLE
        self.current_task = None
        self.path: list   = []
        self.wait_counter = 0

        # Carrying indicator (for rendering)
        self.carrying = False

        # Local congestion memory {(x,y): penalty}
        self.memory_map: dict = {}

        # ── MAPPO cost overlay ───────────────────────────────────────────────
        # A learned per-cell extra cost, updated by MAPPOPolicy after each
        # rollout.  Shape: dict[(x,y)] -> float  (sparse; 0.0 if not present)
        # A* reads this via mappo_extra_cost(pos).
        self.mappo_overlay: dict[tuple, float] = {}

        # Performance / specialisation
        self.tasks_completed = 0
        self.total_reward    = 0.0
        self.perf_history:   list[float]   = []
        self.spec_scores     = defaultdict(float)
        self.spec_counts     = defaultdict(int)
        self.specialization  = "generalist"

        # RL learner
        self.ql = AgentQLearner()

    # ── Location helpers ────────────────────────────────────────────────────────

    @property
    def location(self) -> tuple:
        return (self.x, self.y)

    def move_to(self, x: int, y: int) -> None:
        self.x, self.y = x, y

    def lerp_pixel(self, cell_px: float, cell_py: float) -> None:
        self.px += (cell_px - self.px) * AGENT_LERP
        self.py += (cell_py - self.py) * AGENT_LERP

    # ── Battery ─────────────────────────────────────────────────────────────────

    def needs_charge(self) -> bool:
        return self.battery <= BATTERY_LOW

    def critically_low(self) -> bool:
        return self.battery <= BATTERY_CRITICAL

    def is_full(self) -> bool:
        return self.battery >= BATTERY_RESUME

    def drain_battery(self) -> None:
        prev = self.battery
        self.battery = max(0.0, self.battery - self.drain_per_step)
        if prev > 0 and self.battery == 0:
            self.ql.reward_flat_battery(0, 0, False, False)

    def charge_tick(self) -> None:
        self.battery = min(BATTERY_FULL, self.battery + CHARGE_RATE)

    # ── MAPPO overlay helpers ───────────────────────────────────────────────────

    def mappo_extra_cost(self, pos: tuple) -> float:
        """
        Returns the MAPPO-learned extra cost for stepping into `pos`.
        Called by A* as an additional edge weight.
        Clamped to [0, 5] to avoid overwhelming the base heuristic.
        """
        return min(5.0, self.mappo_overlay.get(pos, 0.0))

    def update_mappo_overlay(self, overlay: dict[tuple, float]) -> None:
        """
        Called by MAPPOPolicy.apply_overlay_to_agents() after each update.
        Merges new overlay into existing one with EMA (alpha=0.3) so
        the cost landscape changes smoothly, not abruptly.
        """
        alpha = 0.3
        for pos, cost in overlay.items():
            old = self.mappo_overlay.get(pos, 0.0)
            self.mappo_overlay[pos] = (1 - alpha) * old + alpha * cost
        # Decay all existing entries slightly so old penalties fade
        for pos in list(self.mappo_overlay.keys()):
            self.mappo_overlay[pos] *= 0.98
            if self.mappo_overlay[pos] < 0.01:
                del self.mappo_overlay[pos]

    # ── Priority ────────────────────────────────────────────────────────────────

    @property
    def task_priority(self) -> int:
        """Returns current task priority (1-5), or 0 if idle."""
        if self.current_task:
            return getattr(self.current_task, 'priority_score', 1)
        return 0

    # ── A* pathfinding ──────────────────────────────────────────────────────────

    def compute_path(
        self,
        start:     tuple,
        goal:      tuple,
        grid_size: int,
        obstacles: set,
        reserved:  dict | None = None,
    ) -> list:
        """
        Space-time A* from start → goal.

        Edge cost = 1 + congestion_memory + MAPPO_overlay
        The MAPPO overlay is the key new addition — the RL policy's learned
        penalties feed directly into pathfinding cost here.

        Falls back to greedy Manhattan if no path found.
        """

        def h(a: tuple, b: tuple) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def neighbours(pos: tuple, t: int) -> list:
            x, y = pos
            result = []
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < grid_size and 0 <= ny < grid_size):
                    continue
                if (nx, ny) in obstacles and (nx, ny) != goal:
                    continue
                if reserved:
                    t1 = t + 1
                    if (nx, ny, t1) in reserved and reserved[(nx, ny, t1)] != self.id:
                        continue
                    if (nx, ny, t) in reserved and (x, y, t1) in reserved:
                        if (reserved.get((nx, ny, t)) not in (None, self.id)
                                and reserved.get((x, y, t1)) not in (None, self.id)):
                            continue
                result.append((nx, ny))
            return result

        open_heap = [(h(start, goal), 0.0, 0, start, [])]
        visited:  dict = {}

        while open_heap:
            _, cost, t, pos, path = heapq.heappop(open_heap)

            if pos == goal:
                self.path = path + [goal]
                return self.path

            state_key = (pos, min(t, 25))
            if state_key in visited and visited[state_key] <= cost:
                continue
            visited[state_key] = cost

            for npos in neighbours(pos, t):
                # Base cost
                extra = self.memory_map.get(npos, 0.0)
                # ── MAPPO cost-shaping injected here ──────────────────────
                extra += self.mappo_extra_cost(npos)
                # ──────────────────────────────────────────────────────────
                new_cost = cost + 1.0 + extra
                heapq.heappush(
                    open_heap,
                    (new_cost + h(npos, goal), new_cost, t + 1, npos, path + [npos]),
                )

        # Fallback
        self.path = self._manhattan_fallback(start, goal, grid_size, obstacles)
        return self.path

    def _manhattan_fallback(
        self, start: tuple, goal: tuple, grid_size: int, obstacles: set
    ) -> list:
        path: list = []
        x, y   = start
        tx, ty = goal
        for _ in range(grid_size * 3):
            if x == tx and y == ty:
                break
            options = []
            if x < tx: options.append((x + 1, y))
            if x > tx: options.append((x - 1, y))
            if y < ty: options.append((x, y + 1))
            if y > ty: options.append((x, y - 1))
            moved = False
            for opt in options:
                if (opt not in obstacles
                        and 0 <= opt[0] < grid_size
                        and 0 <= opt[1] < grid_size):
                    x, y = opt
                    path.append((x, y))
                    moved = True
                    break
            if not moved:
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in obstacles and 0 <= nx < grid_size and 0 <= ny < grid_size:
                        x, y = nx, ny
                        path.append((x, y))
                        break
                else:
                    path.append((x, y))
        return path

    def step_path(self) -> tuple:
        """Advance one step; drain battery; reset wait counter; return new location."""
        if self.path:
            next_pos = self.path.pop(0)
            self.move_to(*next_pos)
            self.drain_battery()
            self.wait_counter = 0   # ← BUG FIX: was never reset on successful move
            return self.location

    # ── Memory ──────────────────────────────────────────────────────────────────

    def update_memory(self, pos: tuple, congestion_cost: float) -> None:
        old = self.memory_map.get(pos, 0.0)
        self.memory_map[pos] = 0.8 * old + 0.2 * congestion_cost

    # ── Specialisation / reward ─────────────────────────────────────────────────

    def record_task(self, reward: float, category: str, queue_len: int,
                    has_critical: bool) -> None:
        self.tasks_completed += 1
        self.total_reward    += reward
        self.perf_history.append(reward)
        if len(self.perf_history) > 100:
            self.perf_history.pop(0)
        self.spec_scores[category] += reward
        self.spec_counts[category] += 1
        self._update_specialization()
        self.ql.reward_delivery(
            priority_val = int(getattr(self.current_task, 'priority_score', 2)) if self.current_task else 2,
            battery      = self.battery,
            queue_len    = queue_len,
            has_critical = has_critical,
            is_working   = False,
        )

    def _update_specialization(self) -> None:
        eligible = {
            cat: self.spec_scores[cat] / self.spec_counts[cat]
            for cat in ('short', 'long', 'heavy')
            if self.spec_counts[cat] >= SPEC_THRESHOLD
        }
        if not eligible:
            self.specialization = "generalist"
            return
        best = max(eligible, key=eligible.get)
        self.specialization = {
            'short': 'runner', 'long': 'hauler', 'heavy': 'lifter'
        }[best]

    @property
    def avg_reward(self) -> float:
        if not self.perf_history:
            return 0.0
        return sum(self.perf_history) / len(self.perf_history)

    def __repr__(self) -> str:
        return (
            f"Agent({self.id},{self.role.value}) "
            f"pos={self.location} batt={self.battery:.0f}% "
            f"state={self.state.name}"
        )
