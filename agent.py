"""
agent.py
Each agent has: A* pathfinding, local memory map, battery management,
collision reservation, and a dynamically assigned role.
"""

import heapq
import random
from collections import defaultdict
from enum import Enum, auto


class AgentRole(Enum):
    FAST = "fast"
    HEAVY = "heavy"


class AgentState(Enum):
    IDLE = auto()
    TO_PICKUP = auto()
    TO_DROP = auto()
    TO_CHARGER = auto()
    CHARGING = auto()
    WAITING = auto()       # deadlock / collision hold


class Agent:
    BATTERY_DRAIN_PER_STEP = 1.2   # fast agents
    HEAVY_DRAIN_PER_STEP   = 0.8   # heavy agents move slower, drain less per step
    CHARGE_RATE            = 8.0   # % per tick while charging
    LOW_BATTERY_THRESHOLD  = 25.0
    FULL_BATTERY           = 100.0

    ROLE_THRESHOLD = 6  # tasks before role locks in

    def __init__(self, agent_id: int, x: int, y: int, role: AgentRole):
        self.id          = agent_id
        self.x           = x
        self.y           = y
        self.role        = role

        # Physics
        self.speed = 2.0 if role == AgentRole.FAST else 1.0
        self.capacity = 15.0 if role == AgentRole.FAST else 40.0

        # Battery
        self.battery = self.FULL_BATTERY
        self.drain   = self.BATTERY_DRAIN_PER_STEP if role == AgentRole.FAST else self.HEAVY_DRAIN_PER_STEP

        # Task state
        self.state          = AgentState.IDLE
        self.current_task   = None
        self.path           = []          # list of (x,y) waypoints from A*
        self.wait_counter   = 0

        # Local memory: remembers congestion costs it has observed
        # {(x,y): cost_penalty}  — shared read from environment but privately cached
        self.memory_map: dict = {}

        # Performance tracking
        self.tasks_completed    = 0
        self.total_reward       = 0.0
        self.performance_history= []
        self.spec_scores        = defaultdict(float)
        self.spec_counts        = defaultdict(int)
        self.specialization     = "generalist"

        # Pathfinding
        self._grid_size  = None
        self._obstacles  = None

    # ──────────────────────────────────────────
    # Location helpers
    # ──────────────────────────────────────────
    @property
    def location(self):
        return (self.x, self.y)

    def move_to(self, x: int, y: int):
        self.x, self.y = x, y

    # ──────────────────────────────────────────
    # Battery
    # ──────────────────────────────────────────
    def needs_charge(self) -> bool:
        return self.battery <= self.LOW_BATTERY_THRESHOLD

    def is_full(self) -> bool:
        return self.battery >= self.FULL_BATTERY

    def drain_battery(self):
        self.battery = max(0.0, self.battery - self.drain)

    def charge_tick(self):
        self.battery = min(self.FULL_BATTERY, self.battery + self.CHARGE_RATE)

    # ──────────────────────────────────────────
    # A* pathfinding
    # ──────────────────────────────────────────
    def compute_path(self, start: tuple, goal: tuple,
                     grid_size: int, obstacles: set,
                     reserved: dict = None) -> list:
        """
        A* from start to goal on a Manhattan grid.
        obstacles: set of (x,y) cells that are walls / shelves.
        reserved:  {(x,y,t): agent_id} — space-time reservations for collision avoidance.
        Returns list of (x,y) cells from start (exclusive) to goal (inclusive).
        """
        self._grid_size = grid_size
        self._obstacles = obstacles

        def h(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        def neighbors(pos, t):
            x, y = pos
            dirs = [(0,1),(0,-1),(1,0),(-1,0),(0,0)]  # include wait-in-place
            result = []
            for dx, dy in dirs:
                nx, ny = x+dx, y+dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if (nx, ny) not in obstacles:
                        # Check space-time reservation
                        if reserved and (nx, ny, t+1) in reserved and reserved[(nx,ny,t+1)] != self.id:
                            continue
                        # Swap conflict: agents crossing each other
                        if reserved and (nx, ny, t) in reserved and (x, y, t+1) in reserved:
                            if reserved.get((nx,ny,t)) != self.id and reserved.get((x,y,t+1)) != self.id:
                                continue
                        result.append((nx, ny))
            return result

        open_heap = []
        heapq.heappush(open_heap, (h(start, goal), 0, start, []))
        visited = {}

        while open_heap:
            est, cost, pos, path = heapq.heappop(open_heap)
            t = cost

            if pos == goal:
                self.path = path + [goal]
                return self.path

            state_key = (pos, min(t, 20))  # cap time for memory
            if state_key in visited and visited[state_key] <= cost:
                continue
            visited[state_key] = cost

            for npos in neighbors(pos, t):
                # Use memory map for congestion cost
                extra = self.memory_map.get(npos, 0.0)
                move_cost = 1 + extra
                new_cost = cost + move_cost
                est_total = new_cost + h(npos, goal)
                heapq.heappush(open_heap, (est_total, new_cost, npos, path + [npos]))

        # No path found — return direct Manhattan path as fallback
        fallback = self._manhattan_fallback(start, goal, grid_size, obstacles)
        self.path = fallback
        return fallback

    def _manhattan_fallback(self, start, goal, grid_size, obstacles):
        """Simple greedy fallback if A* finds nothing."""
        path = []
        x, y = start
        tx, ty = goal
        for _ in range(grid_size * 2):
            if x == tx and y == ty:
                break
            options = []
            if x < tx: options.append((x+1, y))
            if x > tx: options.append((x-1, y))
            if y < ty: options.append((x, y+1))
            if y > ty: options.append((x, y-1))
            moved = False
            for opt in options:
                if opt not in obstacles and 0 <= opt[0] < grid_size and 0 <= opt[1] < grid_size:
                    x, y = opt
                    path.append((x, y))
                    moved = True
                    break
            if not moved:
                path.append((x, y))  # wait in place
        return path

    def step_path(self) -> tuple:
        """
        Advance one step along the computed path.
        Returns the next cell, or current cell if path is empty.
        """
        if self.path:
            next_pos = self.path.pop(0)
            self.move_to(*next_pos)
            self.drain_battery()
        return self.location

    def update_memory(self, pos: tuple, congestion_cost: float):
        """Learn the congestion cost for a cell."""
        # Exponential moving average
        old = self.memory_map.get(pos, 0.0)
        self.memory_map[pos] = 0.8 * old + 0.2 * congestion_cost

    # ──────────────────────────────────────────
    # Performance / specialization
    # ──────────────────────────────────────────
    def record_task(self, reward: float, category: str):
        self.tasks_completed += 1
        self.total_reward    += reward
        self.performance_history.append(reward)
        self.spec_scores[category] += reward
        self.spec_counts[category] += 1
        self._update_specialization()

    def _update_specialization(self):
        eligible = {
            cat: self.spec_scores[cat] / self.spec_counts[cat]
            for cat in ['short', 'long', 'heavy']
            if self.spec_counts[cat] >= self.ROLE_THRESHOLD
        }
        if not eligible:
            self.specialization = "generalist"
            return
        best = max(eligible, key=eligible.get)
        self.specialization = {'short': 'speed_runner',
                               'long':  'long_hauler',
                               'heavy': 'heavy_lifter'}[best]

    @property
    def avg_reward(self):
        if not self.performance_history:
            return 0.0
        return sum(self.performance_history) / len(self.performance_history)

    def __repr__(self):
        return (f"Agent({self.id},{self.role.value}) "
                f"pos={self.location} batt={self.battery:.0f}% "
                f"state={self.state.name}")
