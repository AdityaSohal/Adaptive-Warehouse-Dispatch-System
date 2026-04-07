"""
environment.py
20×20+ grid with:
  • Physical shelf obstacles (A* routes around them)
  • Space-time reservation table for collision prevention
  • Deadlock detection and yield-and-replan resolution
  • 3 charging stations with smart pre-reservation
  • Dynamic congestion zones that shift every N ticks
  • Inventory tracking per pickup shelf
"""

from __future__ import annotations

import random
from collections import defaultdict

from agent import Agent, AgentRole, AgentState
from task import Task
from config import (
    GRID_SIZE, NUM_PICKUP_ZONES, NUM_DROP_ZONES, NUM_CHARGING,
    NUM_SHELF_ROWS, SHELF_ROW_LEN,
    NUM_CONGESTION, DYNAMIC_CONGESTION, CONGESTION_REFRESH,
    DEADLOCK_WAIT, RESERVATION_HORIZON,
    NUM_FAST_AGENTS, NUM_HEAVY_AGENTS,
)


class ChargingStation:
    def __init__(self, x: int, y: int, station_id: int):
        self.id          = station_id
        self.location    = (x, y)
        self.x, self.y   = x, y
        self.occupied_by: int | None = None

    @property
    def is_free(self) -> bool:
        return self.occupied_by is None


class WarehouseEnvironment:
    def __init__(
        self,
        grid_size:       int  = GRID_SIZE,
        num_pickup:      int  = NUM_PICKUP_ZONES,
        num_drop:        int  = NUM_DROP_ZONES,
        num_congestion:  int  = NUM_CONGESTION,
        dynamic_traffic: bool = DYNAMIC_CONGESTION,
        seed:            int | None = None,
    ):
        if seed is not None:
            random.seed(seed)

        self.grid_size       = grid_size
        self.dynamic_traffic = dynamic_traffic
        self.tick            = 0

        # ── Build fixed layout ──────────────────────────────────────────────────
        all_cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
        random.shuffle(all_cells)
        idx = 0

        self.pickup_zones: list[tuple]  = [tuple(c) for c in all_cells[idx:idx+num_pickup]];  idx += num_pickup
        self.drop_zones:   list[tuple]  = [tuple(c) for c in all_cells[idx:idx+num_drop]];    idx += num_drop
        _charge_locs:      list[tuple]  = [tuple(c) for c in all_cells[idx:idx+NUM_CHARGING]]; idx += NUM_CHARGING

        self.charging_stations: list[ChargingStation] = [
            ChargingStation(x, y, i + 1) for i, (x, y) in enumerate(_charge_locs)
        ]

        # ── Shelf obstacles ─────────────────────────────────────────────────────
        # Place shelf rows in the interior (clear of zones & chargers)
        reserved_cells: set[tuple] = (
            set(self.pickup_zones)
            | set(self.drop_zones)
            | {s.location for s in self.charging_stations}
        )
        self.obstacles: set[tuple] = set()
        self._place_shelves(reserved_cells)

        # ── Inventory: each pickup zone starts with 3–6 items ──────────────────
        self.inventory: dict[tuple, int] = {p: random.randint(3, 6) for p in self.pickup_zones}

        # ── Congestion zones ────────────────────────────────────────────────────
        free = [c for c in all_cells if tuple(c) not in reserved_cells and tuple(c) not in self.obstacles]
        random.shuffle(free)
        self.congestion_zones: set[tuple] = set(tuple(c) for c in free[:num_congestion])
        self.congestion_cost:  dict[tuple, float] = {
            c: random.uniform(0.5, 2.5) for c in self.congestion_zones
        }

        # ── Agents ─────────────────────────────────────────────────────────────
        remaining = [tuple(c) for c in all_cells[idx:] if tuple(c) not in self.obstacles]
        random.shuffle(remaining)
        self.agents: list[Agent] = []
        n_agents = NUM_FAST_AGENTS + NUM_HEAVY_AGENTS
        agent_starts = remaining[:n_agents]

        from agent import AgentRole
        for i in range(NUM_FAST_AGENTS):
            x, y = agent_starts[i]
            self.agents.append(Agent(i + 1, x, y, AgentRole.FAST))
        for i in range(NUM_HEAVY_AGENTS):
            x, y = agent_starts[NUM_FAST_AGENTS + i]
            self.agents.append(Agent(NUM_FAST_AGENTS + i + 1, x, y, AgentRole.HEAVY))

        # ── Space-time reservation table  {(x,y,t): agent_id} ─────────────────
        self.reservations: dict = {}

        # ── Deadlock tracking ───────────────────────────────────────────────────
        self.wait_counts: dict[int, int] = defaultdict(int)

        # ── Metrics ─────────────────────────────────────────────────────────────
        self.collision_events   = 0
        self.deadlocks_resolved = 0
        self.charges_completed  = 0
        self.reward_history:    list[float] = []
        self.delivery_times:    list[float] = []

    # ── Shelf placement ─────────────────────────────────────────────────────────
    def _place_shelves(self, reserved: set[tuple]) -> None:
        """Place NUM_SHELF_ROWS horizontal shelf rows in the interior."""
        gs = self.grid_size
        step_y = max(3, gs // (NUM_SHELF_ROWS + 1))
        for row in range(NUM_SHELF_ROWS):
            y = step_y * (row + 1)
            start_x = random.randint(2, max(2, gs - SHELF_ROW_LEN - 2))
            for dx in range(SHELF_ROW_LEN):
                cell = (start_x + dx, y)
                if cell not in reserved:
                    self.obstacles.add(cell)
            # Add a second parallel row shifted one cell
            y2 = y + 1
            if y2 < gs:
                for dx in range(SHELF_ROW_LEN):
                    cell = (start_x + dx, y2)
                    if cell not in reserved:
                        self.obstacles.add(cell)

    # ── Grid helpers ────────────────────────────────────────────────────────────
    def manhattan(self, a: tuple, b: tuple) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def congestion_penalty(self, start: tuple, end: tuple) -> float:
        penalty = 0.0
        x, y = start
        tx, ty = end
        sx = 1 if tx > x else -1
        while x != tx:
            x += sx
            if (x, y) in self.congestion_zones:
                penalty += self.congestion_cost.get((x, y), 0)
        sy = 1 if ty > y else -1
        while y != ty:
            y += sy
            if (x, y) in self.congestion_zones:
                penalty += self.congestion_cost.get((x, y), 0)
        return penalty

    def refresh_congestion(self) -> None:
        n  = max(1, len(self.congestion_zones) // 4)
        existing = list(self.congestion_zones)
        random.shuffle(existing)
        for cell in existing[:n]:
            self.congestion_zones.discard(cell)
            self.congestion_cost.pop(cell, None)
        available = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
            if (x, y) not in self.congestion_zones
            and (x, y) not in self.pickup_zones
            and (x, y) not in self.drop_zones
            and (x, y) not in self.obstacles
        ]
        for cell in random.sample(available, min(n, len(available))):
            cell = tuple(cell)
            self.congestion_zones.add(cell)
            self.congestion_cost[cell] = random.uniform(0.5, 2.5)

    # ── Cost / reward ────────────────────────────────────────────────────────────
    def compute_cost(self, agent: Agent, task: Task) -> float:
        dist    = self.manhattan(agent.location, task.pickup) + task.distance
        time_   = dist / max(agent.speed, 0.1)
        load    = task.weight / max(agent.capacity, 1)
        cong    = (self.congestion_penalty(agent.location, task.pickup)
                   + self.congestion_penalty(task.pickup, task.drop))
        pbonus  = (task.priority_score - 1) * 1.0
        urgency = task.urgency_ratio * 2.0
        return max(0.1, time_ + load + cong - pbonus + urgency)

    def compute_reward(self, agent: Agent, task: Task) -> float:
        base     = task.priority_score * 2.5
        time_rem = max(0, task.time_remaining / 10.0)
        load_pen = task.weight / max(agent.capacity, 1)
        return max(0.1, base + time_rem - load_pen)

    # ── Collision detection & prevention ────────────────────────────────────────
    def reserve_path(self, agent: Agent) -> bool:
        for t, cell in enumerate(agent.path[:RESERVATION_HORIZON]):
            key = (cell[0], cell[1], self.tick + t)
            if key in self.reservations and self.reservations[key] != agent.id:
                return False
            self.reservations[key] = agent.id
        return True

    def clear_reservations(self, agent: Agent) -> None:
        keys = [k for k, v in self.reservations.items() if v == agent.id]
        for k in keys:
            del self.reservations[k]

    def detect_collision(self, agent: Agent, next_pos: tuple) -> bool:
        return any(
            a.id != agent.id and a.location == next_pos
            for a in self.agents
        )

    def resolve_collision(self, agent: Agent) -> None:
        agent.state        = AgentState.WAITING
        agent.wait_counter += 1
        self.wait_counts[agent.id] += 1
        self.collision_events += 1

    # ── Deadlock detection & resolution ─────────────────────────────────────────
    def check_deadlocks(self) -> None:
        for agent in self.agents:
            if agent.state == AgentState.WAITING:
                if self.wait_counts[agent.id] >= DEADLOCK_WAIT:
                    self.clear_reservations(agent)
                    agent.wait_counter = 0
                    self.wait_counts[agent.id] = 0
                    self.deadlocks_resolved += 1
                    self._yield_agent(agent)

    def _yield_agent(self, agent: Agent) -> None:
        x, y = agent.location
        options = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        random.shuffle(options)
        for pos in options:
            ox, oy = pos
            if (0 <= ox < self.grid_size and 0 <= oy < self.grid_size
                    and pos not in self.obstacles
                    and not self.detect_collision(agent, pos)):
                agent.move_to(ox, oy)
                agent.drain_battery()
                if agent.current_task:
                    task = agent.current_task
                    goal = (task.pickup
                            if agent.state in (AgentState.TO_PICKUP, AgentState.WAITING)
                            else task.drop)
                    agent.compute_path(
                        agent.location, goal,
                        self.grid_size, self.obstacles,
                        self.reservations,
                    )
                    self.reserve_path(agent)
                agent.state = (AgentState.TO_PICKUP
                               if agent.current_task and not agent.carrying
                               else AgentState.TO_DROP if agent.current_task
                               else AgentState.IDLE)
                break

    # ── Charging ─────────────────────────────────────────────────────────────────
    def nearest_free_charger(self, agent: Agent) -> ChargingStation | None:
        free = [s for s in self.charging_stations
                if s.is_free or s.occupied_by == agent.id]
        if not free:
            return None
        return min(free, key=lambda s: self.manhattan(agent.location, s.location))

    def occupy_charger(self, agent: Agent, station: ChargingStation) -> None:
        station.occupied_by = agent.id
        agent.state         = AgentState.CHARGING

    def release_charger(self, agent: Agent) -> None:
        for s in self.charging_stations:
            if s.occupied_by == agent.id:
                s.occupied_by = None
        agent.state = AgentState.IDLE
        self.charges_completed += 1
        agent.ql.reward_charge_completed(
            agent.battery, 0, False, False
        )

    # ── Task execution ───────────────────────────────────────────────────────────
    def simulate_task_step(self, agent: Agent) -> float | None:
        if agent.state == AgentState.WAITING:
            agent.wait_counter += 1
            if agent.wait_counter > 3:
                agent.state       = (AgentState.TO_PICKUP
                                     if agent.current_task
                                     and not agent.carrying
                                     else AgentState.TO_DROP)
                agent.wait_counter = 0
            return None

        task = agent.current_task
        if task is None:
            return None

        goal = task.pickup if agent.state == AgentState.TO_PICKUP else task.drop

        if not agent.path:
            agent.compute_path(
                agent.location, goal,
                self.grid_size, self.obstacles,
                self.reservations,
            )
            self.reserve_path(agent)

        if agent.path:
            next_pos = agent.path[0]
            if next_pos in self.congestion_zones:
                agent.update_memory(next_pos, self.congestion_cost.get(next_pos, 0))
            if self.detect_collision(agent, next_pos):
                self.resolve_collision(agent)
                return None
            agent.step_path()

        # Arrival checks
        if agent.location == task.pickup and agent.state == AgentState.TO_PICKUP:
            agent.state    = AgentState.TO_DROP
            agent.carrying = True
            task.start_transit()
            agent.path = []
            # Consume inventory
            inv = self.inventory.get(task.pickup, 0)
            if inv > 0:
                self.inventory[task.pickup] = inv - 1

        elif agent.location == task.drop and agent.state == AgentState.TO_DROP:
            reward = self.compute_reward(agent, task)
            task.complete(reward)
            agent.carrying       = False
            agent.current_task   = None
            agent.state          = AgentState.IDLE
            agent.record_task(reward, task.category, 0, False)
            self.clear_reservations(agent)
            self.reward_history.append(reward)
            if len(self.reward_history) > 200:
                self.reward_history.pop(0)
            return reward

        return None

    # ── Charger ticking ──────────────────────────────────────────────────────────
    def tick_chargers(self) -> None:
        for station in self.charging_stations:
            if station.occupied_by is not None:
                agent = self.get_agent(station.occupied_by)
                if agent:
                    agent.charge_tick()
                    if agent.is_full():
                        self.release_charger(agent)

    # ── Helpers ──────────────────────────────────────────────────────────────────
    def get_agent(self, agent_id: int) -> Agent | None:
        for a in self.agents:
            if a.id == agent_id:
                return a
        return None

    def occupied_cells(self) -> set[tuple]:
        return {a.location for a in self.agents}

    def get_grid_state(self) -> dict:
        return {
            'grid_size':         self.grid_size,
            'agents':            list(self.agents),
            'pickup_zones':      self.pickup_zones,
            'drop_zones':        self.drop_zones,
            'congestion_zones':  list(self.congestion_zones),
            'charging_stations': self.charging_stations,
            'obstacles':         list(self.obstacles),
            'inventory':         dict(self.inventory),
            'tick':              self.tick,
        }
