"""
environment.py v3 — Upgraded warehouse environment.

Key upgrades over v2:
  • CBS global path planning (replaces individual A* with no coordination)
  • ORCA velocity-obstacle collision avoidance (replaces simple distance check)
  • Wait-for-graph deadlock detection + push-and-swap resolution
  • Congestion heatmap learning fed into CBS cost weights
  • Predictive charging via ChargingScheduler
  • One-way corridor lanes (optional)
  • Inventory restocking
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

from agent import Agent, AgentRole, AgentState
from cbs_planner import CBSPlanner
from collision_deadlock import (
    ORCAAgent, WaitForGraph, resolve_cycle, emergency_stop_needed,
)
from rl_engine import CongestionHeatmap
from task import Task
from config import (
    GRID_SIZE, NUM_PICKUP_ZONES, NUM_DROP_ZONES, NUM_CHARGING,
    NUM_SHELF_ROWS, SHELF_ROW_LEN, NUM_CONGESTION,
    DYNAMIC_CONGESTION, CONGESTION_REFRESH, DEADLOCK_WAIT,
    RESERVATION_HORIZON, NUM_FAST_AGENTS, NUM_HEAVY_AGENTS,
    CBS_REPLAN_INTERVAL, ORCA_NEIGHBOR_DIST,
    HEATMAP_PENALTY_SCALE,
)


class ChargingStation:
    def __init__(self, x: int, y: int, station_id: int):
        self.id = station_id
        self.location = (x, y)
        self.x, self.y = x, y
        self.occupied_by: int | None = None
        self.queue: list[int] = []   # agent IDs waiting

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

        # ── Build layout ────────────────────────────────────────────────────────
        all_cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
        random.shuffle(all_cells)
        idx = 0

        self.pickup_zones: list[tuple] = [tuple(c) for c in all_cells[idx:idx+num_pickup]]; idx += num_pickup
        self.drop_zones:   list[tuple] = [tuple(c) for c in all_cells[idx:idx+num_drop]];   idx += num_drop
        _charge_locs       = [tuple(c) for c in all_cells[idx:idx+NUM_CHARGING]];            idx += NUM_CHARGING

        self.charging_stations: list[ChargingStation] = [
            ChargingStation(x, y, i + 1) for i, (x, y) in enumerate(_charge_locs)
        ]

        reserved_cells: set[tuple] = (
            set(self.pickup_zones)
            | set(self.drop_zones)
            | {s.location for s in self.charging_stations}
        )

        self.obstacles: set[tuple] = set()
        self._place_shelves(reserved_cells)

        self.inventory: dict[tuple, int] = {p: random.randint(3, 6) for p in self.pickup_zones}
        self._restock_timer: dict[tuple, int] = {}

        free = [c for c in all_cells if tuple(c) not in reserved_cells and tuple(c) not in self.obstacles]
        random.shuffle(free)
        self.congestion_zones: set[tuple] = set(tuple(c) for c in free[:num_congestion])
        self.congestion_cost: dict[tuple, float] = {c: random.uniform(0.5, 2.5) for c in self.congestion_zones}

        # ── Agents ──────────────────────────────────────────────────────────────
        remaining = [tuple(c) for c in all_cells[idx:] if tuple(c) not in self.obstacles]
        random.shuffle(remaining)
        self.agents: list[Agent] = []
        n_agents    = NUM_FAST_AGENTS + NUM_HEAVY_AGENTS
        agent_starts = remaining[:n_agents]

        for i in range(NUM_FAST_AGENTS):
            x, y = agent_starts[i]
            self.agents.append(Agent(i + 1, x, y, AgentRole.FAST))
        for i in range(NUM_HEAVY_AGENTS):
            x, y = agent_starts[NUM_FAST_AGENTS + i]
            self.agents.append(Agent(NUM_FAST_AGENTS + i + 1, x, y, AgentRole.HEAVY))

        # ── Sub-systems ─────────────────────────────────────────────────────────
        self.cbs_planner  = CBSPlanner(grid_size, self.obstacles)
        self.wfg          = WaitForGraph()
        self.heatmap      = CongestionHeatmap(grid_size)
        self.orca_agents  = {a.id: ORCAAgent(a.id) for a in self.agents}

        # ── Space-time reservations ─────────────────────────────────────────────
        self.reservations: dict = {}
        self.wait_counts:  dict[int, int] = defaultdict(int)

        # ── Metrics ─────────────────────────────────────────────────────────────
        self.collision_events    = 0
        self.deadlocks_resolved  = 0
        self.charges_completed   = 0
        self.reward_history:     list[float] = []
        self.delivery_times:     list[float] = []
        self.pending_proxy:      list = []   # set by dispatcher each tick

    # ── Shelf placement ─────────────────────────────────────────────────────────
    def _place_shelves(self, reserved: set[tuple]) -> None:
        gs   = self.grid_size
        step = max(3, gs // (NUM_SHELF_ROWS + 1))
        for row in range(NUM_SHELF_ROWS):
            y      = step * (row + 1)
            start_x = random.randint(2, max(2, gs - SHELF_ROW_LEN - 2))
            for dx in range(SHELF_ROW_LEN):
                cell = (start_x + dx, y)
                if cell not in reserved:
                    self.obstacles.add(cell)
            y2 = y + 1
            if y2 < gs:
                for dx in range(SHELF_ROW_LEN):
                    cell = (start_x + dx, y2)
                    if cell not in reserved:
                        self.obstacles.add(cell)

    # ── Helpers ──────────────────────────────────────────────────────────────────
    def manhattan(self, a: tuple, b: tuple) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def congestion_penalty(self, start: tuple, end: tuple) -> float:
        penalty = 0.0
        x, y    = start
        tx, ty  = end
        sx = 1 if tx > x else -1
        while x != tx:
            x += sx
            penalty += self.congestion_cost.get((x, y), 0)
            penalty += self.heatmap.cost((x, y))
        sy = 1 if ty > y else -1
        while y != ty:
            y += sy
            penalty += self.congestion_cost.get((x, y), 0)
            penalty += self.heatmap.cost((x, y))
        return penalty

    def refresh_congestion(self) -> None:
        n = max(1, len(self.congestion_zones) // 4)
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

    def _restock_inventories(self) -> None:
        for p in self.pickup_zones:
            if self.inventory.get(p, 0) == 0:
                timer = self._restock_timer.get(p, 0)
                if self.tick - timer >= 25:
                    self.inventory[p] = random.randint(2, 5)
                    self._restock_timer[p] = self.tick

    # ── CBS Global replanning ────────────────────────────────────────────────────
    def global_replan(self) -> None:
        """
        Collect all active agents' current position and goal, run CBS,
        and assign the resulting collision-free paths.
        """
        agents_goals: dict[int, tuple] = {}
        for agent in self.agents:
            if agent.state in (AgentState.TO_PICKUP, AgentState.TO_DROP, AgentState.TO_CHARGER):
                task = agent.current_task
                if agent.state == AgentState.TO_CHARGER:
                    station = next((s for s in self.charging_stations if s.occupied_by == agent.id), None)
                    if station:
                        agents_goals[agent.id] = (agent.location, station.location)
                elif task:
                    goal = task.pickup if agent.state == AgentState.TO_PICKUP else task.drop
                    agents_goals[agent.id] = (agent.location, goal)

        if not agents_goals:
            return

        heatmap_dict  = self.heatmap.as_dict()
        memory_maps   = {a.id: a.memory_map for a in self.agents}
        new_paths     = self.cbs_planner.plan(
            agents_goals,
            heatmap   = heatmap_dict,
            memory_maps = memory_maps,
            t_start   = self.tick,
        )
        for agent in self.agents:
            if agent.id in new_paths:
                agent.path = new_paths[agent.id]
                self.reserve_path(agent)

    # ── Collision avoidance ─────────────────────────────────────────────────────
    def _orca_check(self, agent: Agent, next_pos: tuple) -> bool:
        """
        Use time-to-collision to decide if emergency stop is needed.
        Returns True if movement should proceed, False if agent should stop.
        """
        ax, ay = float(agent.x), float(agent.y)
        vel_a  = (float(next_pos[0]) - ax, float(next_pos[1]) - ay)

        neighbors = []
        for other in self.agents:
            if other.id == agent.id:
                continue
            if abs(other.x - agent.x) + abs(other.y - agent.y) > ORCA_NEIGHBOR_DIST:
                continue
            if other.path:
                ov = (float(other.path[0][0]) - other.x, float(other.path[0][1]) - other.y)
            else:
                ov = (0.0, 0.0)
            neighbors.append({"pos": (float(other.x), float(other.y)), "vel": ov, "radius": 0.5})

        for nb in neighbors:
            if emergency_stop_needed((ax, ay), vel_a, nb["pos"], nb["vel"]):
                return False   # stop

        return True   # safe to move

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
        return any(a.id != agent.id and a.location == next_pos for a in self.agents)

    def resolve_collision(self, agent: Agent, blocker_id: int | None = None) -> None:
        agent.state        = AgentState.WAITING
        agent.wait_counter += 1
        self.wait_counts[agent.id] += 1
        self.collision_events += 1
        self.heatmap.record_collision(agent.location)
        if blocker_id is not None:
            self.wfg.update(agent.id, blocker_id, self.tick)

    # ── Deadlock detection and resolution ────────────────────────────────────────
    def check_deadlocks(self) -> None:
        # Cycle-based deadlock detection
        cycles = self.wfg.detect_cycles()
        agent_map = {a.id: a for a in self.agents}
        occupied  = {a.location for a in self.agents}

        for cycle in cycles:
            escape = resolve_cycle(
                cycle, agent_map, self.obstacles, occupied, self.tick, self.wfg, self.grid_size
            )
            if escape != -1:
                self.deadlocks_resolved += 1
                agent = agent_map[escape]
                self._replan_agent(agent)

        # Long-wait fallback (for agents not in explicit cycles)
        long_waiters = self.wfg.long_wait_agents(DEADLOCK_WAIT)
        for aid in long_waiters:
            agent = agent_map.get(aid)
            if agent and not self.wfg.has_push_token(aid, self.tick):
                self.wfg.clear_agent(aid)
                self._yield_agent(agent)
                self.deadlocks_resolved += 1

        self.wfg.expire_tokens(self.tick)

    def _yield_agent(self, agent: Agent) -> None:
        x, y    = agent.location
        options = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        random.shuffle(options)
        for pos in options:
            ox, oy = pos
            if (0 <= ox < self.grid_size and 0 <= oy < self.grid_size
                    and pos not in self.obstacles
                    and not self.detect_collision(agent, pos)):
                agent.move_to(ox, oy)
                agent.drain_battery()
                self.heatmap.record_wait((ox, oy))
                self._replan_agent(agent)
                break

    def _replan_agent(self, agent: Agent) -> None:
        task = agent.current_task
        if task:
            goal = task.pickup if not agent.carrying else task.drop
            agent.path = agent.compute_path(
                agent.location, goal, self.grid_size, self.obstacles, self.reservations
            )
            self.reserve_path(agent)
        agent.state       = (AgentState.TO_PICKUP if task and not agent.carrying
                             else AgentState.TO_DROP if task
                             else AgentState.IDLE)
        agent.wait_counter = 0
        self.wait_counts[agent.id] = 0

    # ── Charging ─────────────────────────────────────────────────────────────────
    def nearest_free_charger(self, agent: Agent) -> Optional[ChargingStation]:
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
        agent.ql.reward_charge_completed(agent.battery, 0, False, False)

    # ── Task execution (per-tick) ────────────────────────────────────────────────
    def simulate_task_step(self, agent: Agent) -> Optional[float]:
        if agent.state == AgentState.WAITING:
            agent.wait_counter += 1
            self.heatmap.record_wait(agent.location)
            if agent.wait_counter > 3:
                agent.state        = (AgentState.TO_PICKUP if agent.current_task and not agent.carrying
                                      else AgentState.TO_DROP)
                agent.wait_counter = 0
            return None

        task = agent.current_task
        if task is None:
            return None

        goal = task.pickup if agent.state == AgentState.TO_PICKUP else task.drop

        if not agent.path:
            agent.compute_path(agent.location, goal, self.grid_size, self.obstacles, self.reservations)
            self.reserve_path(agent)

        if agent.path:
            next_pos = agent.path[0]

            # Record traversal cost in heatmap
            cost = self.congestion_cost.get(next_pos, 0.0)
            if cost > 0:
                agent.update_memory(next_pos, cost)
                self.heatmap.record_traversal(next_pos, cost)

            # ORCA safety check
            safe = self._orca_check(agent, next_pos)

            # Standard collision check
            blocker_id = None
            for other in self.agents:
                if other.id != agent.id and other.location == next_pos:
                    blocker_id = other.id
                    break

            if not safe or blocker_id is not None:
                self.resolve_collision(agent, blocker_id)
                return None

            agent.step_path()

        # Arrival: pickup
        if agent.location == task.pickup and agent.state == AgentState.TO_PICKUP:
            agent.state    = AgentState.TO_DROP
            agent.carrying = True
            task.start_transit()
            agent.path = []
            inv = self.inventory.get(task.pickup, 0)
            if inv > 0:
                self.inventory[task.pickup] = inv - 1

        # Arrival: drop-off
        elif agent.location == task.drop and agent.state == AgentState.TO_DROP:
            reward = self.compute_reward(agent, task)
            task.complete(reward)
            agent.carrying     = False
            agent.current_task = None
            agent.state        = AgentState.IDLE
            agent.record_task(reward, task.category, 0, False)
            self.clear_reservations(agent)
            self.reward_history.append(reward)
            if len(self.reward_history) > 200:
                self.reward_history.pop(0)
            return reward

        return None

    # ── Reward / cost ────────────────────────────────────────────────────────────
    def compute_cost(self, agent: Agent, task: Task) -> float:
        from scheduler import assignment_cost
        return assignment_cost(agent, task, self)

    def compute_reward(self, agent: Agent, task: Task) -> float:
        base     = task.priority_score * 2.5
        time_rem = max(0, task.time_remaining / 10.0)
        load_pen = task.weight / max(agent.capacity, 1)
        return max(0.1, base + time_rem - load_pen)

    # ── Charger tick ─────────────────────────────────────────────────────────────
    def tick_chargers(self) -> None:
        for station in self.charging_stations:
            if station.occupied_by is not None:
                agent = self.get_agent(station.occupied_by)
                if agent:
                    agent.charge_tick()
                    if agent.is_full():
                        self.release_charger(agent)

    # ── Misc ─────────────────────────────────────────────────────────────────────
    def get_agent(self, agent_id: int) -> Optional[Agent]:
        for a in self.agents:
            if a.id == agent_id:
                return a
        return None

    def occupied_cells(self) -> set[tuple]:
        return {a.location for a in self.agents}

    def get_grid_state(self) -> dict:
        return {
            "grid_size":         self.grid_size,
            "agents":            list(self.agents),
            "pickup_zones":      self.pickup_zones,
            "drop_zones":        self.drop_zones,
            "congestion_zones":  list(self.congestion_zones),
            "charging_stations": self.charging_stations,
            "obstacles":         list(self.obstacles),
            "inventory":         dict(self.inventory),
            "tick":              self.tick,
            "heatmap_top10":     self.heatmap.top_n(10),
        }
