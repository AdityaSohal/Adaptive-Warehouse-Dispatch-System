"""
environment.py
Grid environment with:
  - Collision detection + prevention via space-time reservations
  - Deadlock detection + resolution (wait-and-replan)
  - 3 charging stations with intelligent routing
  - Dynamic congestion zones
"""

import random
from collections import defaultdict
from agent import Agent, AgentRole, AgentState
from task import Task


class ChargingStation:
    def __init__(self, x: int, y: int, station_id: int):
        self.id       = station_id
        self.x, self.y = x, y
        self.location = (x, y)
        self.occupied_by = None   # agent.id or None

    @property
    def is_free(self) -> bool:
        return self.occupied_by is None


class WarehouseEnvironment:
    def __init__(self, grid_size: int = 20,
                 num_pickup: int = 5, num_drop: int = 5,
                 num_congestion: int = 10,
                 dynamic_traffic: bool = True):
        self.grid_size       = grid_size
        self.dynamic_traffic = dynamic_traffic
        self.tick            = 0

        # Build zones
        all_cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
        random.shuffle(all_cells)
        idx = 0
        self.pickup_zones    = all_cells[idx:idx+num_pickup];   idx += num_pickup
        self.drop_zones      = all_cells[idx:idx+num_drop];     idx += num_drop
        self.charging_locs   = all_cells[idx:idx+3];            idx += 3
        raw_cong             = all_cells[idx:idx+num_congestion]

        self.congestion_zones = set(map(tuple, raw_cong))
        self.congestion_cost  = {c: random.uniform(0.5, 2.5) for c in self.congestion_zones}

        # Obstacles = congestion zones as walls for pathfinding? No —
        # they're traversable but costly. Only actual fixed walls go here.
        self.obstacles: set = set()   # extend if you add shelves

        # Charging stations
        self.charging_stations = [
            ChargingStation(x, y, i+1)
            for i, (x, y) in enumerate(self.charging_locs)
        ]

        # Create agents: 4 fast + 2 heavy
        self.agents: list[Agent] = []
        agent_positions = random.sample(all_cells[idx:idx+10], 6)
        for i in range(4):
            x, y = agent_positions[i]
            self.agents.append(Agent(i+1, x, y, AgentRole.FAST))
        for i in range(2):
            x, y = agent_positions[4+i]
            self.agents.append(Agent(5+i, x, y, AgentRole.HEAVY))

        # Space-time reservation table: {(x, y, t): agent_id}
        self.reservations: dict = {}

        # Deadlock tracking: {agent_id: ticks_waiting}
        self.wait_counts: dict = defaultdict(int)
        self.DEADLOCK_THRESHOLD = 5   # ticks waiting → replan

        # Metrics
        self.round_rewards: list = []
        self.collision_events     = 0
        self.deadlocks_resolved   = 0
        self.charges_completed    = 0

    # ──────────────────────────────────────────
    # Grid helpers
    # ──────────────────────────────────────────
    def manhattan(self, a: tuple, b: tuple) -> int:
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

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

    def refresh_congestion(self):
        n = max(1, len(self.congestion_zones) // 4)
        existing = list(self.congestion_zones)
        random.shuffle(existing)
        for cell in existing[:n]:
            self.congestion_zones.discard(cell)
            self.congestion_cost.pop(cell, None)
        all_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        available = [c for c in all_cells
                     if c not in self.congestion_zones
                     and c not in self.pickup_zones
                     and c not in self.drop_zones
                     and c not in self.obstacles]
        for cell in random.sample(available, min(n, len(available))):
            cell = tuple(cell)
            self.congestion_zones.add(cell)
            self.congestion_cost[cell] = random.uniform(0.5, 2.5)

    # ──────────────────────────────────────────
    # Cost / reward
    # ──────────────────────────────────────────
    def compute_cost(self, agent: Agent, task: Task) -> float:
        dist  = self.manhattan(agent.location, task.pickup) + task.distance
        time_ = dist / agent.speed
        load  = task.weight / agent.capacity
        cong  = (self.congestion_penalty(agent.location, task.pickup)
                 + self.congestion_penalty(task.pickup, task.drop))
        pbonus = (task.priority_score - 1) * 0.8
        return max(0.1, time_ + load + cong - pbonus)

    def compute_reward(self, agent: Agent, task: Task) -> float:
        return -self.compute_cost(agent, task)

    # ──────────────────────────────────────────
    # Collision detection & prevention
    # ──────────────────────────────────────────
    def reserve_path(self, agent: Agent):
        """Register the agent's planned path in the space-time table."""
        for t, cell in enumerate(agent.path):
            key = (cell[0], cell[1], self.tick + t)
            if key in self.reservations and self.reservations[key] != agent.id:
                # Conflict detected — signal the agent to replan
                return False
            self.reservations[key] = agent.id
        return True

    def clear_reservations(self, agent: Agent):
        keys_to_del = [k for k, v in self.reservations.items() if v == agent.id]
        for k in keys_to_del:
            del self.reservations[k]

    def detect_collision(self, agent: Agent, next_pos: tuple) -> bool:
        """Return True if next_pos is occupied by another agent this tick."""
        for other in self.agents:
            if other.id != agent.id and other.location == next_pos:
                return True
        return False

    def resolve_collision(self, agent: Agent):
        """Make agent wait one tick and replan."""
        agent.state       = AgentState.WAITING
        agent.wait_counter += 1
        self.wait_counts[agent.id] += 1
        self.collision_events += 1

    # ──────────────────────────────────────────
    # Deadlock detection + resolution
    # ──────────────────────────────────────────
    def check_deadlocks(self):
        """
        Deadlock = a group of agents all waiting on each other.
        Resolution: highest-priority agent gets replanned; others yield.
        Using wait-threshold heuristic (simpler than full cycle detection).
        """
        for agent in self.agents:
            if agent.state == AgentState.WAITING:
                if self.wait_counts[agent.id] >= self.DEADLOCK_THRESHOLD:
                    # Force replan: clear reservation and compute fresh path
                    self.clear_reservations(agent)
                    agent.wait_counter  = 0
                    self.wait_counts[agent.id] = 0
                    agent.state         = AgentState.IDLE if not agent.current_task else agent.state
                    self.deadlocks_resolved += 1
                    # Yield: move agent to adjacent free cell temporarily
                    self._yield_agent(agent)

    def _yield_agent(self, agent: Agent):
        """Move agent one step sideways to break a deadlock."""
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
                # Recompute path from new position if task exists
                if agent.current_task:
                    task = agent.current_task
                    goal = (task.pickup if agent.state in (AgentState.TO_PICKUP, AgentState.WAITING)
                            else task.drop)
                    agent.compute_path(agent.location, goal,
                                       self.grid_size, self.obstacles,
                                       self.reservations)
                    self.reserve_path(agent)
                break

    # ──────────────────────────────────────────
    # Charging logic
    # ──────────────────────────────────────────
    def nearest_free_charger(self, agent: Agent):
        """Return the nearest free charging station, or None."""
        free = [s for s in self.charging_stations if s.is_free or s.occupied_by == agent.id]
        if not free:
            return None
        return min(free, key=lambda s: self.manhattan(agent.location, s.location))

    def occupy_charger(self, agent: Agent, station: ChargingStation):
        station.occupied_by = agent.id
        agent.state = AgentState.CHARGING

    def release_charger(self, agent: Agent):
        for s in self.charging_stations:
            if s.occupied_by == agent.id:
                s.occupied_by = None
        agent.state = AgentState.IDLE

    # ──────────────────────────────────────────
    # Task execution
    # ──────────────────────────────────────────
    def simulate_task_step(self, agent: Agent) -> float | None:
        """
        Advance agent one tick along its task path.
        Returns reward when task is delivered, None otherwise.
        """
        if agent.state == AgentState.WAITING:
            agent.wait_counter += 1
            if agent.wait_counter > 2:
                agent.state = (AgentState.TO_PICKUP
                               if agent.current_task and
                               agent.location != agent.current_task.pickup
                               else AgentState.TO_DROP)
                agent.wait_counter = 0
            return None

        task = agent.current_task
        if task is None:
            return None

        # Determine goal
        if agent.state == AgentState.TO_PICKUP:
            goal = task.pickup
        elif agent.state == AgentState.TO_DROP:
            goal = task.drop
        else:
            return None

        # Recompute path if empty
        if not agent.path:
            agent.compute_path(agent.location, goal,
                               self.grid_size, self.obstacles,
                               self.reservations)
            self.reserve_path(agent)

        # Step
        if agent.path:
            next_pos = agent.path[0]

            # Update agent memory with congestion
            if next_pos in self.congestion_zones:
                agent.update_memory(next_pos, self.congestion_cost.get(next_pos, 0))

            # Collision check
            if self.detect_collision(agent, next_pos):
                self.resolve_collision(agent)
                return None

            agent.step_path()

        # Check arrival
        if agent.location == task.pickup and agent.state == AgentState.TO_PICKUP:
            agent.state = AgentState.TO_DROP
            task.start_transit()
            agent.path = []

        elif agent.location == task.drop and agent.state == AgentState.TO_DROP:
            reward = self.compute_reward(agent, task)
            task.complete(reward)
            agent.record_task(reward, task.category)
            agent.current_task = None
            agent.state        = AgentState.IDLE
            self.clear_reservations(agent)
            return reward

        return None

    def tick_chargers(self):
        """Each tick: charge agents on stations, release when full."""
        for station in self.charging_stations:
            if station.occupied_by is not None:
                agent = self.get_agent(station.occupied_by)
                if agent:
                    agent.charge_tick()
                    if agent.is_full():
                        self.release_charger(agent)
                        self.charges_completed += 1

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────
    def get_agent(self, agent_id: int) -> Agent | None:
        for a in self.agents:
            if a.id == agent_id:
                return a
        return None

    def get_grid_state(self) -> dict:
        return {
            'grid_size':        self.grid_size,
            'agents':           list(self.agents),
            'pickup_zones':     self.pickup_zones,
            'drop_zones':       self.drop_zones,
            'congestion_zones': list(self.congestion_zones),
            'charging_stations': self.charging_stations,
            'tick':             self.tick,
        }
