"""
environment.py - Warehouse grid environment.
Manages the 2D grid, congestion zones, pickup/drop locations,
agent positions, and the cost/reward computation logic.
"""

import random
from agent import Agent
from task import Task


class WarehouseEnvironment:
    """
    Simulates a 2D grid-based warehouse with agents, tasks, and congestion zones.
    Handles movement cost calculation and reward computation.
    """

    def __init__(self, grid_size: int = 10, num_agents: int = 4,
                 num_pickup_zones: int = 4, num_drop_zones: int = 4,
                 num_congestion_zones: int = 6, dynamic_traffic: bool = True):
        self.grid_size = grid_size
        self.dynamic_traffic = dynamic_traffic

        # Generate zone locations
        all_cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
        random.shuffle(all_cells)

        self.pickup_zones = all_cells[:num_pickup_zones]
        self.drop_zones = all_cells[num_pickup_zones:num_pickup_zones + num_drop_zones]
        self.congestion_zones = set(
            map(tuple, all_cells[num_pickup_zones + num_drop_zones:
                                  num_pickup_zones + num_drop_zones + num_congestion_zones])
        )

        # Build congestion cost map: each congested cell costs extra
        self.congestion_cost = {cell: random.uniform(0.5, 2.0) for cell in self.congestion_zones}

        # Create agents at random starting positions
        self.agents = []
        agent_positions = random.sample(all_cells, num_agents)
        for i, pos in enumerate(agent_positions):
            self.agents.append(Agent(agent_id=i + 1, x=pos[0], y=pos[1]))

        # Metrics storage
        self.round_rewards = []   # Average reward per round

    def refresh_congestion(self):
        """
        Dynamically shift congestion zones each round (BONUS: dynamic traffic).
        Randomly moves some congestion zones to new locations.
        """
        all_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        # Remove some old zones, add new ones
        num_to_change = max(1, len(self.congestion_zones) // 3)
        current = list(self.congestion_zones)
        random.shuffle(current)
        for cell in current[:num_to_change]:
            self.congestion_zones.discard(cell)
            del self.congestion_cost[cell]

        available = [c for c in all_cells if c not in self.congestion_zones
                     and c not in self.pickup_zones and c not in self.drop_zones]
        for cell in random.sample(available, min(num_to_change, len(available))):
            cell = tuple(cell)
            self.congestion_zones.add(cell)
            self.congestion_cost[cell] = random.uniform(0.5, 2.0)

    def manhattan_distance(self, a: tuple, b: tuple) -> int:
        """Compute Manhattan distance between two grid cells."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def path_congestion_penalty(self, start: tuple, end: tuple) -> float:
        """
        Estimate the congestion penalty for traveling from start to end.
        We sample intermediate cells along the straight-line path (simplified).
        """
        penalty = 0.0
        # Walk along x then y (L-shaped Manhattan path)
        x, y = start
        tx, ty = end

        # Move along x axis
        step_x = 1 if tx > x else -1
        while x != tx:
            x += step_x
            cell = (x, y)
            if cell in self.congestion_zones:
                penalty += self.congestion_cost.get(cell, 0)

        # Move along y axis
        step_y = 1 if ty > y else -1
        while y != ty:
            y += step_y
            cell = (x, y)
            if cell in self.congestion_zones:
                penalty += self.congestion_cost.get(cell, 0)

        return penalty

    def compute_cost(self, agent: Agent, task: Task) -> float:
        """
        Compute the total cost for an agent to complete a task.

        Cost = travel_distance / agent.speed
               + load_penalty
               + congestion_penalty
               - priority_bonus

        Args:
            agent: The agent performing the task
            task: The task to complete

        Returns:
            Float representing total cost (higher = worse)
        """
        # Distance: agent → pickup → drop
        dist_to_pickup = self.manhattan_distance(agent.location, task.pickup)
        dist_pickup_to_drop = task.distance
        total_distance = dist_to_pickup + dist_pickup_to_drop

        # Time penalty: slower agents take longer
        travel_cost = total_distance / agent.speed

        # Load penalty: heavy loads relative to capacity
        load_penalty = task.weight / agent.capacity

        # Congestion penalty for both legs of the journey
        cong_to_pickup = self.path_congestion_penalty(agent.location, task.pickup)
        cong_pickup_to_drop = self.path_congestion_penalty(task.pickup, task.drop)
        congestion_penalty = cong_to_pickup + cong_pickup_to_drop

        # Priority bonus: urgent tasks get a slight cost reduction (incentive to pick them up)
        priority_bonus = (task.priority - 1) * 0.5  # 0 to 2.0 bonus

        total_cost = travel_cost + load_penalty + congestion_penalty - priority_bonus
        return max(total_cost, 0.1)  # Ensure non-negative

    def compute_reward(self, agent: Agent, task: Task) -> float:
        """
        Reward = negative of cost. Lower cost = higher (less negative) reward.
        """
        return -self.compute_cost(agent, task)

    def simulate_task(self, agent: Agent, task: Task) -> float:
        """
        Execute a task: compute reward, update agent position and load,
        record performance, and return the reward.

        Args:
            agent: Agent executing the task
            task: Task to execute

        Returns:
            Reward (negative cost)
        """
        reward = self.compute_reward(agent, task)

        # Move agent to drop location after completing the task
        agent.move_to(*task.drop)
        agent.current_load = 0.0
        agent.is_busy = False

        # Record this experience for the agent's specialization learning
        agent.record_performance(reward, task.category)

        # Mark task as done
        task.complete(reward)

        return reward

    def get_agent_by_id(self, agent_id: int):
        """Look up an agent by ID."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    def get_grid_state(self):
        """
        Return a snapshot of the current grid state for visualization.
        Returns dict with agent positions, task pickups/drops, congestion zones.
        """
        return {
            'grid_size': self.grid_size,
            'agents': [(a.id, a.x, a.y, a.role) for a in self.agents],
            'pickup_zones': self.pickup_zones,
            'drop_zones': self.drop_zones,
            'congestion_zones': list(self.congestion_zones),
        }
