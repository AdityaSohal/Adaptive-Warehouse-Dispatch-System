"""
agent.py - Defines the Robot/Agent class for the warehouse simulation.
Each agent has physical attributes, a performance history, and a dynamically assigned role.
"""

import random
from collections import defaultdict


class Agent:
    """
    Represents a warehouse robot with speed, capacity, and learning capabilities.
    Agents track their performance across task categories and evolve specialized roles.
    """

    # Role thresholds: how many tasks in a category before role is assigned
    ROLE_THRESHOLD = 5

    def __init__(self, agent_id: int, x: int, y: int, speed: float = None, capacity: float = None):
        self.id = agent_id
        self.x = x
        self.y = y

        # Physical attributes (randomized if not provided)
        self.speed = speed if speed is not None else round(random.uniform(0.5, 2.0), 2)
        self.capacity = capacity if capacity is not None else round(random.uniform(5.0, 20.0), 2)

        self.current_load = 0.0           # How much weight the agent is currently carrying
        self.is_busy = False              # Whether agent is currently on a task

        # Performance tracking
        self.performance_history = []     # List of rewards per task
        self.total_tasks = 0

        # Specialization: track cumulative reward by task category
        # Categories: 'short', 'long', 'heavy'
        self.specialization_scores = defaultdict(float)
        self.specialization_counts = defaultdict(int)

        # Dynamically assigned role
        # Possible: 'generalist', 'fast_agent', 'heavy_agent', 'long_distance_agent'
        self.role = 'generalist'

    @property
    def location(self):
        """Return current location as a tuple."""
        return (self.x, self.y)

    @property
    def avg_performance(self):
        """Average reward across all tasks."""
        if not self.performance_history:
            return 0.0
        return sum(self.performance_history) / len(self.performance_history)

    def record_performance(self, reward: float, task_category: str):
        """
        Record a reward after completing a task and update specialization data.

        Args:
            reward: The reward received (negative cost)
            task_category: One of 'short', 'long', 'heavy'
        """
        self.performance_history.append(reward)
        self.total_tasks += 1

        # Update specialization scores for this category
        self.specialization_scores[task_category] += reward
        self.specialization_counts[task_category] += 1

        # After enough data, update the agent's role
        self._update_role()

    def _update_role(self):
        """
        Dynamically assign a role based on which category the agent performs best in.
        Only updates if the agent has enough experience in at least one category.
        """
        # Find categories with enough experience
        eligible = {
            cat: self.specialization_scores[cat] / self.specialization_counts[cat]
            for cat in ['short', 'long', 'heavy']
            if self.specialization_counts[cat] >= self.ROLE_THRESHOLD
        }

        if not eligible:
            self.role = 'generalist'
            return

        # Pick the category with the highest average reward
        best_category = max(eligible, key=eligible.get)

        role_map = {
            'short': 'fast_agent',
            'long': 'long_distance_agent',
            'heavy': 'heavy_agent',
        }
        self.role = role_map[best_category]

    def avg_score_for_category(self, category: str) -> float:
        """Return average reward for a specific task category."""
        count = self.specialization_counts[category]
        if count == 0:
            return 0.0
        return self.specialization_scores[category] / count

    def move_to(self, x: int, y: int):
        """Update the agent's position."""
        self.x = x
        self.y = y

    def __repr__(self):
        return (f"Agent(id={self.id}, pos=({self.x},{self.y}), "
                f"speed={self.speed}, capacity={self.capacity}, role={self.role})")
