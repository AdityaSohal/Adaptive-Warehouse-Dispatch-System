"""
task.py - Defines the Task class representing a pickup-and-delivery job.
Tasks have weight, priority, and spatial attributes used for cost calculation.
"""

import random


class Task:
    """
    Represents a warehouse dispatch task with a pickup and delivery location.
    Tasks are categorized by distance and weight for agent specialization.
    """

    # Thresholds for task categorization
    SHORT_DISTANCE_THRESHOLD = 6    # Manhattan distance <= this is 'short'
    HEAVY_WEIGHT_THRESHOLD = 12.0   # Weight >= this is 'heavy'

    _id_counter = 0  # Class-level counter for unique IDs

    def __init__(self, pickup: tuple, drop: tuple, weight: float = None, priority: int = None, task_id: int = None):
        Task._id_counter += 1
        self.id = task_id if task_id is not None else Task._id_counter

        self.pickup = pickup    # (x, y)
        self.drop = drop        # (x, y)

        self.weight = weight if weight is not None else round(random.uniform(1.0, 20.0), 2)
        self.priority = priority if priority is not None else random.randint(1, 5)  # 1=low, 5=urgent

        self.assigned_agent_id = None
        self.completed = False
        self.reward_received = None

    @property
    def distance(self) -> int:
        """Manhattan distance from pickup to drop location."""
        return abs(self.drop[0] - self.pickup[0]) + abs(self.drop[1] - self.pickup[1])

    @property
    def category(self) -> str:
        """
        Classify the task as 'short', 'long', or 'heavy' for specialization tracking.
        Heavy takes priority over distance classification.
        """
        if self.weight >= self.HEAVY_WEIGHT_THRESHOLD:
            return 'heavy'
        if self.distance <= self.SHORT_DISTANCE_THRESHOLD:
            return 'short'
        return 'long'

    def assign(self, agent_id: int):
        """Mark this task as assigned to an agent."""
        self.assigned_agent_id = agent_id

    def complete(self, reward: float):
        """Mark this task as completed with a given reward."""
        self.completed = True
        self.reward_received = reward

    def __repr__(self):
        return (f"Task(id={self.id}, pickup={self.pickup}, drop={self.drop}, "
                f"weight={self.weight:.1f}, priority={self.priority}, "
                f"dist={self.distance}, cat={self.category})")


def generate_tasks(n: int, grid_size: int, pickup_zones: list, drop_zones: list) -> list:
    """
    Generate n random tasks with pickups from pickup_zones and drops from drop_zones.

    Args:
        n: Number of tasks to generate
        grid_size: Size of the grid (for bounds checking)
        pickup_zones: List of valid (x, y) pickup locations
        drop_zones: List of valid (x, y) drop locations

    Returns:
        List of Task objects
    """
    tasks = []
    for _ in range(n):
        pickup = random.choice(pickup_zones)
        drop = random.choice(drop_zones)
        # Ensure pickup != drop
        while drop == pickup:
            drop = random.choice(drop_zones)
        tasks.append(Task(pickup=pickup, drop=drop))
    return tasks
