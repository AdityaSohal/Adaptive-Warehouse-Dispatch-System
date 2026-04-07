"""
task.py
Task / order with priority levels, categories, and full lifecycle tracking.
"""

import random
import time
from enum import Enum, auto


class Priority(Enum):
    LOW      = 1
    NORMAL   = 2
    HIGH     = 3
    URGENT   = 4
    CRITICAL = 5


class TaskStatus(Enum):
    PENDING    = auto()
    ASSIGNED   = auto()
    IN_TRANSIT = auto()
    DELIVERED  = auto()
    FAILED     = auto()


PRIORITY_LABELS = {
    Priority.LOW:      "LOW",
    Priority.NORMAL:   "NORMAL",
    Priority.HIGH:     "HIGH",
    Priority.URGENT:   "URGENT",
    Priority.CRITICAL: "CRITICAL",
}

PRIORITY_COLORS = {
    Priority.LOW:      (100, 180, 100),
    Priority.NORMAL:   (100, 160, 220),
    Priority.HIGH:     (255, 200, 50),
    Priority.URGENT:   (255, 140, 30),
    Priority.CRITICAL: (220, 60,  60),
}


class Task:
    SHORT_DIST  = 6
    HEAVY_WGHT  = 14.0

    _counter = 0

    def __init__(self, pickup: tuple, drop: tuple,
                 weight: float = None, priority: Priority = None,
                 task_id: int = None):
        Task._counter += 1
        self.id       = task_id if task_id is not None else Task._counter
        self.pickup   = pickup
        self.drop     = drop
        self.weight   = weight   if weight   is not None else round(random.uniform(1.0, 25.0), 1)
        self.priority = priority if priority is not None else random.choice(list(Priority))

        self.status          = TaskStatus.PENDING
        self.assigned_agent  = None
        self.reward          = None
        self.created_at      = time.time()
        self.completed_at    = None

        # Deadline: critical orders expire faster
        deadline_seconds = {
            Priority.LOW:      60,
            Priority.NORMAL:   45,
            Priority.HIGH:     30,
            Priority.URGENT:   20,
            Priority.CRITICAL: 12,
        }
        self.deadline = self.created_at + deadline_seconds[self.priority]

    @property
    def distance(self) -> int:
        return abs(self.drop[0]-self.pickup[0]) + abs(self.drop[1]-self.pickup[1])

    @property
    def category(self) -> str:
        if self.weight >= self.HEAVY_WGHT:
            return 'heavy'
        if self.distance <= self.SHORT_DIST:
            return 'short'
        return 'long'

    @property
    def priority_score(self) -> int:
        return self.priority.value

    @property
    def is_expired(self) -> bool:
        return time.time() > self.deadline and self.status == TaskStatus.PENDING

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.deadline - time.time())

    @property
    def priority_color(self):
        return PRIORITY_COLORS[self.priority]

    def assign(self, agent):
        self.status         = TaskStatus.ASSIGNED
        self.assigned_agent = agent.id

    def start_transit(self):
        self.status = TaskStatus.IN_TRANSIT

    def complete(self, reward: float):
        self.status       = TaskStatus.DELIVERED
        self.reward       = reward
        self.completed_at = time.time()

    def fail(self):
        self.status = TaskStatus.FAILED

    def __lt__(self, other):
        return self.priority_score > other.priority_score

    def __repr__(self):
        return (f"Task({self.id} {self.priority.name} "
                f"pick={self.pickup} drop={self.drop} "
                f"w={self.weight:.1f} cat={self.category})")


def generate_tasks(n: int, grid_size: int,
                   pickup_zones: list, drop_zones: list,
                   force_priority: Priority = None) -> list:
    tasks = []
    for _ in range(n):
        pickup = random.choice(pickup_zones)
        drop   = random.choice(drop_zones)
        while drop == pickup:
            drop = random.choice(drop_zones)
        weight   = round(random.uniform(1.0, 25.0), 1)
        priority = force_priority or random.choice(list(Priority))
        tasks.append(Task(pickup=pickup, drop=drop,
                          weight=weight, priority=priority))
    return tasks
