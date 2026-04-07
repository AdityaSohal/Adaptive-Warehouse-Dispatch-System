"""
task.py — Order lifecycle with 5 priority levels, category tags, and deadlines.
"""

import random
import time
from enum import Enum, auto
from config import (
    PRIORITY_DEADLINES, HEAVY_WEIGHT_KG, SHORT_DIST_CELLS,
    COL,
)


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


# ── Shared label / colour maps ──────────────────────────────────────────────────
PRIORITY_NAMES  = {p: p.name for p in Priority}
PRIORITY_COLORS = {p: COL['priority'][p.value] for p in Priority}


class Task:
    _counter = 0

    def __init__(
        self,
        pickup:   tuple,
        drop:     tuple,
        weight:   float | None = None,
        priority: Priority | None = None,
        task_id:  int | None = None,
    ):
        Task._counter += 1
        self.id       = task_id if task_id is not None else Task._counter
        self.pickup   = tuple(pickup)
        self.drop     = tuple(drop)
        self.weight   = weight   if weight   is not None else round(random.uniform(1.0, 28.0), 1)
        self.priority = priority if priority is not None else random.choice(list(Priority))

        self.status         = TaskStatus.PENDING
        self.assigned_agent = None
        self.reward         = None
        self.created_at     = time.time()
        self.completed_at   = None

        dl_secs         = PRIORITY_DEADLINES.get(self.priority.value, 60)
        self.deadline   = self.created_at + dl_secs

    # ── Derived properties ──────────────────────────────────────────────────────
    @property
    def distance(self) -> int:
        return abs(self.drop[0] - self.pickup[0]) + abs(self.drop[1] - self.pickup[1])

    @property
    def category(self) -> str:
        if self.weight >= HEAVY_WEIGHT_KG:
            return 'heavy'
        if self.distance <= SHORT_DIST_CELLS:
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
    def urgency_ratio(self) -> float:
        """0 = just created, 1 = deadline, >1 = past deadline."""
        elapsed   = time.time() - self.created_at
        total_ttl = self.deadline - self.created_at
        return elapsed / max(total_ttl, 1e-6)

    @property
    def color(self) -> tuple:
        return PRIORITY_COLORS[self.priority]

    # ── Lifecycle mutators ──────────────────────────────────────────────────────
    def assign(self, agent) -> None:
        self.status         = TaskStatus.ASSIGNED
        self.assigned_agent = agent.id

    def start_transit(self) -> None:
        self.status = TaskStatus.IN_TRANSIT

    def complete(self, reward: float) -> None:
        self.status       = TaskStatus.DELIVERED
        self.reward       = reward
        self.completed_at = time.time()

    def fail(self) -> None:
        self.status = TaskStatus.FAILED

    # ── Comparison (for heapq) ──────────────────────────────────────────────────
    def __lt__(self, other: 'Task') -> bool:
        # Higher priority first; tie-break by deadline (sooner = more urgent)
        if self.priority_score != other.priority_score:
            return self.priority_score > other.priority_score
        return self.deadline < other.deadline

    def __repr__(self) -> str:
        return (
            f"Task({self.id} {self.priority.name} "
            f"pick={self.pickup} drop={self.drop} "
            f"w={self.weight:.1f}kg cat={self.category} st={self.status.name})"
        )


def generate_tasks(
    n:            int,
    grid_size:    int,
    pickup_zones: list,
    drop_zones:   list,
    force_priority: Priority | None = None,
) -> list[Task]:
    """Generate n random tasks using the supplied zone lists."""
    tasks = []
    for _ in range(n):
        pickup = random.choice(pickup_zones)
        drop   = random.choice([d for d in drop_zones if d != pickup] or drop_zones)
        weight   = round(random.uniform(1.0, 28.0), 1)
        priority = force_priority or random.choice(list(Priority))
        tasks.append(Task(pickup=pickup, drop=drop, weight=weight, priority=priority))
    return tasks
