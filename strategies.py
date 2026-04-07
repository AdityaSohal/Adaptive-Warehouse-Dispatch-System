"""
strategies.py — 5 assignment strategies for the UCB1 bandit to choose among.

Each strategy takes (tasks, agents, env) and returns {task_id: agent}.
Only idle, non-critical-battery agents are passed in.
"""

from __future__ import annotations

import random
from agent import Agent, AgentRole
from task  import Task
# forward ref to WarehouseEnvironment avoids circular import at runtime


def nearest_agent_strategy(
    tasks:  list[Task],
    agents: list[Agent],
    env,
) -> dict[int, Agent]:
    """Closest robot to each pickup zone — minimises deadhead distance."""
    assignments: dict[int, Agent] = {}
    available = list(agents)
    for task in tasks:
        if not available:
            break
        best = min(available, key=lambda a: env.manhattan(a.location, task.pickup))
        assignments[task.id] = best
        available.remove(best)
    return assignments


def fastest_agent_strategy(
    tasks:  list[Task],
    agents: list[Agent],
    env,
) -> dict[int, Agent]:
    """Highest-speed robot first — good for CRITICAL orders."""
    assignments: dict[int, Agent] = {}
    pool = sorted(agents, key=lambda a: -a.speed)
    for i, task in enumerate(tasks):
        if i >= len(pool):
            break
        assignments[task.id] = pool[i]
    return assignments


def least_loaded_strategy(
    tasks:  list[Task],
    agents: list[Agent],
    env,
) -> dict[int, Agent]:
    """Agent with fewest completed tasks — balances fleet utilisation."""
    assignments: dict[int, Agent] = {}
    available = list(agents)
    for task in tasks:
        if not available:
            break
        best = min(available, key=lambda a: a.tasks_completed)
        assignments[task.id] = best
        available.remove(best)
    return assignments


def random_strategy(
    tasks:  list[Task],
    agents: list[Agent],
    env,
) -> dict[int, Agent]:
    """Random assignment — exploration baseline for the bandit."""
    assignments: dict[int, Agent] = {}
    pool = random.sample(agents, len(agents))
    for i, task in enumerate(tasks):
        if i >= len(pool):
            break
        assignments[task.id] = pool[i]
    return assignments


def specialized_strategy(
    tasks:  list[Task],
    agents: list[Agent],
    env,
) -> dict[int, Agent]:
    """
    Role-aware + cost-minimising:
      heavy tasks  → HEAVY agents first
      short tasks  → FAST  agents first
      long tasks   → lowest compute_cost among all available
    Tie-break always on env.compute_cost.
    """
    assignments: dict[int, Agent] = {}
    available = list(agents)
    for task in tasks:
        if not available:
            break
        if task.category == 'heavy':
            pool = [a for a in available if a.role == AgentRole.HEAVY] or available
        elif task.category == 'short':
            pool = [a for a in available if a.role == AgentRole.FAST] or available
        else:
            pool = available
        best = min(pool, key=lambda a: env.compute_cost(a, task))
        assignments[task.id] = best
        available.remove(best)
    return assignments


# Registry consumed by StrategyBandit and Dispatcher
STRATEGY_REGISTRY: dict[int, tuple[str, callable]] = {
    0: ('nearest',     nearest_agent_strategy),
    1: ('fastest',     fastest_agent_strategy),
    2: ('balanced',    least_loaded_strategy),
    3: ('random',      random_strategy),
    4: ('specialized', specialized_strategy),
}

STRATEGY_NAMES = [v[0] for v in STRATEGY_REGISTRY.values()]
