"""
strategies.py
4 assignment strategies. Each takes (tasks, agents, env) and returns
{task_id: agent}. Agents list contains only idle, non-low-battery agents.
"""

import random


def nearest_agent_strategy(tasks, agents, env) -> dict:
    """Assign each task to the closest available agent."""
    assignments = {}
    available   = list(agents)
    for task in tasks:
        if not available:
            break
        best = min(available, key=lambda a: env.manhattan(a.location, task.pickup))
        assignments[task.id] = best
        available.remove(best)
    return assignments


def fastest_agent_strategy(tasks, agents, env) -> dict:
    """Assign each task to the fastest available agent."""
    assignments = {}
    available   = sorted(agents, key=lambda a: -a.speed)
    pool        = list(available)
    for task in tasks:
        if not pool:
            break
        assignments[task.id] = pool[0]
        pool.pop(0)
    return assignments


def least_loaded_strategy(tasks, agents, env) -> dict:
    """Assign each task to the agent with lowest load ratio."""
    assignments = {}
    available   = list(agents)

    for task in tasks:
        if not available:
            break

        best = min(
            available,
            key=lambda a: getattr(a, "current_load", 0) / max(getattr(a, "capacity", 1), 1)
        )

        assignments[task.id] = best
        available.remove(best)

    return assignments


def random_strategy(tasks, agents, env) -> dict:
    """Random assignment baseline."""
    assignments = {}
    pool = list(agents)
    random.shuffle(pool)
    for i, task in enumerate(tasks):
        if not pool:
            break
        agent = pool[i % len(pool)]
        assignments[task.id] = agent
        pool = [a for a in pool if a != agent]
    return assignments


def specialized_strategy(tasks, agents, env) -> dict:
    """
    Role-aware assignment:
      - heavy tasks  → HEAVY role agents first
      - short tasks  → FAST  role agents first
      - long tasks   → best by cost estimate
    """
    from agent import AgentRole
    assignments = {}
    available   = list(agents)

    for task in tasks:
        if not available:
            break
        if task.category == 'heavy':
            preferred = [a for a in available if a.role == AgentRole.HEAVY]
        elif task.category == 'short':
            preferred = [a for a in available if a.role == AgentRole.FAST]
        else:
            preferred = available

        pool = preferred if preferred else available
        best = min(pool, key=lambda a: env.compute_cost(a, task))
        assignments[task.id] = best
        available.remove(best)
    return assignments


STRATEGY_REGISTRY = {
    0: ('nearest_agent',  nearest_agent_strategy),
    1: ('fastest_agent',  fastest_agent_strategy),
    2: ('least_loaded',   least_loaded_strategy),
    3: ('random',         random_strategy),
    4: ('specialized',    specialized_strategy),
}
