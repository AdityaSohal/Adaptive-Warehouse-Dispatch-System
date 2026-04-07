"""
strategies.py - Task assignment strategies for the warehouse simulation.
Implements 4 strategies: nearest, fastest, least_loaded, and random.
Each strategy maps a list of tasks to available agents.
"""

import random


def nearest_agent_strategy(tasks: list, agents: list, env) -> dict:
    """
    Assign each task to the agent closest to the pickup location.
    Uses Manhattan distance from agent's current position to task pickup.

    Returns:
        dict: {task_id -> agent}
    """
    assignments = {}
    available_agents = list(agents)  # copy so we can manage load

    for task in tasks:
        if not available_agents:
            break  # No agents left to assign

        # Find the nearest agent to this task's pickup
        best_agent = min(
            available_agents,
            key=lambda a: env.manhattan_distance(a.location, task.pickup)
        )
        assignments[task.id] = best_agent

        # Mark agent as busy (remove from pool for this round's assignment pass)
        # In this simulation, each agent handles one task per round
        available_agents.remove(best_agent)

    return assignments


def fastest_agent_strategy(tasks: list, agents: list, env) -> dict:
    """
    Assign each task to the agent with the highest speed.
    Ties broken by distance.

    Returns:
        dict: {task_id -> agent}
    """
    assignments = {}
    # Sort agents by speed descending
    sorted_agents = sorted(agents, key=lambda a: -a.speed)
    available_agents = list(sorted_agents)

    for task in tasks:
        if not available_agents:
            break
        best_agent = available_agents[0]  # Fastest available
        assignments[task.id] = best_agent
        available_agents.remove(best_agent)

    return assignments


def least_loaded_strategy(tasks: list, agents: list, env) -> dict:
    """
    Assign each task to the agent with the lowest current load relative to capacity.
    This balances workload across agents.

    Returns:
        dict: {task_id -> agent}
    """
    assignments = {}
    available_agents = list(agents)

    for task in tasks:
        if not available_agents:
            break
        # Load ratio: current_load / capacity (0 = empty, 1 = full)
        best_agent = min(available_agents, key=lambda a: a.current_load / a.capacity)
        assignments[task.id] = best_agent
        available_agents.remove(best_agent)

    return assignments


def random_strategy(tasks: list, agents: list, env) -> dict:
    """
    Randomly assign each task to a different agent.
    Serves as baseline / exploration reference.

    Returns:
        dict: {task_id -> agent}
    """
    assignments = {}
    available_agents = list(agents)
    random.shuffle(available_agents)

    for i, task in enumerate(tasks):
        if not available_agents:
            break
        agent = available_agents[i % len(available_agents)]
        assignments[task.id] = agent
        available_agents = [a for a in available_agents if a != agent]

    return assignments


def specialized_strategy(tasks: list, agents: list, env) -> dict:
    """
    BONUS: Assign tasks to agents based on their specialized role.
    - short tasks → fast_agent
    - heavy tasks → heavy_agent
    - long tasks  → long_distance_agent
    - fallback    → nearest agent

    Returns:
        dict: {task_id -> agent}
    """
    assignments = {}
    available_agents = list(agents)

    role_preference = {
        'short': 'fast_agent',
        'heavy': 'heavy_agent',
        'long': 'long_distance_agent',
    }

    for task in tasks:
        if not available_agents:
            break

        preferred_role = role_preference.get(task.category)

        # Try to find an agent with the preferred role
        matching = [a for a in available_agents if a.role == preferred_role]
        if matching:
            best = min(matching, key=lambda a: env.manhattan_distance(a.location, task.pickup))
        else:
            # Fallback: nearest agent
            best = min(available_agents, key=lambda a: env.manhattan_distance(a.location, task.pickup))

        assignments[task.id] = best
        available_agents.remove(best)

    return assignments


# Registry of all strategies for the meta-learner to choose from
STRATEGY_REGISTRY = {
    0: ('nearest_agent',   nearest_agent_strategy),
    1: ('fastest_agent',   fastest_agent_strategy),
    2: ('least_loaded',    least_loaded_strategy),
    3: ('random',          random_strategy),
}
