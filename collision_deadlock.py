"""
collision_deadlock.py — ORCA velocity-obstacle collision avoidance and
                        wait-for-graph deadlock detection + push-and-swap resolution.

ORCA (Optimal Reciprocal Collision Avoidance):
  Each agent computes a set of half-planes in velocity space that are forbidden
  by nearby agents' trajectories.  It then projects its preferred velocity onto
  the intersection of allowed half-planes to find a collision-free velocity.
  Reference: van den Berg et al., "Reciprocal n-Body Collision Avoidance", ISRR 2011.

Deadlock detection:
  A directed "wait-for" graph is built where agent A → agent B means A is
  waiting for B to vacate A's next cell.  A cycle in this graph is a deadlock.
  We resolve it using push-and-swap: identify the cycle, assign one "escape
  agent", and move it to the nearest free cell, breaking the cycle.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from typing import Optional

from config import (
    ORCA_TAU, ORCA_NEIGHBOR_DIST, ORCA_MAX_SPEED, ORCA_TIME_HORIZON,
    TTC_EMERGENCY_THRESH, DEADLOCK_WAIT, PUSH_SWAP_RADIUS, DEADLOCK_TOKEN_TTL,
    GRID_SIZE,
)


# ══════════════════════════════════════════════════════════════════════════════
# ORCA  —  velocity-obstacle collision avoidance
# ══════════════════════════════════════════════════════════════════════════════

def _dot(a: tuple, b: tuple) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _sub(a: tuple, b: tuple) -> tuple:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: tuple, b: tuple) -> tuple:
    return (a[0] + b[0], a[1] + b[1])


def _scale(s: float, v: tuple) -> tuple:
    return (s * v[0], s * v[1])


def _norm(v: tuple) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2)


def _normalize(v: tuple) -> tuple:
    n = _norm(v)
    if n < 1e-9:
        return (0.0, 0.0)
    return (v[0] / n, v[1] / n)


class ORCAAgent:
    """
    Lightweight ORCA wrapper around a warehouse agent.
    Call `compute_orca_velocity` to get a collision-avoidance-adjusted velocity.
    """

    def __init__(self, agent_id: int, radius: float = 0.5):
        self.id     = agent_id
        self.radius = radius  # in grid-cell units

    def compute_orca_velocity(
        self,
        pos:           tuple[float, float],
        pref_vel:      tuple[float, float],
        neighbors:     list[dict],          # [{pos, vel, radius}]
        max_speed:     float = ORCA_MAX_SPEED,
        tau:           float = ORCA_TIME_HORIZON,
    ) -> tuple[float, float]:
        """
        Given preferred velocity `pref_vel`, return a velocity that avoids
        all neighbors over time horizon `tau`.

        Each neighbor produces one ORCA half-plane: a (point, normal) in velocity
        space.  We project pref_vel onto the feasible region defined by all
        half-planes using a greedy linear program.

        If no collision risk exists the preferred velocity is returned unchanged.
        """
        half_planes: list[tuple[tuple, tuple]] = []   # (point, outward_normal)

        for nb in neighbors:
            rel_pos = _sub(nb["pos"], pos)
            rel_vel = _sub(pref_vel, nb["vel"])
            dist    = _norm(rel_pos)
            comb_r  = self.radius + nb["radius"]

            if dist < comb_r:
                # Already overlapping — push straight out
                n = _normalize(rel_pos) if dist > 1e-9 else (1.0, 0.0)
                u = _scale(comb_r - dist + 0.01, n)
                half_planes.append((_scale(0.5, u), n))
                continue

            w = _sub(rel_vel, _scale(1.0 / tau, rel_pos))
            w_len = _norm(w)
            if w_len < 1e-9:
                continue

            # Check if w is inside the velocity obstacle cone
            dot_w_rp = _dot(w, _normalize(rel_pos))
            if dot_w_rp < 0:
                continue  # diverging, no constraint needed

            # Compute ORCA half-plane
            u_dir = _normalize(w)
            # Project onto cone boundary
            theta = math.asin(min(1.0, comb_r / dist))
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            # Rotate u_dir by theta to find cone edge
            edge = (
                cos_t * (-rel_pos[1] / dist) - sin_t * (rel_pos[0] / dist),
                sin_t * (-rel_pos[1] / dist) + cos_t * (rel_pos[0] / dist),
            )
            u = _scale(_dot(w, edge) - w_len, edge)
            half_planes.append((_scale(0.5, u), _normalize(u) if _norm(u) > 1e-9 else u_dir))

        if not half_planes:
            return pref_vel

        # Greedy linear program: project pref_vel against each half-plane
        new_vel = pref_vel
        for point, normal in half_planes:
            if _dot(_sub(new_vel, point), normal) < 0:
                # Violated — project onto the half-plane boundary
                dot_val = _dot(_sub(new_vel, point), normal)
                new_vel = _sub(new_vel, _scale(dot_val, normal))

        # Clamp to max_speed
        speed = _norm(new_vel)
        if speed > max_speed:
            new_vel = _scale(max_speed / speed, new_vel)

        return new_vel


def compute_time_to_collision(
    pos_a:  tuple[float, float],
    vel_a:  tuple[float, float],
    pos_b:  tuple[float, float],
    vel_b:  tuple[float, float],
    radius: float = 0.5,
) -> float:
    """
    Compute the time until two circular agents collide given constant velocities.
    Returns math.inf if they never collide.
    """
    rel_pos = _sub(pos_b, pos_a)
    rel_vel = _sub(vel_b, vel_a)
    comb_r  = radius * 2

    a = _dot(rel_vel, rel_vel)
    b = 2.0 * _dot(rel_pos, rel_vel)
    c = _dot(rel_pos, rel_pos) - comb_r ** 2

    if a < 1e-9:
        return math.inf  # no relative motion

    disc = b * b - 4 * a * c
    if disc < 0:
        return math.inf  # no intersection

    t1 = (-b - math.sqrt(disc)) / (2 * a)
    t2 = (-b + math.sqrt(disc)) / (2 * a)
    if t2 < 0:
        return math.inf  # collision in the past
    return max(0.0, t1)


def emergency_stop_needed(
    pos_a:  tuple[float, float],
    vel_a:  tuple[float, float],
    pos_b:  tuple[float, float],
    vel_b:  tuple[float, float],
    thresh: float = TTC_EMERGENCY_THRESH,
) -> bool:
    """Return True if an emergency stop should be triggered (TTC < threshold)."""
    ttc = compute_time_to_collision(pos_a, vel_a, pos_b, vel_b)
    return ttc < thresh


# ══════════════════════════════════════════════════════════════════════════════
# Deadlock detection — wait-for graph + cycle detection
# ══════════════════════════════════════════════════════════════════════════════

class WaitForGraph:
    """
    Directed graph: edge A → B means agent A is waiting for agent B to move.
    Cycles = deadlocks.
    """

    def __init__(self):
        self._edges: dict[int, int] = {}    # agent_id → blocking_agent_id
        self._wait_ticks: dict[int, int] = defaultdict(int)
        self._push_tokens: dict[int, int] = {}   # agent_id → expiry tick

    def update(self, waiting_agent: int, blocking_agent: int, tick: int) -> None:
        self._edges[waiting_agent] = blocking_agent
        self._wait_ticks[waiting_agent] += 1

    def clear_agent(self, agent_id: int) -> None:
        self._edges.pop(agent_id, None)
        self._wait_ticks.pop(agent_id, None)

    def detect_cycles(self) -> list[list[int]]:
        """
        Returns a list of cycles (each cycle is a list of agent IDs).
        Uses DFS on the wait-for graph.
        """
        cycles: list[list[int]] = []
        visited: set[int] = set()
        in_stack: dict[int, int] = {}   # agent_id → stack position
        stack: list[int] = []

        def dfs(node: int) -> None:
            visited.add(node)
            in_stack[node] = len(stack)
            stack.append(node)

            nb = self._edges.get(node)
            if nb is not None:
                if nb in in_stack:
                    # Found cycle
                    cycle_start = in_stack[nb]
                    cycles.append(list(stack[cycle_start:]))
                elif nb not in visited:
                    dfs(nb)

            stack.pop()
            del in_stack[node]

        for node in list(self._edges.keys()):
            if node not in visited:
                dfs(node)

        return cycles

    def long_wait_agents(self, threshold: int = DEADLOCK_WAIT) -> list[int]:
        """Return agents that have been waiting longer than the threshold."""
        return [
            aid for aid, ticks in self._wait_ticks.items()
            if ticks >= threshold
        ]

    def issue_push_token(self, agent_id: int, tick: int) -> None:
        self._push_tokens[agent_id] = tick + DEADLOCK_TOKEN_TTL

    def has_push_token(self, agent_id: int, tick: int) -> bool:
        exp = self._push_tokens.get(agent_id, 0)
        return exp > tick

    def expire_tokens(self, tick: int) -> None:
        self._push_tokens = {k: v for k, v in self._push_tokens.items() if v > tick}


# ══════════════════════════════════════════════════════════════════════════════
# Push-and-swap deadlock resolution
# ══════════════════════════════════════════════════════════════════════════════

def find_push_target(
    agent_pos:  tuple[int, int],
    obstacles:  set[tuple],
    occupied:   set[tuple],
    grid_size:  int = GRID_SIZE,
    radius:     int = PUSH_SWAP_RADIUS,
) -> Optional[tuple[int, int]]:
    """
    BFS outward from `agent_pos` to find the nearest free cell within `radius`.
    Used to move an escape agent out of a deadlocked cluster.
    """
    queue: deque[tuple[int, int]] = deque([agent_pos])
    seen:  set[tuple[int, int]]   = {agent_pos}

    while queue:
        pos = queue.popleft()
        dist = abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1])
        if dist > radius:
            break

        if pos != agent_pos and pos not in obstacles and pos not in occupied:
            return pos

        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in seen:
                seen.add((nx, ny))
                queue.append((nx, ny))

    return None


def resolve_cycle(
    cycle:     list[int],
    agent_map: dict[int, object],   # agent_id → Agent object
    obstacles: set[tuple],
    occupied:  set[tuple],
    tick:      int,
    wfg:       WaitForGraph,
    grid_size: int = GRID_SIZE,
) -> int:
    """
    Choose one "escape agent" from the cycle (lowest battery first, i.e. the
    most vulnerable robot).  Issue it a push token and move it to the nearest
    safe cell.  Returns the agent_id of the escape agent, or -1 on failure.
    """
    # Pick escape agent: lowest battery in the cycle
    escape_id = min(
        cycle,
        key=lambda aid: getattr(agent_map.get(aid), "battery", 100.0),
    )
    escape_agent = agent_map.get(escape_id)
    if escape_agent is None:
        return -1

    target = find_push_target(
        escape_agent.location, obstacles, occupied, grid_size
    )
    if target is None:
        # Try random agent in cycle
        for aid in cycle:
            ag = agent_map.get(aid)
            if ag is None:
                continue
            target = find_push_target(ag.location, obstacles, occupied, grid_size)
            if target:
                escape_id = aid
                escape_agent = ag
                break

    if target is None:
        return -1

    # Execute the push
    escape_agent.move_to(*target)
    escape_agent.drain_battery()
    occupied.discard(escape_agent.location)
    occupied.add(target)

    # Clear wait state for all agents in cycle
    for aid in cycle:
        wfg.clear_agent(aid)

    wfg.issue_push_token(escape_id, tick)
    return escape_id
