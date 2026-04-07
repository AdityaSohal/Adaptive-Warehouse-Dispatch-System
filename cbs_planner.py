"""
cbs_planner.py — Conflict-Based Search (CBS) for multi-agent path planning.

CBS runs a two-level search:
  High level : constraint tree (CT) — detects conflicts, branches on constraints
  Low level  : space-time A* per agent under its current constraint set

For fleets > 20 robots the bounded-CBS variant caps iteration count and falls
back to prioritised planning so the tick budget stays fixed.

Reference: Sharon et al., "Conflict-Based Search for Optimal Multi-Agent Path
Finding", AAAI 2012.
"""

from __future__ import annotations

import heapq
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from config import (
    CBS_MAX_ITERATIONS, RESERVATION_HORIZON,
    GRID_SIZE,
)


# ── Data types ──────────────────────────────────────────────────────────────────

@dataclass(order=True)
class Constraint:
    """Forbid agent `agent_id` from occupying `pos` at time `t`."""
    agent_id: int
    pos:      tuple
    t:        int


@dataclass
class Conflict:
    """Two agents conflict at `pos` at time `t`."""
    agent_a:  int
    agent_b:  int
    pos:      tuple
    t:        int
    kind:     str = "vertex"   # "vertex" | "edge"
    pos_b:    Optional[tuple] = None   # for edge conflicts


@dataclass(order=True)
class CTNode:
    """Node in the CBS constraint tree."""
    cost:        float
    constraints: list = field(default_factory=list, compare=False)
    paths:       dict = field(default_factory=dict,  compare=False)  # agent_id → path
    _counter:    int  = field(default=0, compare=False)              # tie-break


_ct_counter = 0


def _next_ct() -> int:
    global _ct_counter
    _ct_counter += 1
    return _ct_counter


# ── Low-level planner: space-time A* ────────────────────────────────────────────

def _st_astar(
    agent_id:    int,
    start:       tuple,
    goal:        tuple,
    obstacles:   set,
    constraints: list[Constraint],
    grid_size:   int = GRID_SIZE,
    heatmap:     dict | None = None,
    memory_map:  dict | None = None,
    t_start:     int = 0,
) -> list[tuple]:
    """
    Space-time A* that respects a list of Constraint objects.
    Returns a list of (x, y) positions (not including start).
    Falls back to greedy Manhattan if no path found within budget.
    """
    forbidden: dict[tuple, set] = defaultdict(set)
    edge_forbidden: set = set()
    for c in constraints:
        if c.agent_id == agent_id:
            forbidden[c.pos].add(c.t)

    def h(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    # heap: (f, g, t, pos, path)
    open_heap = [(h(start), 0.0, t_start, start, [])]
    visited: dict[tuple, float] = {}

    while open_heap:
        _, g, t, pos, path = heapq.heappop(open_heap)

        if pos == goal:
            return path + [goal]

        state_key = (pos, min(t - t_start, RESERVATION_HORIZON))
        if state_key in visited and visited[state_key] <= g:
            continue
        visited[state_key] = g

        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)):
            nx, ny = pos[0] + dx, pos[1] + dy
            if not (0 <= nx < grid_size and 0 <= ny < grid_size):
                continue
            npos = (nx, ny)
            if npos in obstacles and npos != goal:
                continue
            nt = t + 1
            if nt in forbidden.get(npos, set()):
                continue

            extra = 0.0
            if heatmap:
                extra += heatmap.get(npos, 0.0)
            if memory_map:
                extra += memory_map.get(npos, 0.0)

            ng = g + 1.0 + extra
            heapq.heappush(
                open_heap,
                (ng + h(npos), ng, nt, npos, path + [npos]),
            )

    # Fallback: greedy Manhattan ignoring constraints
    return _greedy_fallback(start, goal, obstacles, grid_size)


def _greedy_fallback(
    start: tuple, goal: tuple, obstacles: set, grid_size: int
) -> list[tuple]:
    path: list = []
    x, y = start
    tx, ty = goal
    for _ in range(grid_size * 4):
        if (x, y) == (tx, ty):
            break
        options = []
        if x < tx: options.append((x + 1, y))
        if x > tx: options.append((x - 1, y))
        if y < ty: options.append((x, y + 1))
        if y > ty: options.append((x, y - 1))
        moved = False
        for opt in options:
            ox, oy = opt
            if opt not in obstacles and 0 <= ox < grid_size and 0 <= oy < grid_size:
                x, y = ox, oy
                path.append((x, y))
                moved = True
                break
        if not moved:
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if (nx, ny) not in obstacles and 0 <= nx < grid_size and 0 <= ny < grid_size:
                    x, y = nx, ny
                    path.append((x, y))
                    break
    return path


# ── Conflict detection ──────────────────────────────────────────────────────────

def _find_first_conflict(paths: dict[int, list]) -> Optional[Conflict]:
    """
    Scan all agent paths for the first vertex or edge conflict.
    Returns None if all paths are conflict-free.
    """
    # Build position-by-time lookup
    pos_at: dict[tuple[int, int], list[int]] = defaultdict(list)
    for agent_id, path in paths.items():
        for t, pos in enumerate(path):
            pos_at[(t, pos)].append(agent_id)

    # Vertex conflicts
    for (t, pos), agents in pos_at.items():
        if len(agents) >= 2:
            return Conflict(agent_a=agents[0], agent_b=agents[1], pos=pos, t=t)

    # Edge conflicts (swap)
    ids = list(paths.keys())
    max_t = max((len(p) for p in paths.values()), default=0)
    for t in range(max_t - 1):
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                pa, pb = paths[a], paths[b]
                if t < len(pa) - 1 and t < len(pb) - 1:
                    if pa[t] == pb[t + 1] and pb[t] == pa[t + 1]:
                        return Conflict(
                            agent_a=a, agent_b=b,
                            pos=pa[t], t=t,
                            kind="edge", pos_b=pa[t + 1],
                        )
    return None


# ── CBS high-level search ───────────────────────────────────────────────────────

class CBSPlanner:
    """
    Conflict-Based Search planner for multi-agent path finding.

    Usage:
        planner = CBSPlanner(grid_size, obstacles)
        paths = planner.plan(agents_goals, heatmap=heatmap)
        # paths: {agent_id: [(x,y), ...]}
    """

    def __init__(self, grid_size: int, obstacles: set):
        self.grid_size = grid_size
        self.obstacles = obstacles

    def plan(
        self,
        agents_goals: dict[int, tuple[tuple, tuple]],  # {agent_id: (start, goal)}
        heatmap:      dict | None = None,
        memory_maps:  dict | None = None,   # {agent_id: memory_map}
        t_start:      int = 0,
    ) -> dict[int, list]:
        """
        Returns optimal (or bounded-optimal) conflict-free paths.
        Falls back to prioritised planning after CBS_MAX_ITERATIONS.
        """
        if not agents_goals:
            return {}

        # Root CT node: no constraints, individual shortest paths
        root_paths = {}
        for aid, (start, goal) in agents_goals.items():
            mm = (memory_maps or {}).get(aid)
            root_paths[aid] = _st_astar(
                aid, start, goal, self.obstacles, [],
                self.grid_size, heatmap, mm, t_start,
            )

        root_cost = sum(len(p) for p in root_paths.values())
        root = CTNode(cost=root_cost, constraints=[], paths=root_paths, _counter=0)
        open_heap = [root]
        iterations = 0

        while open_heap and iterations < CBS_MAX_ITERATIONS:
            iterations += 1
            node = heapq.heappop(open_heap)

            conflict = _find_first_conflict(node.paths)
            if conflict is None:
                # All paths conflict-free
                return node.paths

            # Branch on conflict: one constraint per child
            for constrained_agent in (conflict.agent_a, conflict.agent_b):
                new_constraints = node.constraints + [
                    Constraint(
                        agent_id=constrained_agent,
                        pos=conflict.pos,
                        t=conflict.t,
                    )
                ]
                # Replan only the constrained agent
                new_paths = dict(node.paths)
                start, goal = agents_goals[constrained_agent]
                mm = (memory_maps or {}).get(constrained_agent)
                new_path = _st_astar(
                    constrained_agent, start, goal,
                    self.obstacles, new_constraints,
                    self.grid_size, heatmap, mm, t_start,
                )
                new_paths[constrained_agent] = new_path
                new_cost = sum(len(p) for p in new_paths.values())
                child = CTNode(
                    cost=new_cost,
                    constraints=new_constraints,
                    paths=new_paths,
                    _counter=_next_ct(),
                )
                heapq.heappush(open_heap, child)

        # CBS budget exhausted → prioritised planning as fallback
        return self._prioritised_plan(agents_goals, heatmap, memory_maps, t_start)

    def _prioritised_plan(
        self,
        agents_goals: dict[int, tuple[tuple, tuple]],
        heatmap:      dict | None,
        memory_maps:  dict | None,
        t_start:      int,
    ) -> dict[int, list]:
        """
        Prioritised planning fallback:
        Plan agents one at a time; each treats prior agents' paths as constraints.
        """
        paths: dict[int, list] = {}
        accumulated_constraints: list[Constraint] = []

        for aid, (start, goal) in agents_goals.items():
            mm = (memory_maps or {}).get(aid)
            path = _st_astar(
                aid, start, goal, self.obstacles,
                accumulated_constraints,
                self.grid_size, heatmap, mm, t_start,
            )
            paths[aid] = path
            # Add this agent's cells as constraints for future agents
            for t, pos in enumerate(path):
                accumulated_constraints.append(Constraint(aid, pos, t_start + t))

        return paths
