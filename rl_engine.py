"""
rl_engine.py — Reinforcement-Learning components for the warehouse dispatcher.

Two independent RL components:

1.  AgentQLearner   — per-robot Q-table that learns WHEN to charge vs. work.
    State  : (battery_bucket, queue_pressure, critical_flag, own_load)
    Actions: {0: keep_working, 1: go_charge}
    Reward : positive for completing deliveries, negative for running flat.

2.  StrategyBandit  — UCB1 bandit that selects among assignment strategies.
    Arm    : one per strategy in STRATEGY_REGISTRY.
    Reward : negative average delivery cost returned by the environment.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Tuple

from config import (
    QL_ALPHA, QL_GAMMA,
    QL_EPSILON_START, QL_EPSILON_MIN, QL_EPSILON_DECAY,
    BANDIT_UCB_C,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Per-agent Q-learner (charging / working decision)
# ══════════════════════════════════════════════════════════════════════════════

class AgentQLearner:
    """
    Tabular Q-learner for a single agent's work-vs-charge decision.

    State features
    --------------
    battery_bucket  : 'C' (<15%), 'L' (<35%), 'M' (<65%), 'H' (≥65%)
    queue_pressure  : '0' (empty), 'S' (1-3 tasks), 'B' (≥4 tasks)
    critical_present: 'c' (yes) / 'n' (no)
    own_state       : 'w' (working / heading to task) / 'i' (idle)
    """

    ACTIONS = {0: 'work', 1: 'charge'}

    def __init__(self) -> None:
        self.Q:   dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self.epsilon = QL_EPSILON_START
        self.updates  = 0
        self._last_state:  str | None = None
        self._last_action: int | None = None

    # ── State encoding ──────────────────────────────────────────────────────────
    @staticmethod
    def encode(battery: float, queue_len: int, has_critical: bool, is_working: bool) -> str:
        bb = 'C' if battery < 15 else 'L' if battery < 35 else 'M' if battery < 65 else 'H'
        qp = '0' if queue_len == 0 else 'S' if queue_len < 4 else 'B'
        cr = 'c' if has_critical else 'n'
        ow = 'w' if is_working else 'i'
        return f"{bb}{qp}{cr}{ow}"

    # ── Action selection (ε-greedy) ─────────────────────────────────────────────
    def act(self, state: str) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        q = self.Q[state]
        return 0 if q[0] >= q[1] else 1

    # ── Save last (state, action) for deferred reward ──────────────────────────
    def remember(self, state: str, action: int) -> None:
        self._last_state  = state
        self._last_action = action

    # ── Q-update (call once reward is known) ───────────────────────────────────
    def update(self, next_state: str, reward: float) -> None:
        if self._last_state is None:
            return
        s, a = self._last_state, self._last_action
        best_next = max(self.Q[next_state])
        td_target = reward + QL_GAMMA * best_next
        self.Q[s][a] += QL_ALPHA * (td_target - self.Q[s][a])
        self.epsilon   = max(QL_EPSILON_MIN, self.epsilon * QL_EPSILON_DECAY)
        self.updates  += 1
        self._last_state  = None
        self._last_action = None

    # ── Convenience wrappers ────────────────────────────────────────────────────
    def decide_charge(
        self,
        battery:      float,
        queue_len:    int,
        has_critical: bool,
        is_working:   bool,
    ) -> bool:
        """Return True if the agent should divert to a charger right now."""
        s      = self.encode(battery, queue_len, has_critical, is_working)
        action = self.act(s)
        self.remember(s, action)
        return action == 1   # 1 = charge

    def reward_delivery(self, priority_val: int, battery: float, queue_len: int,
                        has_critical: bool, is_working: bool) -> None:
        """Call when a delivery is completed."""
        r  = float(priority_val) * 1.8
        ns = self.encode(battery, queue_len, has_critical, is_working)
        self.update(ns, r)

    def reward_flat_battery(self, battery: float, queue_len: int,
                            has_critical: bool, is_working: bool) -> None:
        """Call when battery hits 0 (very bad)."""
        ns = self.encode(battery, queue_len, has_critical, is_working)
        self.update(ns, -12.0)

    def reward_charge_completed(self, battery: float, queue_len: int,
                                has_critical: bool, is_working: bool) -> None:
        """Call when a full charge cycle finishes."""
        ns = self.encode(battery, queue_len, has_critical, is_working)
        self.update(ns, 3.0)

    @property
    def best_action(self) -> dict[str, str]:
        """Compact view of learned policy for display."""
        policy = {}
        for state, (q0, q1) in self.Q.items():
            policy[state] = self.ACTIONS[0 if q0 >= q1 else 1]
        return policy

    def summary(self) -> dict:
        return {
            'epsilon':  round(self.epsilon, 4),
            'updates':  self.updates,
            'q_states': len(self.Q),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  UCB1 strategy bandit
# ══════════════════════════════════════════════════════════════════════════════

class StrategyBandit:
    """
    Upper-Confidence-Bound (UCB1) bandit over assignment strategy arms.
    UCB1 balances exploration/exploitation without a fixed ε parameter.

    score_i = avg_reward_i + C * sqrt( ln(total_pulls) / pulls_i )
    """

    def __init__(self, num_arms: int, arm_names: list[str]) -> None:
        self.n         = num_arms
        self.names     = arm_names
        self.counts    = [0] * num_arms
        self.rewards   = [0.0] * num_arms
        self.total     = 0
        self.history   = []          # [(tick, arm_idx, reward)]

    # ── Arm selection ───────────────────────────────────────────────────────────
    def select(self) -> int:
        # Pull each arm once before using UCB1
        for i in range(self.n):
            if self.counts[i] == 0:
                return i

        ln_total = math.log(self.total)
        scores   = [
            self.rewards[i] / self.counts[i]
            + BANDIT_UCB_C * math.sqrt(ln_total / self.counts[i])
            for i in range(self.n)
        ]
        return scores.index(max(scores))

    # ── Update after observing reward ──────────────────────────────────────────
    def update(self, arm: int, reward: float, tick: int = 0) -> None:
        self.counts[arm]  += 1
        self.rewards[arm] += reward
        self.total        += 1
        self.history.append((tick, arm, reward))
        if len(self.history) > 500:
            self.history.pop(0)

    # ── Analytics ──────────────────────────────────────────────────────────────
    def avg_rewards(self) -> list[float]:
        return [
            self.rewards[i] / self.counts[i] if self.counts[i] > 0 else 0.0
            for i in range(self.n)
        ]

    def best_arm(self) -> Tuple[int, str]:
        avgs = self.avg_rewards()
        idx  = avgs.index(max(avgs))
        return idx, self.names[idx]

    def arm_stats(self) -> list[dict]:
        avgs = self.avg_rewards()
        return [
            {'name': self.names[i], 'pulls': self.counts[i], 'avg_reward': round(avgs[i], 3)}
            for i in range(self.n)
        ]

    def summary(self) -> dict:
        idx, name = self.best_arm()
        return {
            'best_strategy': name,
            'total_pulls':   self.total,
            'arm_stats':     self.arm_stats(),
        }
