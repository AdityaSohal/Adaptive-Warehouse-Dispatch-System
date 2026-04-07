"""
learning.py - Meta-learning via Epsilon-Greedy Multi-Armed Bandit.
The system learns which task assignment strategy performs best over time.

Key Idea:
  - With probability epsilon: explore (pick a random strategy)
  - Otherwise: exploit (pick the strategy with the highest average reward)
  - After each round, update the chosen strategy's score and count
"""

import random
from strategies import STRATEGY_REGISTRY


class EpsilonGreedyBandit:
    """
    Multi-Armed Bandit for strategy selection.

    Arms = assignment strategies (nearest, fastest, least_loaded, random)
    Reward = average reward per round for that strategy

    Tracks:
        strategy_scores[i]  - cumulative reward for strategy i
        strategy_counts[i]  - number of times strategy i was selected
        history             - full log of (round, strategy_index, avg_reward)
    """

    def __init__(self, epsilon: float = 0.2):
        self.epsilon = epsilon
        self.num_strategies = len(STRATEGY_REGISTRY)

        # Initialize scores and counts to zero
        self.strategy_scores = [0.0] * self.num_strategies
        self.strategy_counts = [0]   * self.num_strategies

        # Full history for visualization
        self.history = []  # List of dicts: {round, strategy_idx, strategy_name, reward}
        self.last_chosen = None

    def select_strategy(self) -> int:
        """
        Epsilon-greedy selection:
          - With prob epsilon: pick a random strategy (explore)
          - Otherwise: pick the strategy with highest average reward (exploit)

        Returns:
            Index of the chosen strategy
        """
        if random.random() < self.epsilon:
            # Explore: random choice
            chosen = random.randint(0, self.num_strategies - 1)
        else:
            # Exploit: best average reward so far
            avg_rewards = [
                self.strategy_scores[i] / self.strategy_counts[i]
                if self.strategy_counts[i] > 0 else 0.0
                for i in range(self.num_strategies)
            ]
            chosen = avg_rewards.index(max(avg_rewards))

        self.last_chosen = chosen
        return chosen

    def update(self, strategy_idx: int, round_reward: float, round_num: int):
        """
        Update the bandit's estimate for the chosen strategy.

        Uses incremental mean update:
            new_avg = old_avg + (reward - old_avg) / new_count

        Args:
            strategy_idx: Index of the strategy that was used
            round_reward: Average reward achieved in this round
            round_num: Current simulation round number
        """
        self.strategy_counts[strategy_idx] += 1
        # Incremental mean (equivalent to running average)
        n = self.strategy_counts[strategy_idx]
        self.strategy_scores[strategy_idx] += (round_reward - self.strategy_scores[strategy_idx]) / n

        name = STRATEGY_REGISTRY[strategy_idx][0]
        self.history.append({
            'round': round_num,
            'strategy_idx': strategy_idx,
            'strategy_name': name,
            'reward': round_reward,
        })

    def get_best_strategy(self) -> tuple:
        """Return (index, name) of the currently best-estimated strategy."""
        avg_rewards = [
            self.strategy_scores[i] / self.strategy_counts[i]
            if self.strategy_counts[i] > 0 else float('-inf')
            for i in range(self.num_strategies)
        ]
        best_idx = avg_rewards.index(max(avg_rewards))
        return best_idx, STRATEGY_REGISTRY[best_idx][0]

    def get_strategy_function(self, idx: int):
        """Return the callable strategy function for a given index."""
        return STRATEGY_REGISTRY[idx][1]

    def get_usage_counts(self) -> dict:
        """Return {strategy_name: count} for visualization."""
        return {
            STRATEGY_REGISTRY[i][0]: self.strategy_counts[i]
            for i in range(self.num_strategies)
        }

    def summary(self) -> str:
        """Human-readable summary of learned strategy values."""
        lines = ["=== Bandit Strategy Summary ==="]
        for i in range(self.num_strategies):
            name = STRATEGY_REGISTRY[i][0]
            count = self.strategy_counts[i]
            avg = self.strategy_scores[i] if count > 0 else 0.0
            lines.append(f"  [{i}] {name:<22} count={count:>3}  avg_reward={avg:.4f}")
        best_idx, best_name = self.get_best_strategy()
        lines.append(f"\n  → Best strategy: [{best_idx}] {best_name}")
        return "\n".join(lines)
