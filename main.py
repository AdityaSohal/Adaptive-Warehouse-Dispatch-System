"""
main.py - Entry point for the Self-Learning Warehouse Dispatch System.

Run modes:
  1. Simulation only (headless):
       python main.py

  2. Streamlit dashboard:
       streamlit run main.py

The simulation runs for N rounds. Each round:
  - Generates random tasks
  - Uses epsilon-greedy bandit to select an assignment strategy
  - Assigns tasks to agents via the chosen strategy
  - Simulates task execution and computes rewards
  - Updates meta-learner and agent specialization data
  - Records metrics for visualization
"""

import sys
import random
from collections import defaultdict

from environment import WarehouseEnvironment
from task import generate_tasks
from learning import EpsilonGreedyBandit
from visualization import (
    plot_warehouse_grid,
    plot_performance_over_time,
    plot_strategy_usage,
    plot_role_distribution,
    plot_demand_heatmap,
    launch_streamlit_dashboard,
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    'grid_size':          15,    # NxN warehouse grid
    'num_agents':         5,     # Number of robots
    'num_pickup_zones':   5,     # Pickup location count
    'num_drop_zones':     5,     # Drop location count
    'num_congestion':     8,     # Congestion zone count
    'tasks_per_round':    4,     # Tasks generated each round
    'num_rounds':         60,    # Total simulation rounds
    'epsilon':            0.2,   # Exploration rate for bandit
    'dynamic_traffic':    True,  # Refresh congestion zones each round
    'random_seed':        42,
}

random.seed(CONFIG['random_seed'])


# ─────────────────────────────────────────────
# SIMULATION CORE
# ─────────────────────────────────────────────

def run_simulation() -> dict:
    """
    Execute the full simulation and return results for visualization.

    Returns:
        dict with all metrics needed for the dashboard
    """
    print("=" * 60)
    print("  Self-Learning Warehouse Dispatch System")
    print("=" * 60)

    # ── Initialize environment and learner ──────
    env = WarehouseEnvironment(
        grid_size=CONFIG['grid_size'],
        num_agents=CONFIG['num_agents'],
        num_pickup_zones=CONFIG['num_pickup_zones'],
        num_drop_zones=CONFIG['num_drop_zones'],
        num_congestion_zones=CONFIG['num_congestion'],
        dynamic_traffic=CONFIG['dynamic_traffic'],
    )

    bandit = EpsilonGreedyBandit(epsilon=CONFIG['epsilon'])

    # ── Metric containers ──────────────────────
    round_rewards  = []      # Avg reward per round
    role_history   = []      # Role distribution snapshots per round
    all_tasks      = []      # Every task ever created (for heatmap)

    print(f"\nGrid: {CONFIG['grid_size']}x{CONFIG['grid_size']}  |  "
          f"Agents: {CONFIG['num_agents']}  |  "
          f"Rounds: {CONFIG['num_rounds']}  |  "
          f"epsilon: {CONFIG['epsilon']}\n")

    # ── Main simulation loop ────────────────────
    for round_num in range(1, CONFIG['num_rounds'] + 1):

        # 1. Optionally refresh congestion zones (dynamic traffic)
        if CONFIG['dynamic_traffic'] and round_num > 1:
            env.refresh_congestion()

        # 2. Generate random tasks for this round
        tasks = generate_tasks(
            n=CONFIG['tasks_per_round'],
            grid_size=CONFIG['grid_size'],
            pickup_zones=env.pickup_zones,
            drop_zones=env.drop_zones,
        )
        all_tasks.extend(tasks)

        # 3. Select strategy via epsilon-greedy bandit
        strategy_idx = bandit.select_strategy()
        strategy_fn  = bandit.get_strategy_function(strategy_idx)
        strategy_name = bandit.history[-1]['strategy_name'] if bandit.history else '?'

        # 4. Assign tasks to agents using the chosen strategy
        assignments = strategy_fn(tasks, env.agents, env)
        # assignments: {task_id -> agent}

        # 5. Simulate each assigned task
        round_task_rewards = []
        for task in tasks:
            agent = assignments.get(task.id)
            if agent is None:
                continue  # No agent available for this task

            # Execute task: move agent, compute reward, update specialization
            reward = env.simulate_task(agent, task)
            round_task_rewards.append(reward)

        # 6. Compute round average reward
        avg_reward = (sum(round_task_rewards) / len(round_task_rewards)
                      if round_task_rewards else 0.0)
        round_rewards.append(avg_reward)

        # 7. Update bandit with this round's performance
        bandit.update(strategy_idx, avg_reward, round_num)

        # 8. Record role distribution snapshot
        role_counts = defaultdict(int)
        for agent in env.agents:
            role_counts[agent.role] += 1
        role_history.append({'round': round_num, 'role_counts': dict(role_counts)})

        # 9. Print progress every 10 rounds
        if round_num % 10 == 0 or round_num == 1:
            strat_display = STRATEGY_NAMES_SHORT[strategy_idx]
            print(f"  Round {round_num:>3} | strategy={strat_display:<14} | "
                  f"avg_reward={avg_reward:>7.3f} | "
                  f"best_so_far={bandit.get_best_strategy()[1]}")

    # ── Final summary ───────────────────────────
    print("\n" + bandit.summary())
    print("\n── Agent Final Profiles ──")
    for agent in env.agents:
        print(f"  {agent}  |  avg_reward={agent.avg_performance:.4f}  |  tasks={agent.total_tasks}")

    print(f"\n  First-round avg reward : {round_rewards[0]:.4f}")
    print(f"  Final-round  avg reward: {round_rewards[-1]:.4f}")
    if round_rewards[0] != 0:
        pct = (round_rewards[-1] - round_rewards[0]) / abs(round_rewards[0]) * 100
        print(f"  Improvement            : {pct:+.1f}%")

    return {
        'round_rewards':  round_rewards,
        'role_history':   role_history,
        'usage_counts':   bandit.get_usage_counts(),
        'all_tasks':      all_tasks,
        'env':            env,
        'bandit':         bandit,
    }


STRATEGY_NAMES_SHORT = {0: 'nearest_agent', 1: 'fastest_agent',
                         2: 'least_loaded', 3: 'random'}


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main_headless():
    """Run simulation and save all charts as PNG files."""
    import os
    results = run_simulation()

    out_dir = 'output_charts'
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nSaving charts to ./{out_dir}/")

    plot_warehouse_grid(results['env'], round_num=CONFIG['num_rounds'],
                        save_path=f"{out_dir}/grid_final.png")
    plot_performance_over_time(results['round_rewards'],
                               save_path=f"{out_dir}/performance.png")
    plot_strategy_usage(results['usage_counts'],
                        save_path=f"{out_dir}/strategy_usage.png")
    plot_role_distribution(results['role_history'],
                           save_path=f"{out_dir}/role_distribution.png")
    plot_demand_heatmap(results['all_tasks'], results['env'].grid_size,
                        save_path=f"{out_dir}/demand_heatmap.png")

    print(f"  ✓ grid_final.png")
    print(f"  ✓ performance.png")
    print(f"  ✓ strategy_usage.png")
    print(f"  ✓ role_distribution.png")
    print(f"  ✓ demand_heatmap.png")
    print("\nDone! Open the PNG files to view results.")
    return results


# ── Streamlit mode (streamlit run main.py) ──────
try:
    import streamlit as st
    _streamlit_available = True
except ImportError:
    _streamlit_available = False


if _streamlit_available and hasattr(st, 'session_state'):
    # Running inside Streamlit
    if 'sim_results' not in st.session_state:
        st.session_state.sim_results = run_simulation()
    launch_streamlit_dashboard(st.session_state.sim_results)
else:
    # Running as plain Python script
    if __name__ == '__main__':
        main_headless()
