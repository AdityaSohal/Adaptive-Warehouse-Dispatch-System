"""
visualization.py - All plotting and Streamlit dashboard logic.
Provides both standalone matplotlib charts and a Streamlit UI.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────
COLORS = {
    'background': '#0f1117',
    'grid':       '#1e2130',
    'congestion': '#ff4444',
    'pickup':     '#00d4aa',
    'drop':       '#ffd700',
    'agent':      '#4fc3f7',
    'text':       '#e0e0e0',
    'strategies': ['#00d4aa', '#ffd700', '#ff6b6b', '#b39ddb'],
    'roles':      {'generalist': '#888', 'fast_agent': '#4fc3f7',
                   'heavy_agent': '#ff8a65', 'long_distance_agent': '#aed581'},
}

STRATEGY_NAMES = ['nearest_agent', 'fastest_agent', 'least_loaded', 'random']


# ─────────────────────────────────────────────
# GRID VISUALIZATION
# ─────────────────────────────────────────────

def plot_warehouse_grid(env, tasks=None, round_num=0, save_path=None):
    """
    Render the warehouse grid showing:
    - Congestion zones (red)
    - Pickup zones (teal)
    - Drop zones (gold)
    - Agent positions (blue circles with ID)
    - Current tasks (arrows from pickup to drop)
    """
    gs = env.grid_size
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])

    # Draw grid lines
    for i in range(gs + 1):
        ax.axhline(i, color='#2a2d3e', linewidth=0.5)
        ax.axvline(i, color='#2a2d3e', linewidth=0.5)

    # Congestion zones
    for (cx, cy) in env.congestion_zones:
        rect = plt.Rectangle((cx, cy), 1, 1, color=COLORS['congestion'], alpha=0.35, zorder=1)
        ax.add_patch(rect)

    # Pickup zones
    for (px, py) in env.pickup_zones:
        rect = plt.Rectangle((px, py), 1, 1, color=COLORS['pickup'], alpha=0.5, zorder=2)
        ax.add_patch(rect)
        ax.text(px + 0.5, py + 0.5, 'P', ha='center', va='center',
                color='white', fontsize=8, fontweight='bold', zorder=3)

    # Drop zones
    for (dx, dy) in env.drop_zones:
        rect = plt.Rectangle((dx, dy), 1, 1, color=COLORS['drop'], alpha=0.5, zorder=2)
        ax.add_patch(rect)
        ax.text(dx + 0.5, dy + 0.5, 'D', ha='center', va='center',
                color='black', fontsize=8, fontweight='bold', zorder=3)

    # Task arrows
    if tasks:
        for task in tasks:
            px, py = task.pickup
            dx, dy = task.drop
            ax.annotate('',
                xy=(dx + 0.5, dy + 0.5),
                xytext=(px + 0.5, py + 0.5),
                arrowprops=dict(arrowstyle='->', color='white', alpha=0.4, lw=1.5),
                zorder=4
            )

    # Agents
    role_color = COLORS['roles']
    for agent in env.agents:
        color = role_color.get(agent.role, COLORS['agent'])
        circle = plt.Circle((agent.x + 0.5, agent.y + 0.5), 0.38,
                             facecolor=color, zorder=5, linewidth=1.5,
                             edgecolor='white')
        ax.add_patch(circle)
        ax.text(agent.x + 0.5, agent.y + 0.5, str(agent.id),
                ha='center', va='center', color='black',
                fontsize=9, fontweight='bold', zorder=6)

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS['congestion'], alpha=0.5, label='Congestion'),
        mpatches.Patch(color=COLORS['pickup'],     alpha=0.7, label='Pickup Zone'),
        mpatches.Patch(color=COLORS['drop'],       alpha=0.7, label='Drop Zone'),
    ]
    for role, col in role_color.items():
        legend_patches.append(mpatches.Patch(color=col, label=role.replace('_', ' ').title()))

    ax.legend(handles=legend_patches, loc='upper right',
              facecolor='#1a1d2e', labelcolor='white', fontsize=7)

    ax.set_xlim(0, gs)
    ax.set_ylim(0, gs)
    ax.set_xticks(range(gs))
    ax.set_yticks(range(gs))
    ax.tick_params(colors=COLORS['text'], labelsize=7)
    ax.set_title(f'Warehouse Grid — Round {round_num}',
                 color=COLORS['text'], fontsize=13, pad=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig


# ─────────────────────────────────────────────
# PERFORMANCE OVER TIME
# ─────────────────────────────────────────────

def plot_performance_over_time(round_rewards: list, save_path=None):
    """
    Line chart of average reward per round.
    Shows the overall improvement trend.
    """
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])

    rounds = list(range(1, len(round_rewards) + 1))
    ax.plot(rounds, round_rewards, color='#4fc3f7', linewidth=2, label='Avg Reward / Round')

    # Smoothed trend (rolling average window=5)
    if len(round_rewards) >= 5:
        kernel = np.ones(5) / 5
        smoothed = np.convolve(round_rewards, kernel, mode='valid')
        smooth_x = list(range(3, len(round_rewards) - 1))
        ax.plot(smooth_x, smoothed, color='#ffd700', linewidth=2,
                linestyle='--', label='5-Round Moving Avg')

    ax.set_xlabel('Round', color=COLORS['text'])
    ax.set_ylabel('Average Reward', color=COLORS['text'])
    ax.set_title('System Performance Over Time', color=COLORS['text'], fontsize=13)
    ax.tick_params(colors=COLORS['text'])
    ax.spines[:].set_color('#2a2d3e')
    ax.legend(facecolor='#1a1d2e', labelcolor='white')
    ax.grid(True, color='#2a2d3e', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig


# ─────────────────────────────────────────────
# STRATEGY USAGE BAR CHART
# ─────────────────────────────────────────────

def plot_strategy_usage(usage_counts: dict, save_path=None):
    """
    Bar chart showing how often each strategy was selected.
    Dominant bar = strategy the bandit converged to.
    """
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])

    names = list(usage_counts.keys())
    counts = list(usage_counts.values())
    bars = ax.bar(names, counts, color=COLORS['strategies'], edgecolor='white', linewidth=0.8)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', color=COLORS['text'], fontsize=10)

    ax.set_xlabel('Strategy', color=COLORS['text'])
    ax.set_ylabel('Times Selected', color=COLORS['text'])
    ax.set_title('Strategy Usage Frequency (Meta-Learning Convergence)',
                 color=COLORS['text'], fontsize=13)
    ax.tick_params(colors=COLORS['text'])
    ax.spines[:].set_color('#2a2d3e')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right', color=COLORS['text'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig


# ─────────────────────────────────────────────
# AGENT ROLE DISTRIBUTION OVER TIME
# ─────────────────────────────────────────────

def plot_role_distribution(role_history: list, save_path=None):
    """
    Stacked area chart of agent role distribution across rounds.
    Shows emergence of specialization.

    role_history: list of dicts {round: N, role_counts: {role: count}}
    """
    all_roles = ['generalist', 'fast_agent', 'heavy_agent', 'long_distance_agent']
    rounds = [entry['round'] for entry in role_history]

    role_series = {r: [] for r in all_roles}
    for entry in role_history:
        for role in all_roles:
            role_series[role].append(entry['role_counts'].get(role, 0))

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])

    colors = [COLORS['roles'][r] for r in all_roles]
    ax.stackplot(rounds,
                 [role_series[r] for r in all_roles],
                 labels=[r.replace('_', ' ').title() for r in all_roles],
                 colors=colors, alpha=0.85)

    ax.set_xlabel('Round', color=COLORS['text'])
    ax.set_ylabel('Number of Agents', color=COLORS['text'])
    ax.set_title('Agent Role Distribution Over Time (Specialization Emergence)',
                 color=COLORS['text'], fontsize=13)
    ax.tick_params(colors=COLORS['text'])
    ax.spines[:].set_color('#2a2d3e')
    ax.legend(loc='upper left', facecolor='#1a1d2e', labelcolor='white', fontsize=8)
    ax.grid(True, color='#2a2d3e', alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig


# ─────────────────────────────────────────────
# HEATMAP OF TASK DEMAND (BONUS)
# ─────────────────────────────────────────────

def plot_demand_heatmap(task_log: list, grid_size: int, save_path=None):
    """
    Heatmap showing which grid cells were most frequently used as pickup locations.
    """
    heatmap = np.zeros((grid_size, grid_size))
    for task in task_log:
        x, y = task.pickup
        if 0 <= x < grid_size and 0 <= y < grid_size:
            heatmap[y][x] += 1

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])

    im = ax.imshow(heatmap, cmap='YlOrRd', origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax, label='Task Frequency')
    ax.set_title('Pickup Demand Heatmap', color=COLORS['text'], fontsize=13)
    ax.tick_params(colors=COLORS['text'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig


# ─────────────────────────────────────────────
# STREAMLIT DASHBOARD (called from main.py)
# ─────────────────────────────────────────────

def launch_streamlit_dashboard(sim_results: dict):
    """
    This function is called from a Streamlit app context.
    It renders all charts inside the Streamlit UI.

    sim_results keys:
        round_rewards, usage_counts, role_history,
        all_tasks, env, bandit
    """
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed. Run: pip install streamlit")
        return

    st.set_page_config(page_title="Warehouse Dispatch System", layout="wide",
                       page_icon="🤖")

    # ── Header ──────────────────────────────────
    st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    h1, h2, h3 { color: #4fc3f7; }
    .metric-label { color: #aaa !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏭 Self-Learning Warehouse Dispatch System")
    st.caption("Multi-Armed Bandit Meta-Learning + Agent Specialization Simulation")

    # ── KPI Row ──────────────────────────────────
    rr = sim_results['round_rewards']
    bandit = sim_results['bandit']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rounds", len(rr))
    col2.metric("Final Avg Reward", f"{rr[-1]:.3f}" if rr else "N/A")
    col3.metric("Best Strategy", bandit.get_best_strategy()[1].replace('_', ' ').title())
    improvement = ((rr[-1] - rr[0]) / abs(rr[0]) * 100) if rr and rr[0] != 0 else 0
    col4.metric("Improvement", f"{improvement:+.1f}%")

    st.divider()

    # ── Row 1: Grid + Performance ───────────────
    col_a, col_b = st.columns([1, 1.6])
    with col_a:
        st.subheader("🗺️ Warehouse Grid (Final State)")
        fig_grid = plot_warehouse_grid(sim_results['env'], round_num=len(rr))
        st.pyplot(fig_grid)
        plt.close(fig_grid)

    with col_b:
        st.subheader("📈 Performance Over Time")
        fig_perf = plot_performance_over_time(rr)
        st.pyplot(fig_perf)
        plt.close(fig_perf)

    st.divider()

    # ── Row 2: Strategy + Roles ──────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("🎰 Strategy Usage (Bandit Convergence)")
        fig_strat = plot_strategy_usage(sim_results['usage_counts'])
        st.pyplot(fig_strat)
        plt.close(fig_strat)

    with col_d:
        st.subheader("🦾 Agent Role Specialization")
        fig_roles = plot_role_distribution(sim_results['role_history'])
        st.pyplot(fig_roles)
        plt.close(fig_roles)

    st.divider()

    # ── Row 3: Heatmap + Bandit Summary ─────────
    col_e, col_f = st.columns([1, 1])
    with col_e:
        st.subheader("🌡️ Pickup Demand Heatmap")
        fig_heat = plot_demand_heatmap(sim_results['all_tasks'], sim_results['env'].grid_size)
        st.pyplot(fig_heat)
        plt.close(fig_heat)

    with col_f:
        st.subheader("🧠 Bandit Learning Summary")
        st.code(bandit.summary())

        st.subheader("🤖 Final Agent Profiles")
        for agent in sim_results['env'].agents:
            with st.expander(f"Agent {agent.id} — Role: {agent.role}"):
                st.write(f"**Speed:** {agent.speed:.2f}  |  **Capacity:** {agent.capacity:.2f}")
                st.write(f"**Tasks completed:** {agent.total_tasks}")
                st.write(f"**Avg reward:** {agent.avg_performance:.4f}")
                st.write("**Specialization scores:**")
                for cat in ['short', 'long', 'heavy']:
                    n = agent.specialization_counts[cat]
                    avg = agent.avg_score_for_category(cat)
                    st.write(f"  - {cat}: {n} tasks, avg reward {avg:.4f}")
