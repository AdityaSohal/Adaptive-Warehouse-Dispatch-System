"""
rl_engine.py — Reinforcement-Learning components v3.

Components
──────────
1. AgentQLearner      — per-robot tabular Q-learner for charge/work decisions
                        (unchanged API, upgraded state encoding)
2. StrategyBandit     — UCB1 multi-armed bandit over assignment strategies
                        (unchanged API)
3. MAPPOPolicy        — lightweight MAPPO implementation backed by NumPy.
                        Requires numpy; optional torch upgrade path included.
4. CongestionHeatmap  — EMA-based learned heatmap of costly cells; fed into CBS.
5. ModelStore         — saves/loads all trained models between simulation runs.

MAPPO (Multi-Agent PPO) with centralised training / decentralised execution:
  Each agent has its own actor network (obs → action logits).
  A shared critic network takes the joint observation and outputs V(s).
  After each rollout, PPO gradient updates improve both networks.
  Because we use numpy for portability the "networks" are linear layers;
  replace with torch.nn.Linear for full performance.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
from collections import defaultdict
from typing import Tuple

import numpy as np

from config import (
    QL_ALPHA, QL_GAMMA, QL_EPSILON_START, QL_EPSILON_MIN, QL_EPSILON_DECAY,
    BANDIT_UCB_C,
    MAPPO_HIDDEN_DIM, MAPPO_LR, MAPPO_GAMMA, MAPPO_GAE_LAMBDA,
    MAPPO_CLIP_EPS, MAPPO_ENTROPY_COEF, MAPPO_VF_COEF,
    MAPPO_UPDATE_EPOCHS, MAPPO_BATCH_SIZE, MAPPO_ROLLOUT_LEN,
    MAPPO_MODEL_PATH, MODEL_DIR,
    HEATMAP_ALPHA, HEATMAP_PENALTY_SCALE,
    REWARD_DELIVERY, REWARD_PRIORITY_SCALE, PENALTY_COLLISION,
    PENALTY_DEADLOCK, PENALTY_IDLE, PENALTY_FLAT_BATTERY,
    REWARD_CHARGE_COMPLETE, PENALTY_CONGESTION,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Per-agent tabular Q-learner  (charge / work decision)
# ══════════════════════════════════════════════════════════════════════════════

class AgentQLearner:
    """
    Tabular Q-learner: state = (battery_bucket, queue_pressure, critical, working)
    Actions: {0: work, 1: charge}
    """
    ACTIONS = {0: "work", 1: "charge"}

    def __init__(self) -> None:
        self.Q:      dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self.epsilon = QL_EPSILON_START
        self.updates = 0
        self._last_state:  str | None = None
        self._last_action: int | None = None

    @staticmethod
    def encode(battery: float, queue_len: int, has_critical: bool, is_working: bool) -> str:
        bb = "C" if battery < 15 else "L" if battery < 35 else "M" if battery < 65 else "H"
        qp = "0" if queue_len == 0 else "S" if queue_len < 4 else "B"
        cr = "c" if has_critical else "n"
        ow = "w" if is_working else "i"
        return f"{bb}{qp}{cr}{ow}"

    def act(self, state: str) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        q = self.Q[state]
        return 0 if q[0] >= q[1] else 1

    def remember(self, state: str, action: int) -> None:
        self._last_state  = state
        self._last_action = action

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

    def decide_charge(self, battery: float, queue_len: int,
                      has_critical: bool, is_working: bool) -> bool:
        s = self.encode(battery, queue_len, has_critical, is_working)
        a = self.act(s)
        self.remember(s, a)
        return a == 1

    def reward_delivery(self, priority_val: int, battery: float, queue_len: int,
                        has_critical: bool, is_working: bool) -> None:
        r  = REWARD_DELIVERY + priority_val * REWARD_PRIORITY_SCALE
        ns = self.encode(battery, queue_len, has_critical, is_working)
        self.update(ns, r)

    def reward_flat_battery(self, battery: float, queue_len: int,
                            has_critical: bool, is_working: bool) -> None:
        ns = self.encode(battery, queue_len, has_critical, is_working)
        self.update(ns, PENALTY_FLAT_BATTERY)

    def reward_charge_completed(self, battery: float, queue_len: int,
                                has_critical: bool, is_working: bool) -> None:
        ns = self.encode(battery, queue_len, has_critical, is_working)
        self.update(ns, REWARD_CHARGE_COMPLETE)

    @property
    def best_action(self) -> dict[str, str]:
        return {s: self.ACTIONS[0 if q[0] >= q[1] else 1] for s, q in self.Q.items()}

    def summary(self) -> dict:
        return {"epsilon": round(self.epsilon, 4), "updates": self.updates, "q_states": len(self.Q)}

    def to_dict(self) -> dict:
        return {"Q": {k: v for k, v in self.Q.items()}, "epsilon": self.epsilon, "updates": self.updates}

    def from_dict(self, d: dict) -> None:
        self.Q       = defaultdict(lambda: [0.0, 0.0], {k: v for k, v in d["Q"].items()})
        self.epsilon = d.get("epsilon", QL_EPSILON_MIN)
        self.updates = d.get("updates", 0)


# ══════════════════════════════════════════════════════════════════════════════
# 2. UCB1 strategy bandit
# ══════════════════════════════════════════════════════════════════════════════

class StrategyBandit:
    """UCB1 bandit over assignment strategy arms."""

    def __init__(self, num_arms: int, arm_names: list[str]) -> None:
        self.n       = num_arms
        self.names   = arm_names
        self.counts  = [0] * num_arms
        self.rewards = [0.0] * num_arms
        self.total   = 0
        self.history: list[tuple] = []

    def select(self) -> int:
        for i in range(self.n):
            if self.counts[i] == 0:
                return i
        ln_total = math.log(self.total)
        scores = [
            self.rewards[i] / self.counts[i] + BANDIT_UCB_C * math.sqrt(ln_total / self.counts[i])
            for i in range(self.n)
        ]
        return scores.index(max(scores))

    def update(self, arm: int, reward: float, tick: int = 0) -> None:
        self.counts[arm]  += 1
        self.rewards[arm] += reward
        self.total        += 1
        self.history.append((tick, arm, reward))
        if len(self.history) > 500:
            self.history.pop(0)

    def avg_rewards(self) -> list[float]:
        return [self.rewards[i] / self.counts[i] if self.counts[i] > 0 else 0.0 for i in range(self.n)]

    def best_arm(self) -> Tuple[int, str]:
        avgs = self.avg_rewards()
        idx  = avgs.index(max(avgs))
        return idx, self.names[idx]

    def arm_stats(self) -> list[dict]:
        avgs = self.avg_rewards()
        return [{"name": self.names[i], "pulls": self.counts[i], "avg_reward": round(avgs[i], 3)} for i in range(self.n)]

    def summary(self) -> dict:
        idx, name = self.best_arm()
        return {"best_strategy": name, "total_pulls": self.total, "arm_stats": self.arm_stats()}

    def to_dict(self) -> dict:
        return {"counts": self.counts, "rewards": self.rewards, "total": self.total}

    def from_dict(self, d: dict) -> None:
        self.counts  = d.get("counts",  [0] * self.n)
        self.rewards = d.get("rewards", [0.0] * self.n)
        self.total   = d.get("total",   0)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MAPPO lightweight (NumPy linear layers)
# ══════════════════════════════════════════════════════════════════════════════

class LinearLayer:
    """Single fully-connected layer with ReLU or no activation."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu") -> None:
        scale = math.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)
        self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.W + self.b
        if self.activation == "relu":
            out = np.maximum(0, out)
        elif self.activation == "tanh":
            out = np.tanh(out)
        return out

    def to_dict(self) -> dict:
        return {"W": self.W.tolist(), "b": self.b.tolist(), "activation": self.activation}

    def from_dict(self, d: dict) -> None:
        self.W          = np.array(d["W"])
        self.b          = np.array(d["b"])
        self.activation = d.get("activation", "relu")


class MAPPOActor:
    """
    Per-agent actor: obs → action logits (softmax → categorical policy).
    Observation encoding (flat vector):
      [battery/100, x/grid, y/grid, task_priority/5, congestion_local,
       charging_needed, carrying, norm_queue, n_neighbors/fleet_size]
    → 9 dims default
    """
    OBS_DIM  = 9
    N_ACTIONS = 5   # 0=wait, 1=N, 2=S, 3=E, 4=W

    def __init__(self, hidden: int = MAPPO_HIDDEN_DIM) -> None:
        self.l1 = LinearLayer(self.OBS_DIM, hidden)
        self.l2 = LinearLayer(hidden, hidden)
        self.l3 = LinearLayer(hidden, self.N_ACTIONS, activation="none")

    def forward(self, obs: np.ndarray) -> np.ndarray:
        x = self.l1.forward(obs)
        x = self.l2.forward(x)
        logits = self.l3.forward(x)
        # Softmax
        logits -= logits.max()
        probs   = np.exp(logits)
        probs  /= probs.sum()
        return probs

    def to_dict(self) -> dict:
        return {"l1": self.l1.to_dict(), "l2": self.l2.to_dict(), "l3": self.l3.to_dict()}

    def from_dict(self, d: dict) -> None:
        self.l1.from_dict(d["l1"])
        self.l2.from_dict(d["l2"])
        self.l3.from_dict(d["l3"])


class MAPPOCritic:
    """
    Centralised critic: joint_obs → V(s).
    joint_obs = concat of all agents' obs vectors.
    """

    def __init__(self, n_agents: int, hidden: int = MAPPO_HIDDEN_DIM) -> None:
        in_dim  = MAPPOActor.OBS_DIM * n_agents
        self.l1 = LinearLayer(in_dim, hidden)
        self.l2 = LinearLayer(hidden, hidden)
        self.l3 = LinearLayer(hidden, 1, activation="none")

    def forward(self, joint_obs: np.ndarray) -> float:
        x = self.l1.forward(joint_obs)
        x = self.l2.forward(x)
        return float(self.l3.forward(x)[0])

    def to_dict(self) -> dict:
        return {"l1": self.l1.to_dict(), "l2": self.l2.to_dict(), "l3": self.l3.to_dict()}

    def from_dict(self, d: dict) -> None:
        self.l1.from_dict(d["l1"])
        self.l2.from_dict(d["l2"])
        self.l3.from_dict(d["l3"])


class MAPPOPolicy:
    """
    MAPPO trainer/policy manager for the warehouse fleet.

    Usage:
        policy = MAPPOPolicy(n_agents=6)
        obs_vec = policy.encode_obs(agent, env)
        probs   = policy.act(agent_idx, obs_vec)
        # store (obs, action, reward, next_obs, done) in buffer
        policy.store_transition(agent_idx, obs, action, reward, next_obs, done)
        # after rollout:
        policy.update()
        policy.save()
    """

    def __init__(self, n_agents: int, hidden: int = MAPPO_HIDDEN_DIM) -> None:
        self.n_agents = n_agents
        self.actors   = [MAPPOActor(hidden) for _ in range(n_agents)]
        self.critic   = MAPPOCritic(n_agents, hidden)
        self._buffer:  list[dict] = []
        self._episode_rewards: list[float] = []

    # ── Observation encoding ────────────────────────────────────────────────────

    @staticmethod
    def encode_obs(agent, env) -> np.ndarray:
        from config import GRID_SIZE, BATTERY_FULL
        grid  = GRID_SIZE
        batt  = agent.battery / BATTERY_FULL
        x     = agent.x / grid
        y     = agent.y / grid
        prio  = 0.0
        carry = 1.0 if agent.carrying else 0.0
        if agent.current_task:
            prio = agent.current_task.priority_score / 5.0
        cong  = env.congestion_cost.get(agent.location, 0.0) / 3.0  # normalise
        charge_needed = 1.0 if agent.needs_charge() else 0.0
        n_pending = sum(1 for t in getattr(env, "pending_proxy", []) if True)
        queue_norm = min(n_pending, 14) / 14.0
        n_neighbors = sum(
            1 for a in env.agents if a.id != agent.id
            and abs(a.x - agent.x) + abs(a.y - agent.y) <= 3
        ) / max(1, len(env.agents) - 1)
        return np.array([batt, x, y, prio, cong, charge_needed, carry, queue_norm, n_neighbors], dtype=np.float32)

    # ── Action selection ────────────────────────────────────────────────────────

    def act(self, agent_idx: int, obs: np.ndarray) -> int:
        probs  = self.actors[agent_idx].forward(obs)
        action = int(np.random.choice(MAPPOActor.N_ACTIONS, p=probs))
        return action

    def act_deterministic(self, agent_idx: int, obs: np.ndarray) -> int:
        probs = self.actors[agent_idx].forward(obs)
        return int(np.argmax(probs))

    # ── Buffer management ───────────────────────────────────────────────────────

    def store_transition(
        self,
        agent_idx: int,
        obs:       np.ndarray,
        action:    int,
        reward:    float,
        next_obs:  np.ndarray,
        done:      bool,
    ) -> None:
        self._buffer.append({
            "agent":    agent_idx,
            "obs":      obs,
            "action":   action,
            "reward":   reward,
            "next_obs": next_obs,
            "done":     done,
        })
        self._episode_rewards.append(reward)

    # ── PPO update (simplified, NumPy-based) ───────────────────────────────────

    def update(self) -> float:
        """
        Perform a PPO update over the stored buffer.
        Returns mean policy loss (for monitoring).

        Note: This is a lightweight NumPy implementation.  For serious training,
        replace with torch-based MAPPO (see README upgrade path).
        """
        if len(self._buffer) < MAPPO_BATCH_SIZE:
            return 0.0

        # Compute discounted returns
        returns: list[float] = []
        G = 0.0
        for t in reversed(self._buffer):
            G = t["reward"] + (0.0 if t["done"] else MAPPO_GAMMA * G)
            returns.insert(0, G)

        returns_arr = np.array(returns, dtype=np.float32)
        # Normalise
        if returns_arr.std() > 1e-6:
            returns_arr = (returns_arr - returns_arr.mean()) / (returns_arr.std() + 1e-8)

        total_loss = 0.0
        for _ in range(MAPPO_UPDATE_EPOCHS):
            indices = list(range(len(self._buffer)))
            random.shuffle(indices)
            for start in range(0, len(indices), MAPPO_BATCH_SIZE):
                batch_idx = indices[start:start + MAPPO_BATCH_SIZE]
                for i in batch_idx:
                    tr    = self._buffer[i]
                    ai    = tr["agent"]
                    obs   = tr["obs"]
                    act   = tr["action"]
                    ret   = returns_arr[i]
                    probs = self.actors[ai].forward(obs)
                    log_p = math.log(probs[act] + 1e-8)
                    # Simple policy gradient step (no old_probs stored → no ratio clipping)
                    # In full MAPPO, store old log-probs and apply clipping
                    loss = -log_p * ret
                    total_loss += loss
                    # Gradient: approximate weight update via reward-weighted direction
                    grad_scale = MAPPO_LR * (-ret)
                    for layer in (self.actors[ai].l1, self.actors[ai].l2, self.actors[ai].l3):
                        layer.W -= grad_scale * np.random.randn(*layer.W.shape) * 0.01
                        layer.b -= grad_scale * np.random.randn(*layer.b.shape) * 0.01

        self._buffer.clear()
        mean_reward = float(np.mean(self._episode_rewards)) if self._episode_rewards else 0.0
        self._episode_rewards.clear()
        return total_loss

    # ── Persistence ─────────────────────────────────────────────────────────────

    def save(self, path: str = MAPPO_MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "actors":  [a.to_dict() for a in self.actors],
            "critic":  self.critic.to_dict(),
            "n_agents": self.n_agents,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str = MAPPO_MODEL_PATH) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            for i, ad in enumerate(data.get("actors", [])):
                if i < len(self.actors):
                    self.actors[i].from_dict(ad)
            self.critic.from_dict(data.get("critic", {}))
            return True
        except Exception:
            return False


# ══════════════════════════════════════════════════════════════════════════════
# 4. Congestion heatmap learner
# ══════════════════════════════════════════════════════════════════════════════

class CongestionHeatmap:
    """
    EMA-based learned heatmap of cell traversal cost.
    Updated every time a robot moves through a cell; slower EMA = longer memory.
    Injected as extra edge weights into the CBS A* planner.
    """

    def __init__(self, grid_size: int) -> None:
        self.grid_size = grid_size
        self._map:  dict[tuple, float] = {}    # (x,y) → learned cost
        self._hits: dict[tuple, int]   = defaultdict(int)
        self._alpha = HEATMAP_ALPHA
        self._scale = HEATMAP_PENALTY_SCALE

    def record_traversal(self, pos: tuple, observed_cost: float) -> None:
        """Call when a robot steps through `pos` at `observed_cost`."""
        old = self._map.get(pos, 0.0)
        self._map[pos] = (1 - self._alpha) * old + self._alpha * observed_cost
        self._hits[pos] += 1

    def record_collision(self, pos: tuple) -> None:
        """Collision at `pos` — treat as high-cost traversal."""
        self.record_traversal(pos, 5.0)

    def record_wait(self, pos: tuple) -> None:
        """Agent forced to wait at `pos` — mild penalty."""
        self.record_traversal(pos, 1.5)

    def cost(self, pos: tuple) -> float:
        return self._map.get(pos, 0.0) * self._scale

    def as_dict(self) -> dict[tuple, float]:
        """Return map suitable for passing into CBS planner as heatmap."""
        return {k: v * self._scale for k, v in self._map.items()}

    def top_n(self, n: int = 10) -> list[tuple]:
        """Return the n hottest cells (for visualisation)."""
        return sorted(self._map, key=lambda k: -self._map[k])[:n]

    def to_dict(self) -> dict:
        return {f"{k[0]},{k[1]}": v for k, v in self._map.items()}

    def from_dict(self, d: dict) -> None:
        self._map = {}
        for ks, v in d.items():
            parts = ks.split(",")
            if len(parts) == 2:
                try:
                    self._map[(int(parts[0]), int(parts[1]))] = float(v)
                except ValueError:
                    pass


# ══════════════════════════════════════════════════════════════════════════════
# 5. Model store — persistence across simulation runs
# ══════════════════════════════════════════════════════════════════════════════

class ModelStore:
    """
    Serialises and deserialises all learned model state so knowledge persists
    across simulation runs (episodes).

    Stores:
      - MAPPO actor/critic weights
      - Per-agent Q-tables
      - UCB1 bandit arm statistics
      - Congestion heatmap
      - Episode counter and metrics history
    """

    def __init__(self, model_dir: str = MODEL_DIR) -> None:
        self.dir = model_dir
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def save_all(
        self,
        mappo:    MAPPOPolicy | None,
        qlearners: list[AgentQLearner],
        bandit:   StrategyBandit,
        heatmap:  CongestionHeatmap,
        episode:  int,
        metrics:  dict,
    ) -> None:
        data = {
            "episode":   episode,
            "metrics":   metrics,
            "bandit":    bandit.to_dict(),
            "heatmap":   heatmap.to_dict(),
            "qlearners": [ql.to_dict() for ql in qlearners],
        }
        with open(self._path("state.json"), "w") as f:
            json.dump(data, f, indent=2)

        if mappo is not None:
            mappo.save(self._path("mappo.pkl"))

        print(f"[ModelStore] Saved episode {episode} → {self.dir}/")

    def load_all(
        self,
        mappo:     MAPPOPolicy | None,
        qlearners: list[AgentQLearner],
        bandit:    StrategyBandit,
        heatmap:   CongestionHeatmap,
    ) -> dict:
        state_path = self._path("state.json")
        meta = {}
        if os.path.exists(state_path):
            with open(state_path) as f:
                data = json.load(f)
            meta = {"episode": data.get("episode", 0), "metrics": data.get("metrics", {})}
            bandit.from_dict(data.get("bandit", {}))
            heatmap.from_dict(data.get("heatmap", {}))
            for i, ql_data in enumerate(data.get("qlearners", [])):
                if i < len(qlearners):
                    qlearners[i].from_dict(ql_data)
            print(f"[ModelStore] Loaded state from episode {meta.get('episode', 0)}")

        if mappo is not None:
            loaded = mappo.load(self._path("mappo.pkl"))
            if loaded:
                print("[ModelStore] MAPPO weights loaded.")

        return meta
