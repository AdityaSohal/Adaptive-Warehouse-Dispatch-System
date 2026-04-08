"""
rl_engine.py v5 — MAPPO now ACTUALLY AFFECTS MOVEMENT.

What changed from v4
────────────────────
MAPPOPolicy gains two new methods:

  compute_cost_overlay(agent_idx, grid_size) -> dict[(x,y) -> float]
    After each update, for each sampled grid cell, ask the actor network:
    "how uncertain is the policy about what to do at this cell?"
    We measure this via action-distribution entropy.

    High entropy (≈ ln 5) = policy is confused here = high learned cost
    Low entropy  (≈ 0.0)  = policy is confident     = low cost

    Why entropy?
      A well-trained policy becomes confident in clear corridors and
      uncertain near congestion/collision zones.  Entropy directly encodes
      that signal without needing any extra network or head.

  apply_overlay_to_agents(agents, grid_size, env)
    Pushes the computed overlay into each agent's mappo_overlay dict.
    agent.py reads it in A* via mappo_extra_cost(pos).

  update_and_apply(agents, grid_size, env)
    Single call: train weights + push overlay.  Replace update() with
    this in dispatcher.end_episode().

The rest of the file is identical to v4 (real backpropagation).
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
# 1. Per-agent tabular Q-learner
# ══════════════════════════════════════════════════════════════════════════════

class AgentQLearner:
    ACTIONS = {0: "work", 1: "charge"}

    def __init__(self) -> None:
        self.Q: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self.epsilon  = QL_EPSILON_START
        self.updates  = 0
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
        s, a      = self._last_state, self._last_action
        best_next = max(self.Q[next_state])
        td_target = reward + QL_GAMMA * best_next
        self.Q[s][a] += QL_ALPHA * (td_target - self.Q[s][a])
        self.epsilon   = max(QL_EPSILON_MIN, self.epsilon * QL_EPSILON_DECAY)
        self.updates  += 1
        self._last_state  = None
        self._last_action = None

    def decide_charge(self, battery, queue_len, has_critical, is_working) -> bool:
        s = self.encode(battery, queue_len, has_critical, is_working)
        a = self.act(s)
        self.remember(s, a)
        return a == 1

    def reward_delivery(self, priority_val, battery, queue_len, has_critical, is_working):
        r  = REWARD_DELIVERY + priority_val * REWARD_PRIORITY_SCALE
        ns = self.encode(battery, queue_len, has_critical, is_working)
        self.update(ns, r)

    def reward_flat_battery(self, battery, queue_len, has_critical, is_working):
        ns = self.encode(battery, queue_len, has_critical, is_working)
        self.update(ns, PENALTY_FLAT_BATTERY)

    def reward_charge_completed(self, battery, queue_len, has_critical, is_working):
        ns = self.encode(battery, queue_len, has_critical, is_working)
        self.update(ns, REWARD_CHARGE_COMPLETE)

    @property
    def best_action(self) -> dict:
        return {s: self.ACTIONS[0 if q[0] >= q[1] else 1] for s, q in self.Q.items()}

    def summary(self) -> dict:
        return {"epsilon": round(self.epsilon, 4), "updates": self.updates, "q_states": len(self.Q)}

    def to_dict(self) -> dict:
        return {"Q": dict(self.Q), "epsilon": self.epsilon, "updates": self.updates}

    def from_dict(self, d: dict) -> None:
        self.Q       = defaultdict(lambda: [0.0, 0.0], d["Q"])
        self.epsilon = d.get("epsilon", QL_EPSILON_MIN)
        self.updates = d.get("updates", 0)


# ══════════════════════════════════════════════════════════════════════════════
# 2. UCB1 strategy bandit
# ══════════════════════════════════════════════════════════════════════════════

class StrategyBandit:
    def __init__(self, num_arms: int, arm_names: list[str]) -> None:
        self.n       = num_arms
        self.names   = arm_names
        self.counts  = [0]   * num_arms
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

    def avg_rewards(self) -> list:
        return [self.rewards[i] / self.counts[i] if self.counts[i] > 0 else 0.0 for i in range(self.n)]

    def best_arm(self) -> Tuple[int, str]:
        avgs = self.avg_rewards()
        idx  = avgs.index(max(avgs))
        return idx, self.names[idx]

    def arm_stats(self) -> list:
        avgs = self.avg_rewards()
        return [{"name": self.names[i], "pulls": self.counts[i], "avg_reward": round(avgs[i], 3)} for i in range(self.n)]

    def summary(self) -> dict:
        idx, name = self.best_arm()
        return {"best_strategy": name, "total_pulls": self.total, "arm_stats": self.arm_stats()}

    def to_dict(self) -> dict:
        return {"counts": self.counts, "rewards": self.rewards, "total": self.total}

    def from_dict(self, d: dict) -> None:
        self.counts  = d.get("counts",  [0]   * self.n)
        self.rewards = d.get("rewards", [0.0] * self.n)
        self.total   = d.get("total",   0)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MAPPO with real backpropagation + cost-overlay output
# ══════════════════════════════════════════════════════════════════════════════

class LinearLayer:
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu") -> None:
        scale   = math.sqrt(2.0 / in_dim)
        self.W  = np.random.randn(in_dim, out_dim).astype(np.float32) * scale
        self.b  = np.zeros(out_dim, dtype=np.float32)
        self.activation = activation
        self._x_in: np.ndarray | None = None
        self._z:    np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x_in = x
        z = x @ self.W + self.b
        self._z = z
        if self.activation == "relu":
            return np.maximum(0.0, z)
        elif self.activation == "tanh":
            return np.tanh(z)
        return z

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            d_out = d_out * (self._z > 0).astype(np.float32)
        elif self.activation == "tanh":
            d_out = d_out * (1.0 - np.tanh(self._z) ** 2)
        self._dW = np.outer(self._x_in, d_out)
        self._db = d_out.copy()
        return self.W @ d_out

    def apply_gradients(self, lr: float) -> None:
        self.W -= lr * self._dW
        self.b -= lr * self._db

    def to_dict(self) -> dict:
        return {"W": self.W.tolist(), "b": self.b.tolist(), "activation": self.activation}

    def from_dict(self, d: dict) -> None:
        self.W          = np.array(d["W"], dtype=np.float32)
        self.b          = np.array(d["b"], dtype=np.float32)
        self.activation = d.get("activation", "relu")


class MAPPOActor:
    OBS_DIM   = 9
    N_ACTIONS = 5  # 0=wait, 1=N, 2=S, 3=E, 4=W

    def __init__(self, hidden: int = MAPPO_HIDDEN_DIM) -> None:
        self.l1 = LinearLayer(self.OBS_DIM, hidden,         activation="relu")
        self.l2 = LinearLayer(hidden,       hidden,         activation="relu")
        self.l3 = LinearLayer(hidden,       self.N_ACTIONS, activation="none")

    def forward(self, obs: np.ndarray) -> np.ndarray:
        h1     = self.l1.forward(obs)
        h2     = self.l2.forward(h1)
        logits = self.l3.forward(h2)
        logits = logits - logits.max()
        exp_l  = np.exp(logits)
        return exp_l / exp_l.sum()

    def entropy(self, obs: np.ndarray) -> float:
        """
        Shannon entropy H(π(·|obs)).

        High entropy → policy uncertain here  → will become high A* cost
        Low entropy  → policy confident here  → low A* cost
        """
        probs = self.forward(obs)
        probs = np.clip(probs, 1e-8, 1.0)
        return float(-np.sum(probs * np.log(probs)))

    def backward_and_update(
        self, obs: np.ndarray, action: int, advantage: float, lr: float
    ) -> float:
        # Forward (fills layer caches)
        h1         = self.l1.forward(obs)
        h2         = self.l2.forward(h1)
        logits_raw = self.l3.forward(h2)
        logits_raw = logits_raw - logits_raw.max()
        exp_l      = np.exp(logits_raw)
        probs      = exp_l / exp_l.sum()

        loss = -math.log(float(probs[action]) + 1e-8) * advantage

        # Softmax cross-entropy gradient
        d_logits          = probs.copy()
        d_logits[action] -= 1.0
        d_logits         *= -advantage

        # Backprop l3 → l2 (ReLU) → l1 (ReLU)
        self.l3._z = logits_raw
        d_h2       = self.l3.backward(d_logits);  self.l3.apply_gradients(lr)
        d_h1       = self.l2.backward(d_h2);      self.l2.apply_gradients(lr)
        self.l1.backward(d_h1);                   self.l1.apply_gradients(lr)

        return loss

    def to_dict(self) -> dict:
        return {"l1": self.l1.to_dict(), "l2": self.l2.to_dict(), "l3": self.l3.to_dict()}

    def from_dict(self, d: dict) -> None:
        self.l1.from_dict(d["l1"])
        self.l2.from_dict(d["l2"])
        self.l3.from_dict(d["l3"])


class MAPPOCritic:
    def __init__(self, n_agents: int, hidden: int = MAPPO_HIDDEN_DIM) -> None:
        in_dim  = MAPPOActor.OBS_DIM * n_agents
        self.l1 = LinearLayer(in_dim, hidden, activation="relu")
        self.l2 = LinearLayer(hidden, hidden, activation="relu")
        self.l3 = LinearLayer(hidden, 1,      activation="none")

    def forward(self, joint_obs: np.ndarray) -> float:
        x = self.l1.forward(joint_obs)
        x = self.l2.forward(x)
        return float(self.l3.forward(x)[0])

    def backward_and_update(self, joint_obs, target_value, lr) -> float:
        x1   = self.l1.forward(joint_obs)
        x2   = self.l2.forward(x1)
        v    = float(self.l3.forward(x2)[0])
        loss = 0.5 * (v - target_value) ** 2
        d_v  = np.array([v - target_value], dtype=np.float32)
        self.l3._z = self.l3._z if self.l3._z is not None else d_v
        d_x2 = self.l3.backward(d_v);  self.l3.apply_gradients(lr)
        d_x1 = self.l2.backward(d_x2); self.l2.apply_gradients(lr)
        self.l1.backward(d_x1);        self.l1.apply_gradients(lr)
        return loss

    def to_dict(self) -> dict:
        return {"l1": self.l1.to_dict(), "l2": self.l2.to_dict(), "l3": self.l3.to_dict()}

    def from_dict(self, d: dict) -> None:
        self.l1.from_dict(d["l1"])
        self.l2.from_dict(d["l2"])
        self.l3.from_dict(d["l3"])


class MAPPOPolicy:
    """
    MAPPO — real backpropagation + cost-overlay output connected to A*.

    Full data flow:
      1. Each tick: encode_obs(agent, env) → store_transition(...)
      2. Every ROLLOUT_LEN ticks: update_and_apply(agents, grid_size, env)
         a. update()  — real PPO gradient descent on actors + critic
         b. apply_overlay_to_agents() — compute entropy map → push to agents
      3. In A* (agent.py): edge_cost += agent.mappo_extra_cost(cell)

    As training progresses:
      - Policy becomes confident in open corridors → low entropy → low A* cost
      - Policy stays uncertain near obstacles/congestion → high entropy → high A* cost
      - Agents learn to route through cleaner paths without being explicitly told to
    """

    OVERLAY_SAMPLE_CELLS = 120   # cells to sample per overlay update (speed/quality tradeoff)

    def __init__(self, n_agents: int, hidden: int = MAPPO_HIDDEN_DIM) -> None:
        self.n_agents          = n_agents
        self.actors            = [MAPPOActor(hidden) for _ in range(n_agents)]
        self.critic            = MAPPOCritic(n_agents, hidden)
        self._buffer:          list[dict]  = []
        self._episode_rewards: list[float] = []
        self._last_overlays:   list[dict]  = [{} for _ in range(n_agents)]

    # ── Observation encoding ──────────────────────────────────────────────────

    @staticmethod
    def encode_obs(agent, env) -> np.ndarray:
        from config import GRID_SIZE, BATTERY_FULL
        grid          = GRID_SIZE
        batt          = agent.battery / BATTERY_FULL
        x             = agent.x / grid
        y             = agent.y / grid
        prio          = 0.0
        carry         = 1.0 if agent.carrying else 0.0
        if agent.current_task:
            prio = agent.current_task.priority_score / 5.0
        cong          = env.congestion_cost.get(agent.location, 0.0) / 3.0
        charge_needed = 1.0 if agent.needs_charge() else 0.0
        n_pending     = len(getattr(env, "pending_proxy", []))
        queue_norm    = min(n_pending, 14) / 14.0
        n_neighbors   = sum(
            1 for a in env.agents
            if a.id != agent.id and abs(a.x - agent.x) + abs(a.y - agent.y) <= 3
        ) / max(1, len(env.agents) - 1)
        return np.array(
            [batt, x, y, prio, cong, charge_needed, carry, queue_norm, n_neighbors],
            dtype=np.float32,
        )

    # ── Action selection ──────────────────────────────────────────────────────

    def act(self, agent_idx: int, obs: np.ndarray) -> int:
        probs = self.actors[agent_idx].forward(obs)
        return int(np.random.choice(MAPPOActor.N_ACTIONS, p=probs))

    def act_deterministic(self, agent_idx: int, obs: np.ndarray) -> int:
        return int(np.argmax(self.actors[agent_idx].forward(obs)))

    # ── Buffer ────────────────────────────────────────────────────────────────

    def store_transition(self, agent_idx, obs, action, reward, next_obs, done):
        self._buffer.append({
            "agent": agent_idx, "obs": obs, "action": action,
            "reward": reward, "next_obs": next_obs, "done": done,
        })
        self._episode_rewards.append(reward)

    # ── Cost overlay  ─────────────────────────────────────────────────────────

    def compute_cost_overlay(
        self, agent_idx: int, grid_size: int, env=None
    ) -> dict[tuple, float]:
        """
        Sparse {(x,y): extra_cost} map built from action-distribution entropy.

        Cells where the trained policy is uncertain (high entropy) become
        high-cost in A*.  Cells where it is confident (low entropy) are free.

        After enough training:
          - Open corridors → low entropy → cost ≈ 0   → agents prefer these
          - Congested zones → high entropy → cost > 0 → agents route around
        """
        MAX_OVERLAY_COST = 3.0
        MAX_ENTROPY      = math.log(MAPPOActor.N_ACTIONS)   # ln(5) ≈ 1.609
        ENTROPY_THRESH   = 0.55 * MAX_ENTROPY               # only penalise truly uncertain cells

        actor      = self.actors[agent_idx]
        overlay    = {}
        congestion = {}
        if env is not None:
            congestion = getattr(env, "congestion_cost", {})

        all_cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
        sample    = random.sample(all_cells, min(self.OVERLAY_SAMPLE_CELLS, len(all_cells)))

        for (cx, cy) in sample:
            obs = np.array([
                0.8,                                          # healthy battery
                cx / grid_size,                              # x
                cy / grid_size,                              # y
                0.4,                                          # mid-priority task
                congestion.get((cx, cy), 0.0) / 3.0,        # real congestion
                0.0, 0.0,                                     # not charging, not carrying
                0.5, 0.0,                                     # medium queue, no neighbours
            ], dtype=np.float32)

            ent = actor.entropy(obs)

            if ent > ENTROPY_THRESH:
                norm_cost    = ((ent - ENTROPY_THRESH) / (MAX_ENTROPY - ENTROPY_THRESH + 1e-8)
                                * MAX_OVERLAY_COST)
                cong_factor  = 1.0 + congestion.get((cx, cy), 0.0) / 3.0
                overlay[(cx, cy)] = norm_cost * cong_factor

        return overlay

    def apply_overlay_to_agents(self, agents, grid_size: int, env=None) -> None:
        """Push computed overlays to all agents for use in A*."""
        for i, agent in enumerate(agents):
            overlay = self.compute_cost_overlay(i, grid_size, env)
            self._last_overlays[i] = overlay
            agent.update_mappo_overlay(overlay)

    # ── PPO update — real backprop ────────────────────────────────────────────

    def update(self) -> float:
        if len(self._buffer) < MAPPO_BATCH_SIZE:
            return 0.0

        # Monte-Carlo returns
        returns = []
        G = 0.0
        for tr in reversed(self._buffer):
            G = tr["reward"] + (0.0 if tr["done"] else MAPPO_GAMMA * G)
            returns.insert(0, G)
        returns_arr = np.array(returns, dtype=np.float32)

        if returns_arr.std() > 1e-6:
            advantages = (returns_arr - returns_arr.mean()) / (returns_arr.std() + 1e-8)
        else:
            advantages = returns_arr - returns_arr.mean()

        obs_dim          = MAPPOActor.OBS_DIM
        agent_obs_cache: dict[int, np.ndarray] = {}
        total_actor_loss = 0.0

        for _ in range(MAPPO_UPDATE_EPOCHS):
            indices = list(range(len(self._buffer)))
            random.shuffle(indices)
            for start in range(0, len(indices), MAPPO_BATCH_SIZE):
                for i in indices[start : start + MAPPO_BATCH_SIZE]:
                    tr  = self._buffer[i]
                    ai  = tr["agent"]
                    obs = tr["obs"]
                    act = tr["action"]
                    ret = float(returns_arr[i])
                    adv = float(advantages[i])

                    total_actor_loss += self.actors[ai].backward_and_update(obs, act, adv, MAPPO_LR)

                    agent_obs_cache[ai] = obs
                    joint = np.concatenate([
                        agent_obs_cache.get(j, np.zeros(obs_dim, dtype=np.float32))
                        for j in range(self.n_agents)
                    ])
                    self.critic.backward_and_update(joint, ret, MAPPO_LR * MAPPO_VF_COEF)

        self._buffer.clear()
        self._episode_rewards.clear()
        return total_actor_loss

    def update_and_apply(self, agents, grid_size: int, env=None) -> float:
        """
        Train weights + push cost overlay to agents.
        Use this in dispatcher.end_episode() instead of bare update().
        """
        loss = self.update()
        if loss != 0.0:
            self.apply_overlay_to_agents(agents, grid_size, env)
        return loss

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = MAPPO_MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {"actors": [a.to_dict() for a in self.actors],
                "critic": self.critic.to_dict(), "n_agents": self.n_agents}
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
# 4. Congestion heatmap
# ══════════════════════════════════════════════════════════════════════════════

class CongestionHeatmap:
    def __init__(self, grid_size: int) -> None:
        self.grid_size = grid_size
        self._map:  dict[tuple, float] = {}
        self._hits: dict[tuple, int]   = defaultdict(int)
        self._alpha = HEATMAP_ALPHA
        self._scale = HEATMAP_PENALTY_SCALE

    def record_traversal(self, pos, observed_cost):
        old = self._map.get(pos, 0.0)
        self._map[pos] = (1 - self._alpha) * old + self._alpha * observed_cost
        self._hits[pos] += 1

    def record_collision(self, pos): self.record_traversal(pos, 5.0)
    def record_wait(self, pos):      self.record_traversal(pos, 1.5)
    def cost(self, pos):             return self._map.get(pos, 0.0) * self._scale
    def as_dict(self):               return {k: v * self._scale for k, v in self._map.items()}
    def top_n(self, n=10):           return sorted(self._map, key=lambda k: -self._map[k])[:n]

    def to_dict(self):
        return {f"{k[0]},{k[1]}": v for k, v in self._map.items()}

    def from_dict(self, d):
        self._map = {}
        for ks, v in d.items():
            parts = ks.split(",")
            if len(parts) == 2:
                try:
                    self._map[(int(parts[0]), int(parts[1]))] = float(v)
                except ValueError:
                    pass


# ══════════════════════════════════════════════════════════════════════════════
# 5. Model store
# ══════════════════════════════════════════════════════════════════════════════

class ModelStore:
    def __init__(self, model_dir: str = MODEL_DIR) -> None:
        self.dir = model_dir
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, name): return os.path.join(self.dir, name)

    def save_all(self, mappo, qlearners, bandit, heatmap, episode, metrics):
        data = {
            "episode": episode, "metrics": metrics,
            "bandit":  bandit.to_dict(), "heatmap": heatmap.to_dict(),
            "qlearners": [ql.to_dict() for ql in qlearners],
        }
        with open(self._path("state.json"), "w") as f:
            json.dump(data, f, indent=2)
        if mappo is not None:
            mappo.save(self._path("mappo.pkl"))
        print(f"[ModelStore] Saved episode {episode} → {self.dir}/")

    def load_all(self, mappo, qlearners, bandit, heatmap):
        meta       = {}
        state_path = self._path("state.json")
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
        if mappo is not None and mappo.load(self._path("mappo.pkl")):
            print("[ModelStore] MAPPO weights loaded.")
        return meta
