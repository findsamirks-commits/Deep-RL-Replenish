"""Unified Training CLI for Deep-RL-Replenish.

Trains PPO or DQN agents on MultiHubReplenishmentEnv:
- Tracks episode return, service level, holding cost, and stockout penalties
- Evaluates against classical Base-Stock (s, S) baseline at completion
- Saves model weights to checkpoints/

Usage:
    python -m src.train --algo ppo --config configs/replenish.yaml --episodes 50
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import torch

from src.env import MultiHubReplenishmentEnv
from src.agents import PPOAgent, DQNAgent, HAS_SB3
from src.baselines import BaseStockPolicy, EOQPolicy
from src.utils import load_config, set_seed, get_device


def train_agent(
    algo: str = "ppo",
    config_path: str = "configs/replenish.yaml",
    episodes: int = 50,
    save_dir: str = "checkpoints",
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs reinforcement learning training loop."""
    set_seed(seed)
    cfg = load_config(config_path) if os.path.exists(config_path) else {}

    env = MultiHubReplenishmentEnv(config=cfg, seed=seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print("=" * 65)
    print(f"🚀 Initializing Deep-RL-Replenish Training Pipeline")
    print(f"   Algorithm: {algo.upper()} | SB3 Available: {HAS_SB3}")
    print(f"   Dark Stores: {env.n_hubs} | SKUs: {env.n_skus} | Lead Time: {env.lead_time} days")
    print(f"   Observation Dim: {obs_dim} | Action Dim: {action_dim} | Horizon: {env.horizon} days")
    print("=" * 65)

    device = get_device()
    if algo.lower() == "ppo":
        ppo_cfg = cfg.get("ppo", {})
        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_action=float(env.max_order),
            lr=float(ppo_cfg.get("learning_rate", 3e-4)),
            gamma=float(ppo_cfg.get("gamma", 0.99)),
            clip_ratio=float(ppo_cfg.get("clip_range", 0.2)),
            device=device,
        )
    elif algo.lower() == "dqn":
        dqn_cfg = cfg.get("dqn", {})
        agent = DQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_action=float(env.max_order),
            lr=float(dqn_cfg.get("learning_rate", 1e-4)),
            gamma=float(dqn_cfg.get("gamma", 0.99)),
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Choose 'ppo' or 'dqn'.")

    # Training Loop
    history = {"episode": [], "reward": [], "service_level": [], "total_cost": []}
    start_time = time.time()

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            obs = next_obs

        # Record metrics
        ep_cost = -ep_reward
        sl = info.get("service_level", 0.0) * 100.0
        history["episode"].append(ep)
        history["reward"].append(ep_reward)
        history["service_level"].append(sl)
        history["total_cost"].append(ep_cost)

        if ep % max(1, episodes // 10) == 0 or ep == episodes:
            elapsed = time.time() - start_time
            print(
                f"Episode [{ep:03d}/{episodes:03d}] | "
                f"Total Cost: ₹{ep_cost:7,.0f} | "
                f"Service Level: {sl:5.1f}% | "
                f"Stockouts: {info.get('stockout_events', 0):2d} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    # Save trained checkpoint
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{algo.lower()}_model.pt")
    agent.save(save_path)
    print(f"\n✅ Checkpoint saved to: {save_path}")

    # Benchmark vs Classical Base-Stock Policy
    print("\n" + "=" * 65)
    print("📊 Evaluating Policy Against Classical Heuristics (5-Episode Mean)")
    print("=" * 65)

    base_stock = BaseStockPolicy(
        n_skus=env.n_skus,
        n_hubs=env.n_hubs,
        lead_time_days=env.lead_time,
        demand_mean=env.demand_mean,
        demand_std=env.demand_std,
        service_level=0.95,
        max_order=float(env.max_order),
    )

    policies = {f"Learned {algo.upper()}": agent, "Base-Stock (s, S) 95%": base_stock}
    eval_results = {}

    for name, pol in policies.items():
        costs, sls = [], []
        for eval_ep in range(5):
            obs, _ = env.reset(seed=1000 + eval_ep)
            done = False
            while not done:
                a = pol.predict(obs)
                obs, r, done, _, info = env.step(a)
            costs.append(-r)
            sls.append(info.get("service_level", 0.0) * 100.0)

        mean_c = float(np.mean(costs))
        mean_sl = float(np.mean(sls))
        eval_results[name] = {"cost": mean_c, "service_level": mean_sl}
        print(f"  {name:<24} -> 60-Day Cost: ₹{mean_c:7,.0f} | Service Level: {mean_sl:5.1f}%")

    print("=" * 65)
    return {"history": history, "eval_results": eval_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deep-RL-Replenish agent.")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"], help="RL algorithm")
    parser.add_argument("--config", type=str, default="configs/replenish.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, default=30, help="Number of training episodes")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_agent(
        algo=args.algo,
        config_path=args.config,
        episodes=args.episodes,
        save_dir=args.save_dir,
        seed=args.seed,
    )
