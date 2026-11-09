"""Comprehensive Policy Benchmark and Evaluation Module.

Evaluates reinforcement learning policies against classical inventory heuristics:
- Base-Stock (s, S) 95% Service Level
- Base-Stock (s, S) 99% Service Level
- EOQ (Economic Order Quantity with safety buffer)
- Constant-Mean-Demand Ordering
- Uniform Random Baseline

Computes commercial quick-commerce KPIs:
1. Total Episode Cost (₹)
2. Service Level / Fill Rate (%) = Total Demand Served / Total Demand Requested
3. Cost Breakdown (Holding vs Stockout vs Ordering vs Spoilage)
4. Stockout Day Frequency
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np

from src.env import MultiHubReplenishmentEnv
from src.agents import PPOAgent, DQNAgent
from src.baselines import BaseStockPolicy, EOQPolicy, ConstantOrderPolicy, RandomPolicy
from src.utils import load_config, set_seed


def evaluate_policy(
    policy: Any,
    env: MultiHubReplenishmentEnv,
    episodes: int = 10,
    seed_start: int = 1000,
) -> Dict[str, float]:
    """Runs N evaluation episodes for a given policy and aggregates statistics."""
    total_costs = []
    holding_costs = []
    stockout_costs = []
    ordering_costs = []
    spoilage_costs = []
    service_levels = []
    stockout_days = []
    avg_inventories = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        ep_holding = 0.0
        ep_stockout = 0.0
        ep_ordering = 0.0
        ep_spoilage = 0.0
        ep_cost = 0.0
        daily_invs = []
        done = False

        while not done:
            action = policy.predict(obs)
            daily_invs.append(float(np.sum(env.on_hand)))
            obs, reward, terminated, truncated, info = env.step(action)

            ep_holding += info.get("cost_holding", 0.0)
            ep_stockout += info.get("cost_stockout", 0.0)
            ep_ordering += info.get("cost_ordering", 0.0)
            ep_spoilage += info.get("cost_spoilage", 0.0)
            ep_cost += info.get("cost_total", 0.0)

            done = terminated or truncated

        total_costs.append(ep_cost)
        holding_costs.append(ep_holding)
        stockout_costs.append(ep_stockout)
        ordering_costs.append(ep_ordering)
        spoilage_costs.append(ep_spoilage)
        service_levels.append(info.get("service_level", 0.0) * 100.0)
        stockout_days.append(info.get("stockout_events", 0))
        avg_inventories.append(float(np.mean(daily_invs)))

    return {
        "mean_cost": float(np.mean(total_costs)),
        "std_cost": float(np.std(total_costs)),
        "holding_cost": float(np.mean(holding_costs)),
        "stockout_cost": float(np.mean(stockout_costs)),
        "ordering_cost": float(np.mean(ordering_costs)),
        "spoilage_cost": float(np.mean(spoilage_costs)),
        "service_level": float(np.mean(service_levels)),
        "stockout_days": float(np.mean(stockout_days)),
        "avg_inventory": float(np.mean(avg_inventories)),
    }


def run_benchmark_suite(
    config_path: str = "configs/replenish.yaml",
    episodes: int = 10,
    checkpoint_path: Optional[str] = "checkpoints/ppo_model.pt",
) -> Dict[str, Dict[str, float]]:
    """Runs side-by-side benchmark across all policies."""
    cfg = load_config(config_path) if os.path.exists(config_path) else {}
    env = MultiHubReplenishmentEnv(config=cfg, seed=42)

    # Initialize Policies
    policies = {}

    # 1. PPO Agent
    ppo_agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=float(env.max_order),
    )
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ppo_agent.load(checkpoint_path)
            policies["Learned PPO Policy"] = ppo_agent
        except Exception:
            policies["Learned PPO (Default)"] = ppo_agent
    else:
        policies["Learned PPO (Zero-Shot)"] = ppo_agent

    # 2. Base-Stock (s, S) 95%
    policies["Base-Stock (s, S) 95%"] = BaseStockPolicy(
        n_skus=env.n_skus,
        n_hubs=env.n_hubs,
        lead_time_days=env.lead_time,
        demand_mean=env.demand_mean,
        demand_std=env.demand_std,
        service_level=0.95,
        max_order=float(env.max_order),
    )

    # 3. Base-Stock (s, S) 99%
    policies["Base-Stock (s, S) 99%"] = BaseStockPolicy(
        n_skus=env.n_skus,
        n_hubs=env.n_hubs,
        lead_time_days=env.lead_time,
        demand_mean=env.demand_mean,
        demand_std=env.demand_std,
        service_level=0.99,
        max_order=float(env.max_order),
    )

    # 4. EOQ Batching
    policies["EOQ Wilson Batching"] = EOQPolicy(
        n_skus=env.n_skus,
        n_hubs=env.n_hubs,
        lead_time_days=env.lead_time,
        demand_mean=env.demand_mean,
        demand_std=env.demand_std,
        fixed_order_cost=env.c_order,
        holding_cost=env.c_hold,
        max_order=float(env.max_order),
    )

    # 5. Constant Mean Demand
    policies["Constant Mean Demand"] = ConstantOrderPolicy(
        n_skus=env.n_skus,
        n_hubs=env.n_hubs,
        demand_mean=env.demand_mean,
        max_order=float(env.max_order),
    )

    # 6. Random Baseline
    policies["Random Policy"] = RandomPolicy(
        action_dim=env.action_space.shape[0],
        max_order=float(env.max_order),
    )

    results = {}
    print("\n" + "=" * 90)
    print(f"🏆 Running Multi-Hub Replenishment Benchmark ({episodes} episodes x 60 days)")
    print("=" * 90)
    print(
        f"{'Policy Name':<28} | {'Total Cost (₹)':<14} | {'Service Level':<14} | "
        f"{'Holding (₹)':<12} | {'Stockout (₹)':<12} | {'Spoil (₹)':<10}"
    )
    print("-" * 90)

    for name, pol in policies.items():
        metrics = evaluate_policy(pol, env, episodes=episodes)
        results[name] = metrics
        print(
            f"{name:<28} | ₹{metrics['mean_cost']:11,.0f} | "
            f"{metrics['service_level']:11.1f}% | "
            f"₹{metrics['holding_cost']:9,.0f} | "
            f"₹{metrics['stockout_cost']:9,.0f} | "
            f"₹{metrics['spoilage_cost']:7,.0f}"
        )

    print("=" * 90)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate replenishment policies.")
    parser.add_argument("--config", type=str, default="configs/replenish.yaml", help="Path to config")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ppo_model.pt", help="Path to model checkpoint")
    args = parser.parse_args()

    run_benchmark_suite(
        config_path=args.config,
        episodes=args.episodes,
        checkpoint_path=args.checkpoint,
    )
