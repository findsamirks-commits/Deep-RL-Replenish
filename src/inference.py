"""Production Inference Engine for RL Replenishment.

Provides the RLReplenisher scoring engine for batch operational reorder generation:
- Accepts operational inventory data (DataFrames or NumPy matrices)
- Constructs standardized environment observation vectors
- Scores with trained PPO/DQN policies (or Base-Stock fallback)
- Outputs tabular Purchase Order recommendations ready for ERP ingestion
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure repo root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import pandas as pd

from src.agents import PPOAgent, DQNAgent
from src.baselines import BaseStockPolicy
from src.utils import load_config


class RLReplenisher:
    """Production replenishment scoring engine."""

    SKU_NAMES = [
        "Fresh Milk 1L",
        "Whole Wheat Bread",
        "Paneer 200g",
        "Salted Butter 100g",
        "Farm Eggs 12pk",
        "Greek Curd 400g",
        "Bananas 1kg",
        "Organic Apples 500g",
    ]

    HUB_NAMES = [
        "DarkStore-DEL-North",
        "DarkStore-GGN-Cyber",
        "DarkStore-NOI-Sec62",
    ]

    def __init__(
        self,
        checkpoint_path: Optional[str] = "checkpoints/ppo_model.pt",
        config_path: str = "configs/replenish.yaml",
        policy_type: str = "ppo",
        n_skus: int = 8,
        n_hubs: int = 3,
        lead_time_days: int = 2,
    ):
        self.n_skus = n_skus
        self.n_hubs = n_hubs
        self.lead_time = lead_time_days
        self.policy_type = policy_type

        # Observation dimension = n_hubs * n_skus * (1 on_hand + lead_time + 3 demand + 1 shelf_life)
        self.features_per_unit = 1 + self.lead_time + 3 + 1
        self.obs_dim = self.n_hubs * self.n_skus * self.features_per_unit
        self.action_dim = self.n_hubs * self.n_skus

        cfg = load_config(config_path) if os.path.exists(config_path) else {}
        self.demand_mean = np.array(cfg.get("env", {}).get("demand_mean", [5.0] * self.n_skus), dtype=np.float32)
        self.demand_std = np.array(cfg.get("env", {}).get("demand_std", [2.0] * self.n_skus), dtype=np.float32)
        self.max_order = float(cfg.get("env", {}).get("max_order_per_step", 30.0))

        # Initialize policy
        self.agent: Any = None
        if policy_type.lower() == "ppo":
            self.agent = PPOAgent(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                max_action=self.max_order,
            )
        elif policy_type.lower() == "dqn":
            self.agent = DQNAgent(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                max_action=self.max_order,
            )
        else:
            self.agent = BaseStockPolicy(
                n_skus=self.n_skus,
                n_hubs=self.n_hubs,
                lead_time_days=self.lead_time,
                demand_mean=self.demand_mean,
                demand_std=self.demand_std,
            )

        # Attempt to load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path) and hasattr(self.agent, "load"):
            try:
                self.agent.load(checkpoint_path)
                self.is_trained = True
            except Exception:
                self.is_trained = False
        else:
            self.is_trained = False

    def build_obs_vector(
        self,
        on_hand: np.ndarray,
        pipeline: List[np.ndarray],
        demand_history: np.ndarray,
        shelf_life_days: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Constructs standardized observation vector from state tensors."""
        if shelf_life_days is None:
            shelf_life_days = [5.0 if s in [0, 2, 5] else 30.0 for s in range(self.n_skus)]

        obs_features = []
        for h in range(self.n_hubs):
            for s in range(self.n_skus):
                inv = float(on_hand[h, s])
                in_transit = [float(pipeline[l][h, s]) for l in range(self.lead_time)]
                recent_d = [float(demand_history[d, h, s]) for d in range(3)]
                days_left = float(shelf_life_days[s])

                obs_features.extend([inv] + in_transit + recent_d + [days_left])

        return np.array(obs_features, dtype=np.float32)

    def predict_reorder(
        self,
        on_hand: np.ndarray,
        pipeline: Optional[List[np.ndarray]] = None,
        demand_history: Optional[np.ndarray] = None,
        shelf_life_days: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Generates structured replenishment orders for all dark stores and SKUs.

        Args:
            on_hand: Matrix [n_hubs, n_skus] of current physical inventory.
            pipeline: List of L matrices [n_hubs, n_skus] of in-transit orders.
            demand_history: Array [3, n_hubs, n_skus] of last 3 days POS sales.
            shelf_life_days: List of remaining days-to-expire per SKU.

        Returns:
            pd.DataFrame: Tabular reorder recommendations.
        """
        if pipeline is None:
            pipeline = [np.zeros((self.n_hubs, self.n_skus), dtype=np.float32) for _ in range(self.lead_time)]
        if demand_history is None:
            demand_history = np.zeros((3, self.n_hubs, self.n_skus), dtype=np.float32)
            for d in range(3):
                for h in range(self.n_hubs):
                    demand_history[d, h] = self.demand_mean

        obs = self.build_obs_vector(on_hand, pipeline, demand_history, shelf_life_days)
        raw_actions = self.agent.predict(obs, deterministic=True)
        order_matrix = raw_actions.reshape(self.n_hubs, self.n_skus)

        records = []
        for h in range(self.n_hubs):
            hub_name = self.HUB_NAMES[h] if h < len(self.HUB_NAMES) else f"Hub_{h}"
            for s in range(self.n_skus):
                sku_name = self.SKU_NAMES[s] if s < len(self.SKU_NAMES) else f"SKU_{s}"
                curr_inv = float(on_hand[h, s])
                in_trans = float(sum(pipeline[l][h, s] for l in range(self.lead_time)))
                rec_order = float(order_matrix[h, s])

                records.append({
                    "hub_id": h,
                    "hub_name": hub_name,
                    "sku_id": s,
                    "sku_name": sku_name,
                    "on_hand_inventory": round(curr_inv, 1),
                    "in_transit_pipeline": round(in_trans, 1),
                    "inventory_position": round(curr_inv + in_trans, 1),
                    "mean_daily_demand": float(self.demand_mean[s]),
                    "recommended_reorder_qty": round(rec_order, 1),
                    "order_trigger": rec_order > 0.5,
                })

        return pd.DataFrame(records)


if __name__ == "__main__":
    replenisher = RLReplenisher()
    sample_on_hand = np.array([
        [4.0, 10.0, 3.0, 8.0, 25.0, 2.0, 12.0, 20.0],
        [8.0, 5.0, 2.0, 6.0, 18.0, 6.0, 8.0, 15.0],
        [2.0, 12.0, 7.0, 4.0, 22.0, 3.0, 15.0, 18.0],
    ], dtype=np.float32)

    df_orders = replenisher.predict_reorder(sample_on_hand)
    print("=" * 80)
    print("📦 Sample RL Replenishment Recommendations:")
    print("=" * 80)
    print(df_orders[["hub_name", "sku_name", "inventory_position", "recommended_reorder_qty", "order_trigger"]].head(10).to_string(index=False))
