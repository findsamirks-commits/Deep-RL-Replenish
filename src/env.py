"""Multi-Hub Inventory Replenishment Gymnasium Environment.

Simulates multi-SKU x multi-hub inventory dynamics in a retail quick-commerce network:
- Stochastic demand sampling per SKU per hub
- In-transit lead-time delivery pipelines (delayed arrivals)
- Perishable shelf-life batch decay and spoilage
- Four competing cost pressures: holding, stockout penalties, ordering fixed costs, spoilage
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Graceful fallback for pure-Python environments without gymnasium
    class _MockSpace:
        def __init__(self, shape, low=0.0, high=1.0, dtype=np.float32):
            self.shape = shape
            self.low = low
            self.high = high
            self.dtype = dtype

        def sample(self):
            return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

    class _MockGym:
        Env = object
    gym = _MockGym()
    spaces = type("spaces", (), {"Box": _MockSpace, "Discrete": lambda n: _MockSpace((1,), 0, n - 1, np.int64)})


class MultiHubReplenishmentEnv(gym.Env):
    """Multi-SKU x Multi-Hub retail inventory management environment."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        n_skus: int = 8,
        n_hubs: int = 3,
        horizon: int = 60,
        max_order_per_step: int = 30,
        lead_time_days: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        env_cfg = config.get("env", {}) if config else {}

        self.n_skus = env_cfg.get("n_skus", n_skus)
        self.n_hubs = env_cfg.get("n_hubs", n_hubs)
        self.horizon = env_cfg.get("horizon", horizon)
        self.max_order = env_cfg.get("max_order_per_step", max_order_per_step)
        self.lead_time = env_cfg.get("lead_time_days", lead_time_days)
        self.seed_val = env_cfg.get("seed", seed)

        # Demand parameters
        d_mean = env_cfg.get("demand_mean", [5.0] * self.n_skus)
        d_std = env_cfg.get("demand_std", [2.0] * self.n_skus)
        self.demand_mean = np.array(d_mean, dtype=np.float32)
        self.demand_std = np.array(d_std, dtype=np.float32)

        # Perishable parameters
        self.perishable_skus = set(env_cfg.get("perishable_skus", [0, 2, 5]))
        self.shelf_life_days = env_cfg.get("shelf_life_days", 5)

        # Cost parameters
        costs = env_cfg.get("costs", {})
        self.c_hold = float(costs.get("holding_per_unit_per_day", 0.5))
        self.c_stockout = float(costs.get("stockout_per_unit", 10.0))
        self.c_order = float(costs.get("ordering_fixed", 5.0))
        self.c_spoil = float(costs.get("spoilage_per_unit", 8.0))

        self.rng = np.random.default_rng(self.seed_val)

        # Action Space: reorder quantity in cases per (SKU, Hub)
        self.total_action_dim = self.n_skus * self.n_hubs
        self.action_space = spaces.Box(
            low=0.0,
            high=float(self.max_order),
            shape=(self.total_action_dim,),
            dtype=np.float32,
        )

        # Observation Space features per (SKU, Hub):
        # 1. On-hand inventory
        # 2..L+1. Pipeline in-transit inventory (L days)
        # L+2..L+4. Recent 3-day demand history
        # L+5. Days-to-expire indicator
        self.features_per_unit = 1 + self.lead_time + 3 + 1
        self.total_obs_dim = self.total_action_dim * self.features_per_unit
        self.observation_space = spaces.Box(
            low=0.0,
            high=1000.0,
            shape=(self.total_obs_dim,),
            dtype=np.float32,
        )

        self.reset(seed=self.seed_val)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets environment state to day 0."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_day = 0

        # On-hand inventory matrix [n_hubs, n_skus] initialized to 2x mean demand
        self.on_hand = np.zeros((self.n_hubs, self.n_skus), dtype=np.float32)
        for h in range(self.n_hubs):
            self.on_hand[h] = np.clip(self.demand_mean * 2.0, 5.0, 50.0)

        # In-transit pipeline queue: list of arrivals scheduled for future days
        # pipeline[lead_day_offset] = matrix [n_hubs, n_skus]
        self.pipeline = [np.zeros((self.n_hubs, self.n_skus), dtype=np.float32) for _ in range(self.lead_time)]

        # Demand history window (3 days)
        self.demand_history = np.zeros((3, self.n_hubs, self.n_skus), dtype=np.float32)

        # Perishable inventory age tracking: age_batches[hub][sku] = list of (quantity, age_days)
        self.age_batches = {
            h: {s: [(float(self.on_hand[h, s]), 1)] if s in self.perishable_skus else [] for s in range(self.n_skus)}
            for h in range(self.n_hubs)
        }

        # Cumulative trackers
        self.cumulative_cost = 0.0
        self.total_demand_served = 0.0
        self.total_demand_requested = 0.0
        self.total_stockouts = 0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        """Flattens environment state into observation vector."""
        obs_features = []
        for h in range(self.n_hubs):
            for s in range(self.n_skus):
                # 1. On-hand
                inv = self.on_hand[h, s]
                # 2. Pipeline orders
                in_transit = [self.pipeline[l][h, s] for l in range(self.lead_time)]
                # 3. Demand history (last 3 days)
                recent_d = [self.demand_history[d, h, s] for d in range(3)]
                # 4. Shelf-life status
                days_left = float(self.shelf_life_days) if s in self.perishable_skus else 30.0

                unit_vector = [inv] + in_transit + recent_d + [days_left]
                obs_features.extend(unit_vector)

        return np.array(obs_features, dtype=np.float32)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advances simulation by 1 operational day."""
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(self.n_hubs, self.n_skus), 0.0, self.max_order)

        # 1. Receive pipeline shipments that completed lead time
        arriving_orders = self.pipeline.pop(0) if len(self.pipeline) > 0 else np.zeros((self.n_hubs, self.n_skus))
        self.on_hand += arriving_orders

        # Register arriving perishables into age tracking
        for h in range(self.n_hubs):
            for s in range(self.n_skus):
                if s in self.perishable_skus and arriving_orders[h, s] > 0:
                    self.age_batches[h][s].append((float(arriving_orders[h, s]), 0))

        # 2. Sample daily customer demand
        daily_demand = np.zeros((self.n_hubs, self.n_skus), dtype=np.float32)
        for h in range(self.n_hubs):
            raw = self.rng.normal(self.demand_mean, self.demand_std)
            daily_demand[h] = np.maximum(0.0, np.round(raw))

        # Roll demand history
        self.demand_history[0] = self.demand_history[1]
        self.demand_history[1] = self.demand_history[2]
        self.demand_history[2] = daily_demand

        # 3. Fulfill demand from on-hand inventory
        sales = np.minimum(self.on_hand, daily_demand)
        lost_sales = daily_demand - sales
        self.on_hand -= sales

        # Update FIFO for perishables
        for h in range(self.n_hubs):
            for s in range(self.n_skus):
                if s in self.perishable_skus:
                    to_fulfill = float(sales[h, s])
                    new_batches = []
                    for qty, age in self.age_batches[h][s]:
                        if to_fulfill >= qty:
                            to_fulfill -= qty
                        else:
                            new_batches.append((qty - to_fulfill, age))
                            to_fulfill = 0.0
                    self.age_batches[h][s] = new_batches

        # 4. Spoilage check for perishables
        daily_spoilage = 0.0
        for h in range(self.n_hubs):
            for s in range(self.n_skus):
                if s in self.perishable_skus:
                    valid_batches = []
                    for qty, age in self.age_batches[h][s]:
                        if age + 1 >= self.shelf_life_days:
                            daily_spoilage += qty
                            self.on_hand[h, s] = max(0.0, self.on_hand[h, s] - qty)
                        else:
                            valid_batches.append((qty, age + 1))
                    self.age_batches[h][s] = valid_batches

        # 5. Place new reorder action into pipeline
        self.pipeline.append(action)

        # 6. Calculate commercial cost components
        holding_cost = float(np.sum(self.on_hand) * self.c_hold)
        stockout_cost = float(np.sum(lost_sales) * self.c_stockout)
        ordering_cost = float(np.sum(action > 0.5) * self.c_order)
        spoilage_cost = float(daily_spoilage * self.c_spoil)

        total_day_cost = holding_cost + stockout_cost + ordering_cost + spoilage_cost
        reward = -total_day_cost

        # Cumulative tracking
        self.cumulative_cost += total_day_cost
        day_demand_sum = float(np.sum(daily_demand))
        day_sales_sum = float(np.sum(sales))
        self.total_demand_requested += day_demand_sum
        self.total_demand_served += day_sales_sum
        if np.sum(lost_sales) > 0.0:
            self.total_stockouts += 1

        self.current_day += 1
        terminated = self.current_day >= self.horizon
        truncated = False

        info = {
            "day": self.current_day,
            "cost_total": total_day_cost,
            "cost_holding": holding_cost,
            "cost_stockout": stockout_cost,
            "cost_ordering": ordering_cost,
            "cost_spoilage": spoilage_cost,
            "units_demanded": day_demand_sum,
            "units_served": day_sales_sum,
            "service_level": (self.total_demand_served / self.total_demand_requested) if self.total_demand_requested > 0 else 1.0,
            "stockout_events": self.total_stockouts,
        }

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gymnasium inventory environment CLI.")
    parser.add_argument("--demo", action="store_true", help="Run 10 sample episodes with random actions")
    args = parser.parse_args()

    env = MultiHubReplenishmentEnv()
    obs, _ = env.reset()
    print("MultiHubReplenishmentEnv Initialized Successfully!")
    print(f"  Observation Dim: {env.observation_space.shape[0]} | Action Dim: {env.action_space.shape[0]}")
    print(f"  Dark Stores: {env.n_hubs} | SKUs: {env.n_skus} | Lead Time: {env.lead_time} days")

    if args.demo:
        print("\nRunning random-action test rollout...")
        total_r = 0.0
        for _ in range(env.horizon):
            a = env.action_space.sample()
            obs, r, done, _, info = env.step(a)
            total_r += r
            if done:
                break
        print(f"Random Policy 60-Day Cost: ₹{-total_r:,.0f} | Final Service Level: {info['service_level']*100:.1f}%")
