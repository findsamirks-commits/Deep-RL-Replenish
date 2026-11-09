"""Classical Inventory Replenishment Baselines.

Implements benchmark heuristic policies from classical inventory theory:
1. BaseStockPolicy (Order-up-to (s, S) policy with service-level safety stock)
2. EOQPolicy (Economic Order Quantity with reorder point)
3. ConstantOrderPolicy (Deterministic mean-demand ordering)
4. RandomPolicy (Uniform stochastic baseline)
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


class BaseInventoryPolicy:
    """Abstract base class for inventory replenishment policies."""

    def __init__(self, name: str = "BasePolicy"):
        self.name = name

    def predict(
        self, obs: np.ndarray, env_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Computes replenishment action given current observation.

        Args:
            obs: Flattened observation vector from MultiHubReplenishmentEnv.
            env_info: Optional dict with metadata (n_hubs, n_skus, lead_time, etc.).

        Returns:
            np.ndarray: Reorder quantities of shape (n_hubs * n_skus,).
        """
        raise NotImplementedError


class BaseStockPolicy(BaseInventoryPolicy):
    """(s, S) Order-Up-To Base-Stock Policy.

    Computes target inventory position S = mu_L + z * sigma_L:
    - mu_L: Expected demand over lead time + review period (L + 1) * mu_d
    - sigma_L: Demand volatility over lead time sqrt(L + 1) * sigma_d
    - z: Normal quantile for target cycle service level (e.g., z=1.645 for 95%)

    At each decision epoch:
    - Inventory Position (IP) = On-hand + In-transit pipeline
    - If IP < s, order Q = max(0, S - IP), else order 0.
    """

    # Standard normal quantile lookup table for common service levels
    Z_TABLE = {
        0.80: 0.8416,
        0.85: 1.0364,
        0.90: 1.2816,
        0.95: 1.6449,
        0.98: 2.0537,
        0.99: 2.3263,
    }

    def __init__(
        self,
        n_skus: int = 8,
        n_hubs: int = 3,
        lead_time_days: int = 2,
        demand_mean: Optional[np.ndarray] = None,
        demand_std: Optional[np.ndarray] = None,
        service_level: float = 0.95,
        max_order: float = 30.0,
    ):
        super().__init__(name=f"BaseStock-(s,S)-{int(service_level*100)}%")
        self.n_skus = n_skus
        self.n_hubs = n_hubs
        self.lead_time = lead_time_days
        self.service_level = service_level
        self.max_order = max_order

        if demand_mean is None:
            self.demand_mean = np.array([5.0] * n_skus, dtype=np.float32)
        else:
            self.demand_mean = np.asarray(demand_mean, dtype=np.float32)

        if demand_std is None:
            self.demand_std = np.array([2.0] * n_skus, dtype=np.float32)
        else:
            self.demand_std = np.asarray(demand_std, dtype=np.float32)

        # Approximate z-score from table or standard norm approximation
        self.z = self._get_z(service_level)

        # Compute lead-time parameters
        # Total protection period = lead_time + 1 review day
        self.protection_period = self.lead_time + 1
        self.mu_L = self.protection_period * self.demand_mean
        self.sigma_L = np.sqrt(self.protection_period) * self.demand_std

        # Target order-up-to level S and reorder point s
        self.safety_stock = self.z * self.sigma_L
        self.S = self.mu_L + self.safety_stock
        self.s = self.mu_L * 0.75 + self.safety_stock

    def _get_z(self, sl: float) -> float:
        """Finds closest z-score from standard normal quantile mapping."""
        closest_sl = min(self.Z_TABLE.keys(), key=lambda k: abs(k - sl))
        return self.Z_TABLE[closest_sl]

    def predict(
        self, obs: np.ndarray, env_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Generates order-up-to actions for each hub-SKU pair."""
        features_per_unit = 1 + self.lead_time + 3 + 1
        orders = np.zeros((self.n_hubs, self.n_skus), dtype=np.float32)

        idx = 0
        for h in range(self.n_hubs):
            for s in range(self.n_skus):
                feat = obs[idx : idx + features_per_unit]
                idx += features_per_unit

                on_hand = feat[0]
                in_transit = np.sum(feat[1 : 1 + self.lead_time])
                inventory_position = on_hand + in_transit

                target_S = self.S[s]
                reorder_s = self.s[s]

                if inventory_position < reorder_s:
                    qty = max(0.0, target_S - inventory_position)
                    orders[h, s] = min(qty, self.max_order)
                else:
                    orders[h, s] = 0.0

        return orders.flatten()


class EOQPolicy(BaseInventoryPolicy):
    """Economic Order Quantity (EOQ) Policy with Safety Stock.

    Classical Wilson lot-sizing formula:
    Q* = sqrt(2 * D * K / h)
    where:
    - D: Daily demand rate
    - K: Fixed cost per order
    - h: Daily holding cost per unit

    Reorder Point (ROP) = D * L + SS
    Orders batch Q* whenever Inventory Position <= ROP.
    """

    def __init__(
        self,
        n_skus: int = 8,
        n_hubs: int = 3,
        lead_time_days: int = 2,
        demand_mean: Optional[np.ndarray] = None,
        demand_std: Optional[np.ndarray] = None,
        fixed_order_cost: float = 5.0,
        holding_cost: float = 0.5,
        service_level: float = 0.95,
        max_order: float = 30.0,
    ):
        super().__init__(name="EOQ-Wilson-Batching")
        self.n_skus = n_skus
        self.n_hubs = n_hubs
        self.lead_time = lead_time_days
        self.c_order = fixed_order_cost
        self.c_hold = holding_cost
        self.max_order = max_order

        if demand_mean is None:
            self.demand_mean = np.array([5.0] * n_skus, dtype=np.float32)
        else:
            self.demand_mean = np.asarray(demand_mean, dtype=np.float32)

        if demand_std is None:
            self.demand_std = np.array([2.0] * n_skus, dtype=np.float32)
        else:
            self.demand_std = np.asarray(demand_std, dtype=np.float32)

        # Compute EOQ batch size per SKU
        # Q* = sqrt(2 * D * K / h)
        safe_hold = max(self.c_hold, 1e-4)
        self.eoq_q = np.sqrt((2.0 * self.demand_mean * self.c_order) / safe_hold)
        self.eoq_q = np.clip(self.eoq_q, 1.0, self.max_order)

        # Safety stock and ROP = D * L + z * sigma_L
        z = BaseStockPolicy.Z_TABLE.get(0.95, 1.6449)
        self.rop = (self.demand_mean * self.lead_time) + (z * np.sqrt(self.lead_time) * self.demand_std)

    def predict(
        self, obs: np.ndarray, env_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        features_per_unit = 1 + self.lead_time + 3 + 1
        orders = np.zeros((self.n_hubs, self.n_skus), dtype=np.float32)

        idx = 0
        for h in range(self.n_hubs):
            for s in range(self.n_skus):
                feat = obs[idx : idx + features_per_unit]
                idx += features_per_unit

                on_hand = feat[0]
                in_transit = np.sum(feat[1 : 1 + self.lead_time])
                inventory_position = on_hand + in_transit

                if inventory_position <= self.rop[s]:
                    orders[h, s] = self.eoq_q[s]
                else:
                    orders[h, s] = 0.0

        return orders.flatten()


class ConstantOrderPolicy(BaseInventoryPolicy):
    """Deterministic policy ordering expected daily demand each day."""

    def __init__(
        self,
        n_skus: int = 8,
        n_hubs: int = 3,
        demand_mean: Optional[np.ndarray] = None,
        max_order: float = 30.0,
    ):
        super().__init__(name="Constant-Mean-Demand")
        self.n_skus = n_skus
        self.n_hubs = n_hubs
        self.max_order = max_order
        if demand_mean is None:
            self.demand_mean = np.array([5.0] * n_skus, dtype=np.float32)
        else:
            self.demand_mean = np.asarray(demand_mean, dtype=np.float32)

    def predict(
        self, obs: np.ndarray, env_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        orders = np.zeros((self.n_hubs, self.n_skus), dtype=np.float32)
        for h in range(self.n_hubs):
            orders[h] = np.clip(self.demand_mean, 0.0, self.max_order)
        return orders.flatten()


class RandomPolicy(BaseInventoryPolicy):
    """Uniform random ordering policy across action bounds."""

    def __init__(
        self,
        action_dim: int = 24,
        max_order: float = 30.0,
        seed: int = 42,
    ):
        super().__init__(name="Random-Uniform")
        self.action_dim = action_dim
        self.max_order = max_order
        self.rng = np.random.default_rng(seed)

    def predict(
        self, obs: np.ndarray, env_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        return self.rng.uniform(0.0, self.max_order, size=self.action_dim).astype(np.float32)

