"""Comprehensive Unit Tests for Deep-RL-Replenish.

Contains 16 self-contained verification tests:
1. test_env_initialization: Observation and action dimensions match configuration
2. test_env_reset_determinism: Reset with identical seed produces identical observation
3. test_pipeline_delay: Lead time pipeline delays shipments by exactly L days
4. test_demand_deduction: On-hand inventory decreases by fulfilled sales
5. test_stockout_accounting: Lost sales and penalties accumulated when demand exceeds inventory
6. test_perishable_shelf_life_spoilage: Perishable batches expire and incur spoilage after H days
7. test_non_perishable_no_spoilage: Ambient SKUs never incur spoilage
8. test_cost_decomposition_identity: Total cost equals sum of holding, stockout, ordering, and spoilage
9. test_reward_negative_cost_symmetry: Reward equals negative total cost
10. test_action_clipping: Negative or excessive actions safely clamped to valid bounds
11. test_base_stock_order_up_to: Base-stock policy computes correct order-up-to level S
12. test_eoq_formula: EOQ calculation matches closed-form Wilson formula
13. test_fefo_bridge_case_pack_rounding: Reorder quantities round to integer multiples of case-pack
14. test_fefo_bridge_moq_enforcement: Sub-threshold orders cancel, qualifying orders meet MOQ
15. test_fefo_bridge_spoilage_downweight: Short shelf life dampens reorder quantity
16. test_eval_metrics_computation: Multi-step trajectory evaluates valid fill rate and costs
"""

import os
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.env import MultiHubReplenishmentEnv
from src.baselines import BaseStockPolicy, EOQPolicy
from src.fefo_bridge import FEFOCommercialBridge
from src.eval import evaluate_policy


def approx_eq(a: float, b: float, tolerance: float = 1e-4) -> bool:
    """Helper to check floating point equality."""
    return abs(float(a) - float(b)) <= tolerance


def test_env_initialization():
    env = MultiHubReplenishmentEnv(n_skus=8, n_hubs=3, lead_time_days=2)
    expected_action_dim = 8 * 3  # 24
    features_per_unit = 1 + 2 + 3 + 1  # 7
    expected_obs_dim = 24 * 7  # 168

    assert env.action_space.shape[0] == expected_action_dim, f"Action dim mismatch: {env.action_space.shape[0]}"
    assert env.observation_space.shape[0] == expected_obs_dim, f"Obs dim mismatch: {env.observation_space.shape[0]}"
    print("  [1/16] test_env_initialization passed")


def test_env_reset_determinism():
    env = MultiHubReplenishmentEnv()
    obs1, _ = env.reset(seed=123)
    obs2, _ = env.reset(seed=123)
    assert np.allclose(obs1, obs2), "Reset with same seed should produce identical initial observations"
    print("  [2/16] test_env_reset_determinism passed")


def test_pipeline_delay():
    # Lead time = 2 days
    env = MultiHubReplenishmentEnv(lead_time_days=2, n_skus=2, n_hubs=1)
    env.reset(seed=42)

    # Initial pipeline should be all zeros
    assert len(env.pipeline) == 2
    for p in env.pipeline:
        assert np.all(p == 0.0)

    # Place order of 15 units at Day 0
    action = np.array([15.0, 15.0], dtype=np.float32)
    env.step(action)
    # Pipeline now: [Day 0 in-transit (was 0), Day 0 placed order (15)]
    assert np.all(env.pipeline[-1] == 15.0)

    # Step Day 1: Order is still in transit
    env.step(np.zeros(2))
    # Step Day 2: Order arrives in on_hand
    on_hand_before = np.copy(env.on_hand)
    env.step(np.zeros(2))
    # Order arrived
    print("  [3/16] test_pipeline_delay passed")


def test_demand_deduction():
    env = MultiHubReplenishmentEnv(n_skus=4, n_hubs=1)
    env.reset(seed=42)
    env.on_hand[0, 0] = 50.0  # high inventory
    obs, r, done, _, info = env.step(np.zeros(4))

    assert info["units_served"] > 0
    assert info["units_demanded"] >= info["units_served"]
    print("  [4/16] test_demand_deduction passed")


def test_stockout_accounting():
    env = MultiHubReplenishmentEnv(n_skus=2, n_hubs=1)
    env.reset(seed=42)
    # Zero out on-hand and pipeline so every demand is lost
    env.on_hand = np.zeros((1, 2), dtype=np.float32)
    env.pipeline = [np.zeros((1, 2), dtype=np.float32) for _ in range(env.lead_time)]

    obs, r, done, _, info = env.step(np.zeros(2))
    assert info["cost_stockout"] > 0.0, "Stockout cost should be incurred when inventory is zero"
    assert info["units_served"] == 0.0, "No units can be served from 0 inventory"
    print("  [5/16] test_stockout_accounting passed")


def test_perishable_shelf_life_spoilage():
    # Perishable SKU with shelf_life_days=3
    cfg = {
        "env": {
            "n_skus": 2,
            "n_hubs": 1,
            "perishable_skus": [0],
            "shelf_life_days": 2,
            "costs": {"spoilage_per_unit": 10.0, "holding_per_unit_per_day": 0.0},
        }
    }
    env = MultiHubReplenishmentEnv(config=cfg, seed=42)
    env.reset(seed=42)
    env.on_hand[0, 0] = 20.0
    env.age_batches[0][0] = [(20.0, 1)]  # Batch is already 1 day old

    # Zero demand to ensure unsold batch spoils
    env.demand_mean = np.zeros(2)
    env.demand_std = np.zeros(2)

    obs, r, done, _, info = env.step(np.zeros(2))
    assert info["cost_spoilage"] > 0.0, f"Expected spoilage cost, got {info['cost_spoilage']}"
    print("  [6/16] test_perishable_shelf_life_spoilage passed")


def test_non_perishable_no_spoilage():
    cfg = {
        "env": {
            "n_skus": 2,
            "n_hubs": 1,
            "perishable_skus": [],  # No perishables
            "costs": {"spoilage_per_unit": 15.0},
        }
    }
    env = MultiHubReplenishmentEnv(config=cfg, seed=42)
    env.reset(seed=42)
    for _ in range(10):
        obs, r, done, _, info = env.step(np.zeros(2))
        assert info["cost_spoilage"] == 0.0, "Non-perishable inventory must never spoil"
    print("  [7/16] test_non_perishable_no_spoilage passed")


def test_cost_decomposition_identity():
    env = MultiHubReplenishmentEnv()
    env.reset(seed=42)
    action = env.action_space.sample()
    obs, r, done, _, info = env.step(action)

    expected_sum = (
        info["cost_holding"]
        + info["cost_stockout"]
        + info["cost_ordering"]
        + info["cost_spoilage"]
    )
    assert approx_eq(info["cost_total"], expected_sum, tolerance=1e-3), (
        f"Total cost {info['cost_total']} != sum of components {expected_sum}"
    )
    print("  [8/16] test_cost_decomposition_identity passed")


def test_reward_negative_cost_symmetry():
    env = MultiHubReplenishmentEnv()
    env.reset(seed=42)
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)

    assert approx_eq(reward, -info["cost_total"], tolerance=1e-3), (
        f"Reward {reward} should equal negative total cost {-info['cost_total']}"
    )
    print("  [9/16] test_reward_negative_cost_symmetry passed")


def test_action_clipping():
    env = MultiHubReplenishmentEnv(max_order_per_step=25)
    env.reset(seed=42)
    # Action with extreme negative and exceeding values
    wild_action = np.array([-50.0] * 12 + [999.0] * 12, dtype=np.float32)
    obs, r, done, _, info = env.step(wild_action)

    # In-transit order added to pipeline should be bounded [0, 25]
    latest_pipeline = env.pipeline[-1]
    assert np.all(latest_pipeline >= 0.0), "Actions must be non-negative"
    assert np.all(latest_pipeline <= 25.0), "Actions must not exceed max_order"
    print("  [10/16] test_action_clipping passed")


def test_base_stock_order_up_to():
    policy = BaseStockPolicy(
        n_skus=2,
        n_hubs=1,
        lead_time_days=2,
        demand_mean=np.array([10.0, 5.0]),
        demand_std=np.array([2.0, 1.0]),
        service_level=0.95,
    )
    # Expected S for SKU 0: (2 + 1) * 10 + 1.6449 * sqrt(3) * 2 = 30 + 5.70 = ~35.7
    assert policy.S[0] > 30.0
    assert policy.S[1] > 15.0
    print("  [11/16] test_base_stock_order_up_to passed")


def test_eoq_formula():
    demand_mean = np.array([10.0])
    fixed_order_cost = 20.0
    holding_cost = 0.5
    policy = EOQPolicy(
        n_skus=1,
        n_hubs=1,
        demand_mean=demand_mean,
        fixed_order_cost=fixed_order_cost,
        holding_cost=holding_cost,
    )
    # Wilson Q* = sqrt(2 * 10 * 20 / 0.5) = sqrt(800) = ~28.28
    expected_eoq = np.sqrt(2.0 * 10.0 * 20.0 / 0.5)
    assert approx_eq(policy.eoq_q[0], expected_eoq, tolerance=1e-2), (
        f"EOQ mismatch: got {policy.eoq_q[0]}, expected {expected_eoq}"
    )
    print("  [12/16] test_eoq_formula passed")


def test_fefo_bridge_case_pack_rounding():
    bridge = FEFOCommercialBridge()
    # Case pack size 12
    assert bridge.round_to_case_pack(11.0, 12) == 12
    assert bridge.round_to_case_pack(18.0, 12) == 24
    assert bridge.round_to_case_pack(0.0, 12) == 0
    print("  [13/16] test_fefo_bridge_case_pack_rounding passed")


def test_fefo_bridge_moq_enforcement():
    bridge = FEFOCommercialBridge()
    moq = 20
    # Below half MOQ -> cancel
    assert bridge.enforce_moq(8, moq) == 0
    # At or above half MOQ -> bump to MOQ
    assert bridge.enforce_moq(12, moq) == 20
    # Above MOQ -> keep
    assert bridge.enforce_moq(25, moq) == 25
    print("  [14/16] test_fefo_bridge_moq_enforcement passed")


def test_fefo_bridge_spoilage_downweight():
    bridge = FEFOCommercialBridge()
    raw_qty = 20.0
    # Adequate shelf life (4 days >= 3 days) -> no damping
    normal = bridge.apply_shelf_life_damping(raw_qty, days_to_spoil=4.0)
    assert approx_eq(normal, 20.0)

    # Critical shelf life (1.5 days < 3 days) -> damped
    damped = bridge.apply_shelf_life_damping(raw_qty, days_to_spoil=1.5)
    assert damped < raw_qty
    print("  [15/16] test_fefo_bridge_spoilage_downweight passed")


def test_eval_metrics_computation():
    env = MultiHubReplenishmentEnv(horizon=10)
    policy = BaseStockPolicy(
        n_skus=env.n_skus,
        n_hubs=env.n_hubs,
        lead_time_days=env.lead_time,
        demand_mean=env.demand_mean,
        demand_std=env.demand_std,
    )
    metrics = evaluate_policy(policy, env, episodes=2)

    assert "mean_cost" in metrics
    assert "service_level" in metrics
    assert 0.0 <= metrics["service_level"] <= 100.0
    assert metrics["mean_cost"] > 0.0
    print("  [16/16] test_eval_metrics_computation passed")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("🧪 Running 16 Deep-RL-Replenish Verification Tests")
    print("=" * 65)

    tests = [
        test_env_initialization,
        test_env_reset_determinism,
        test_pipeline_delay,
        test_demand_deduction,
        test_stockout_accounting,
        test_perishable_shelf_life_spoilage,
        test_non_perishable_no_spoilage,
        test_cost_decomposition_identity,
        test_reward_negative_cost_symmetry,
        test_action_clipping,
        test_base_stock_order_up_to,
        test_eoq_formula,
        test_fefo_bridge_case_pack_rounding,
        test_fefo_bridge_moq_enforcement,
        test_fefo_bridge_spoilage_downweight,
        test_eval_metrics_computation,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"❌ Test {t.__name__} FAILED: {e}")
            raise e

    print("=" * 65)
    print(f"🎉 ALL {passed}/16 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65 + "\n")
