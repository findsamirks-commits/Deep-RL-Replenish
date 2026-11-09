"""FEFO Bridge and Commercial PO Validator.

Bridges raw continuous RL policy decisions with real-world quick-commerce purchasing constraints:
1. Fresh-Grade Shelf-Life Downweighting:
   - Uses predicted days-to-spoil (from Fresh-Grade ViT) to prevent bullwhip spoilage
   - Short remaining shelf life dampens reorder quantity
2. Minimum Order Quantity (MOQ) Enforcement:
   - Enforces supplier minimum dispatch thresholds
3. Case-Pack Rounding:
   - Rounds fractional RL recommendations to integer supplier case sizes (e.g. multiples of 6 or 12)
4. Dark Store Capacity Constraints:
   - Prevents total hub inventory from exceeding dark store cubic pallet capacity
5. Structured PO Generation:
   - Emits validated Purchase Order payloads consumable by Agentic-Inventory-Optimizer / SAP / ERP
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class SupplierContract:
    sku_id: int
    sku_name: str
    vendor_name: str
    case_pack_size: int
    moq_units: int
    cost_per_unit_inr: float
    is_perishable: bool


class FEFOCommercialBridge:
    """Translates mathematical policy actions into validated retail purchase orders."""

    DEFAULT_CONTRACTS: Dict[int, SupplierContract] = {
        0: SupplierContract(0, "Fresh Milk 1L", "Amul Dairy Co-op", case_pack_size=12, moq_units=24, cost_per_unit_inr=56.0, is_perishable=True),
        1: SupplierContract(1, "Whole Wheat Bread", "Harvest Gold Ltd", case_pack_size=6, moq_units=12, cost_per_unit_inr=45.0, is_perishable=False),
        2: SupplierContract(2, "Paneer 200g", "Mother Dairy Fruits & Veg", case_pack_size=8, moq_units=16, cost_per_unit_inr=85.0, is_perishable=True),
        3: SupplierContract(3, "Salted Butter 100g", "Amul Dairy Co-op", case_pack_size=10, moq_units=20, cost_per_unit_inr=58.0, is_perishable=False),
        4: SupplierContract(4, "Farm Eggs 12pk", "Kegg Farms Agro", case_pack_size=6, moq_units=12, cost_per_unit_inr=95.0, is_perishable=False),
        5: SupplierContract(5, "Greek Curd 400g", "Epigamia Foods", case_pack_size=6, moq_units=12, cost_per_unit_inr=70.0, is_perishable=True),
        6: SupplierContract(6, "Bananas 1kg", "Direct Mandi Sourcing", case_pack_size=10, moq_units=20, cost_per_unit_inr=40.0, is_perishable=False),
        7: SupplierContract(7, "Organic Apples 500g", "Himachal Orchards FPO", case_pack_size=8, moq_units=16, cost_per_unit_inr=140.0, is_perishable=False),
    }

    def __init__(
        self,
        contracts: Optional[Dict[int, SupplierContract]] = None,
        max_hub_capacity_units: float = 350.0,
    ):
        self.contracts = contracts or self.DEFAULT_CONTRACTS
        self.max_hub_capacity = max_hub_capacity_units

    def apply_shelf_life_damping(
        self, raw_qty: float, days_to_spoil: float, min_acceptable_days: float = 3.0
    ) -> float:
        """Dampens order quantity if incoming shelf life is critically low."""
        if days_to_spoil >= min_acceptable_days:
            return raw_qty
        # Proportional damping factor between 0.2 and 1.0
        damping_factor = max(0.2, days_to_spoil / max(1.0, min_acceptable_days))
        return raw_qty * damping_factor

    def round_to_case_pack(self, units: float, case_size: int) -> int:
        """Rounds continuous units up/down to nearest integer case-pack size."""
        if units <= 0.0:
            return 0
        num_cases = round(units / case_size)
        return int(num_cases * case_size)

    def enforce_moq(self, units: int, moq: int) -> int:
        """Enforces minimum order quantity. Cancels small noise orders, bumps close orders."""
        if units <= 0:
            return 0
        if units < (moq / 2.0):
            # Order is too small to justify vendor dispatch cost -> cancel
            return 0
        return max(units, moq)

    def convert_actions_to_pos(
        self,
        raw_orders_df: pd.DataFrame,
        shelf_life_estimates: Optional[Dict[int, float]] = None,
    ) -> pd.DataFrame:
        """Transforms unconstrained RL recommendations into commercial Purchase Orders.

        Args:
            raw_orders_df: DataFrame from RLReplenisher containing recommended_reorder_qty.
            shelf_life_estimates: Dict mapping sku_id -> estimated remaining days to spoil.

        Returns:
            pd.DataFrame: Validated PO records with commercial audit trail.
        """
        shelf_life_map = shelf_life_estimates or {0: 4.0, 2: 3.5, 5: 4.5}
        validated_records = []

        for _, row in raw_orders_df.iterrows():
            sku_id = int(row["sku_id"])
            raw_qty = float(row["recommended_reorder_qty"])
            contract = self.contracts.get(sku_id, None)

            if not contract:
                continue

            # 1. Fresh-Grade shelf-life damping
            d_spoil = shelf_life_map.get(sku_id, 30.0)
            damped_qty = self.apply_shelf_life_damping(raw_qty, d_spoil) if contract.is_perishable else raw_qty

            # 2. Case-pack rounding
            cased_qty = self.round_to_case_pack(damped_qty, contract.case_pack_size)

            # 3. Vendor MOQ enforcement
            final_po_qty = self.enforce_moq(cased_qty, contract.moq_units)

            # 4. Total PO value
            total_value_inr = final_po_qty * contract.cost_per_unit_inr

            # Reason code
            if raw_qty > 0.5 and final_po_qty == 0:
                status = "CANCELLED_BELOW_MOQ"
            elif final_po_qty > raw_qty and raw_qty > 0.5:
                status = "BUMPED_TO_MOQ"
            elif final_po_qty > 0:
                status = "APPROVED_PO"
            else:
                status = "NO_ORDER"

            validated_records.append({
                "hub_id": row["hub_id"],
                "hub_name": row["hub_name"],
                "sku_id": sku_id,
                "sku_name": contract.sku_name,
                "vendor_name": contract.vendor_name,
                "raw_rl_recommendation": round(raw_qty, 1),
                "case_pack_size": contract.case_pack_size,
                "vendor_moq": contract.moq_units,
                "final_po_units": final_po_qty,
                "final_po_cases": final_po_qty // contract.case_pack_size if contract.case_pack_size > 0 else 0,
                "unit_cost_inr": contract.cost_per_unit_inr,
                "total_po_value_inr": round(total_value_inr, 2),
                "approval_status": status,
            })

        df_pos = pd.DataFrame(validated_records)

        # 5. Check aggregate hub capacity and scale back if necessary
        for hub_id in df_pos["hub_id"].unique():
            hub_mask = df_pos["hub_id"] == hub_id
            total_ordered = df_pos.loc[hub_mask, "final_po_units"].sum()
            if total_ordered > self.max_hub_capacity:
                scale = self.max_hub_capacity / max(1.0, total_ordered)
                df_pos.loc[hub_mask, "final_po_units"] = (
                    df_pos.loc[hub_mask, "final_po_units"] * scale
                ).astype(int)
                df_pos.loc[hub_mask, "total_po_value_inr"] = (
                    df_pos.loc[hub_mask, "final_po_units"] * df_pos.loc[hub_mask, "unit_cost_inr"]
                )

        return df_pos


if __name__ == "__main__":
    bridge = FEFOCommercialBridge()
    dummy_orders = pd.DataFrame([
        {"hub_id": 0, "hub_name": "DarkStore-DEL-North", "sku_id": 0, "recommended_reorder_qty": 14.2},
        {"hub_id": 0, "hub_name": "DarkStore-DEL-North", "sku_id": 1, "recommended_reorder_qty": 3.1},
        {"hub_id": 0, "hub_name": "DarkStore-DEL-North", "sku_id": 2, "recommended_reorder_qty": 22.8},
    ])
    po_df = bridge.convert_actions_to_pos(dummy_orders)
    print("=" * 85)
    print("📋 Commercial PO Validation Bridge Output:")
    print("=" * 85)
    print(po_df[["sku_name", "raw_rl_recommendation", "final_po_units", "total_po_value_inr", "approval_status"]].to_string(index=False))

