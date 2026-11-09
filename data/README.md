# Deep-RL-Replenish Demand Data & Simulation Guide

This directory documents how demand signals and supply chain parameters are ingested by the Gymnasium environment.

---

## 1. Built-In Stochastic Demand Generator (Default)

By default, the simulation uses parametric demand distributions configured in `configs/replenish.yaml`:
- **Daily Demand:** Sampled from clipped normal distributions \(\mathcal{N}(\mu_{i}, \sigma_{i}^2)\) or truncated Poisson distributions.
- **Lead Time Delay:** Orders placed at day \(t\) arrive in on-hand inventory at day \(t + L\) (default \(L=2\) days).
- **Perishable Spoilage:** Batch inventory tracks age; inventory exceeding `shelf_life_days` is discarded, incurring spoilage costs.

---

## 2. Ingesting Real Quick-Commerce Historical Demand

To train policies on actual dark-store POS demand history, provide a CSV with this schema:

| Column | Type | Description | Example |
|---|---|---|---|
| `date` | `YYYY-MM-DD` | Timestamp of dark-store operations | `2026-11-01` |
| `sku_id` | `int / str` | Product identifier | `0` or `SKU_AMUL_MILK_1L` |
| `hub_id` | `int / str` | Dark-store fulfillment hub | `0` or `HUB_DELHI_SOUTH` |
| `units_demanded` | `float` | Customer orders placed on that date | `14.0` |
| `actual_lead_time` | `int` | Supplier fulfillment latency in days | `2` |

Place your historical CSV at `data/historical_demand.csv`.
The environment loader will replay empirical demand samples rather than synthetic Gaussians.

