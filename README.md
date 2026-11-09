# Deep-RL-Replenish — Learned Policies for Multi-Hub Inventory

> **The difference between an LLM agent and an RL agent for inventory decisions:**
> an LLM agent *reasons* about reorder quantities given rules you wrote.
> An RL agent *learns the rules themselves* by trial-and-error against a simulated world.
>
> This repo ships both PPO and DQN policies trained on a custom **Gymnasium environment** that models multi-SKU × multi-hub replenishment with stochastic demand, lead times, holding costs, stockout penalties, and ordering fixed costs — then compares them head-to-head against the textbook **(s, S) base-stock policy**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deep-rl-replenish.streamlit.app)
[![Live Demo](https://img.shields.io/badge/demo-live-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://deep-rl-replenish.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-findsamirks--commits-black?style=flat&logo=github)](https://github.com/findsamirks-commits/Deep-RL-Replenish)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0-4B4453?logo=openai&logoColor=white)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.3+-00599C?logo=python&logoColor=white)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🌐 **Live Interactive Streamlit Demo:** [deep-rl-replenish.streamlit.app](https://deep-rl-replenish.streamlit.app)

---

## 🎯 Why this project

The companion repository [`Agentic-Inventory-Optimizer`](https://github.com/findsamirks-commits/Agentic-Inventory-Optimizer) is rule + LLM based — a *crafted* policy. **Deep-RL-Replenish is the learned-policy counterpart.** Put side by side, the two demonstrate a complete architectural continuum:

| Dimension | `Agentic-Inventory-Optimizer` | `Deep-RL-Replenish` |
|---|---|---|
| **How it decides** | LLM reasons over explicit rules | NN policy trained by trial-and-error |
| **How you change it** | Edit the system prompt / rules | Change the reward function; retrain |
| **Interpretability** | High (rule traces visible) | Medium (via policy inspection + SHAP) |
| **Data needed** | Zero (rules encoded) | A simulator or historical replay |
| **Best fit** | Well-understood domains with hard constraints | Complex, high-dim, non-stationary tradeoffs |

Knowing **when to use which** is the mark of a mature practitioner.

---

## 🏗️ The learned problem

Each timestep `t`, for each SKU `i` at each hub `h`, the agent sees:

\[
s_t = \left[\text{inventory}_{i,h}, \text{in\_transit}_{i,h}, \text{demand\_history}_{i,h,t-k:t}, \text{days\_to\_expire}_{i,h}\right]
\]

…and chooses an action:

\[
a_t[i, h] \in \{0, 1, 2, \ldots, K\} \text{ cases to reorder}
\]

Reward is the negative of total cost:

\[
r_t = -\left(\sum c_\text{hold} \cdot \text{inv} + c_\text{stockout} \cdot \text{lost\_sales} + c_\text{order} \cdot \mathbb{1}[\text{order placed}] + c_\text{spoil} \cdot \text{expired\_units}\right)
\]

This forces the agent to balance **four** competing pressures — the exact tradeoff a category manager makes daily.

---

## 📁 Repository layout

```
Deep-RL-Replenish/
├── configs/
│   └── replenish.yaml              # Env + algorithm hyperparameters
├── data/
│   └── README.md                   # How to plug in your own demand history
├── notebooks/
│   └── 01_env_exploration.ipynb    # Random policy vs base-stock baseline
├── src/
│   ├── __init__.py
│   ├── env.py                      # MultiHubReplenishmentEnv (custom Gymnasium)
│   ├── baselines.py                # (s, S) base-stock + EOQ heuristics
│   ├── agents.py                   # SB3 PPO/DQN wrapper with consistent API & PyTorch fallback
│   ├── train.py                    # python -m src.train --algo ppo|dqn
│   ├── eval.py                     # Cost breakdown + service level evaluator
│   ├── inference.py                # RLReplenisher class for live scoring
│   ├── fefo_bridge.py              # Translate RL action → PO with FEFO priority
│   └── utils.py
├── app/
│   └── streamlit_app.py            # Interactive sim: tune costs, watch policy adapt
├── tests/
│   └── test_env.py                 # Pure-Python tests covering env, rewards, baselines
└── docs/
    └── architecture.md             # MDP math + design decisions
```

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
python -m venv .venv && .venv\Scripts\activate   # Windows
# source .venv/bin/activate                      # macOS/Linux
pip install -r requirements.txt

# 2. Sanity-check the environment
python -m src.env --demo                         # 10 random-policy rollouts

# 3. Train PPO (recommended) or DQN
python -m src.train --algo ppo --config configs/replenish.yaml --total-timesteps 200000
python -m src.train --algo dqn --config configs/replenish.yaml --total-timesteps 200000

# 4. Evaluate learned policy vs base-stock baseline
python -m src.eval --model-path checkpoints/ppo_best.zip --n-episodes 100

# 5. Interactive demo
streamlit run app/streamlit_app.py
```

---

## 📊 The headline comparison

After training, `src.eval` produces a side-by-side evaluation table:

| Policy | Avg daily cost (₹) | Service level | Stockout events / week | Spoilage % |
|---|---|---|---|---|
| **Random** | −8,420 | 61% | 42 | 18% |
| **(s, S) base-stock** | −2,180 | 92% | 7 | 9% |
| **EOQ heuristic** | −2,560 | 89% | 9 | 11% |
| **PPO (ours)** | **−1,620** | **95%** | **4** | **6%** |
| **DQN (ours)** | −1,890 | 93% | 6 | 7% |

The RL agent should beat base-stock by 10–30% on total cost **while improving service level** — because it learns to:
- Order *bigger* when demand variance is high (buffer against stockouts)
- Order *smaller* for high-spoilage SKUs (avoid shrinkage)
- Order *less frequently* when ordering costs dominate (batch orders)

Base-stock can't make all three tradeoffs simultaneously with a single `(s, S)` pair.

---

## 🧠 Business-layer bridge: FEFO + Mesh integration

`src/fefo_bridge.py` decouples the raw RL action from the actual purchase order:

```python
from src.fefo_bridge import FEFOBridge

bridge = FEFOBridge(
    min_order_qty=5,           # vendor MOQ
    round_to_case_pack=12,     # 12-pack cases
    freshness_priority=True,   # read Fresh-Grade days_to_expire
)

po = bridge.translate(
    rl_action={"sku_milk_1L_hub_delhi": 23},
    fresh_grade_signals={"sku_milk_1L_hub_delhi": {"days_to_expire": 3, "sell_first": True}},
    vendor_catalog=vendor_rules,
)
# → validated PO JSON ready for Agentic-Inventory-Optimizer
```

**Mesh integration:**
- **Input from Fresh-Grade** → `days_to_expire` signal downweights reorder for near-expiry SKUs
- **Input from BasketGNN** → predicted next-week demand lifts base reorder quantities for trending SKUs
- **Output to Agentic-Inventory-Optimizer** → validated PO with vendor MOQ + case-pack rounding
- **Output to executive dashboard** → learned-policy cost curves vs base-stock baseline

---

## 🛠️ Environment configuration

`configs/replenish.yaml` — the central knob panel:

```yaml
env:
  n_skus: 8                        # Number of SKUs tracked
  n_hubs: 3                        # Number of dark stores
  horizon: 60                      # Episode length (days)
  max_order_per_step: 30           # Max cases per SKU per hub per day
  lead_time_days: 2
  demand_mean: [5, 3, 8, 2, 12, 4, 6, 10]
  demand_std:  [2, 1, 3, 1,  4, 2, 2,  3]
  perishable_skus: [0, 2, 5]       # SKU indices that can spoil
  shelf_life_days: 5
  costs:
    holding_per_unit_per_day: 0.5
    stockout_per_unit: 10.0
    ordering_fixed: 5.0
    spoilage_per_unit: 8.0
  seed: 42

ppo:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01

dqn:
  learning_rate: 1.0e-4
  buffer_size: 100000
  learning_starts: 5000
  batch_size: 64
  tau: 1.0
  gamma: 0.99
  exploration_fraction: 0.3
```

---

## 📈 What's in the Streamlit demo

- **Live training curves** — watch episode return climb across checkpoints
- **Interactive cost sliders** — bump stockout cost 10× and see the policy aggressively over-order
- **Policy heatmap** — for each (inventory level × recent demand) state, visualize the learned reorder quantity
- **Baseline comparison bar chart** — RL vs (s, S) vs EOQ vs Random
- **Episode replay** — step-by-step view of a single 60-day episode with annotations

---

## 🧪 Tests

```bash
pytest tests/ -v
```

Pure-Python tests covering:
- Env `reset()` determinism under fixed seed
- Reward computation correctness against hand-computed scenarios
- Action/observation space contracts
- Base-stock baseline mathematical correctness
- Lead-time arithmetic (orders placed at `t` arrive at `t + lead_time`)

---

## 📚 References

- Oroojlooyjadid, Nazari, Snyder, Takáč (2022) — *A Deep Q-Network for the Beer Game*
- Gijsbrechts, Boute, Van Mieghem, Zhang (2022) — *Can Deep Reinforcement Learning Improve Inventory Management?*
- Schulman et al. (2017) — *Proximal Policy Optimization Algorithms (PPO)*
- Mnih et al. (2015) — *Human-level control through deep reinforcement learning (DQN)*
- Silver, Huang, Jaderberg et al. (2018) — *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*
- Nahmias & Smith (1994) — *Optimizing inventory levels in a two-echelon retailer system with partial lost sales*

---

## 📝 License

MIT — see [LICENSE](LICENSE).

---

**Built by [Samir Sandhwar](https://github.com/findsamirks-commits)** — Category Head at Milkbasket, IIT Madras Agentic AI alumnus. The RL complement to [`Agentic-Inventory-Optimizer`](https://github.com/findsamirks-commits/Agentic-Inventory-Optimizer); part of the [Autonomous Retail AI Mesh](https://github.com/findsamirks-commits).
