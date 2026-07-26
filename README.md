# FedShield 🛡️
### A Lightweight Framework for Secure, Non-IID Federated Learning with Sequential Domain Adaptation

Built from scratch in raw NumPy/Python (no FL frameworks), implementing and benchmarking **5 federated optimization algorithms** across 100 simulated non-IID edge nodes, with convergence analysis and a deep learning extension using a convolutional U-Net architecture in PyTorch.

> Undergraduate research project conducted under faculty supervision at **IIT Kharagpur**, based on primary literature — McMahan et al. (2017), Li et al. (2020), and Li, Huang, Yang, Wang & Zhang on convergence of FedAvg on non-IID data.

---

## What is Federated Learning?

Federated Learning (FL) is a distributed machine learning paradigm where a central server coordinates model training across many edge devices without raw data ever leaving the device. Each client trains locally and uploads only model weight updates.

**The core challenge:** In the real world, client data is *non-IID* — different devices have fundamentally different data distributions. This statistical heterogeneity causes *client drift*, unstable convergence, and degraded global model performance — which is what this project systematically addresses.

---

## Project Architecture

**Phase A — Convex Baseline Analysis (NumPy)**
A multinomial logistic regression model (single-layer softmax, P = 7,850 parameters) is used to cleanly isolate and benchmark federated optimization algorithms. The strictly convex loss surface ensures measured accuracy differences are caused by algorithmic properties — not random neural network instabilities.

**Phase B — Deep Structural Extension (PyTorch)**
The strongest server-side algorithm (FedAvgM) is extended to a non-linear convolutional U-Net variant (P = 101,182 parameters) to study how federated optimization behaves under non-convex loss landscapes and severe client drift.

---

## Algorithms Implemented

---

**1. Standard FedAvg** — McMahan et al. (2017)

Each round, the server samples a subset of clients, each runs E steps of local SGD, and the server aggregates by weighted average:

```
w_{t+1} = Σ_k (n_k / n) · v_k
```

where `v_k` is client k's local model after E steps, `n_k` is its data size, and `n` is total samples across active clients.

---

**2. EMA-FedAvg** — Server-side temporal smoothing

Instead of replacing the global model outright, the server blends the new aggregate with the previous global model via exponential decay:

```
w_{t+1} = (1 - α) · current_agg  +  α · w_t
```

`α` controls history weight. High α → heavy inertia toward past. O(1) server memory (single state variable) vs O(m) for a fixed window of m rounds.

---

**3. FedProx** — Li et al. (2020)

Modifies each client's local objective by adding a proximal penalty term that penalises deviation from the global model:

```
min_w  F_k(w)  +  (μ/2) · ||w - w_t||²
```

The gradient update each step becomes: `grad_CE + μ · (w - w_t)`. This mathematically leashes how far a client can drift, producing tighter gradient dissimilarity bounds and faster convergence under non-IID data.

---

**4. FedAvgM** — Server-side momentum (β = 0.9)

The server maintains a running velocity vector across rounds. After aggregation it computes a pseudo-gradient and applies Nesterov-style momentum:

```
pseudo_grad  = w_t  -  current_agg
V_{t+1}      = β · V_t  +  pseudo_grad
w_{t+1}      = w_t  -  V_{t+1}
```

The accumulated velocity allows the global model to roll past saddle points and local minima that trap standard FedAvg under high client drift.

---

**5. DP-FedAvg** — Geyer, Klein & Nabi (2017)

After local training, each client applies two privacy operations to its update vector `Δw = w_local - w_t` before transmission:

```
Step 1 — L₂ Clipping:    Δw_clipped = Δw · min(1, G / ||Δw||₂)
Step 2 — Noise Injection: Δw_private = Δw_clipped + N(0, σ²I)
```

Clipping bounds sensitivity to any individual's data; Gaussian noise masks the residual signal. Together they provide client-level differential privacy guarantees against gradient inversion attacks.

---

**6. EWC — Elastic Weight Consolidation** — Kirkpatrick et al. (2017)

When training on a new task, EWC adds an elastic penalty to anchor parameters that were important for previous tasks:

```
L_EWC(w) = L_new(w)  +  (λ/2) · Σ_i F_i · (w_i - w*_i)²
```

where `w*` are the weights after Task 1, `F_i` is the Fisher information diagonal (importance of parameter i), and λ=8.0 controls the strength of the anchor. This prevents Task 2 gradients from overwriting Task 1 knowledge.

---

---

## Convergence Analysis

A central component of this project is the **theoretical convergence analysis of EMA-FedAvg** under non-IID data and partial client participation, grounded in the framework of Li, Huang, Yang, Wang & Zhang.

**EMA-FedAvg — Slower but Bounded Convergence:**
The convergence proof for EMA-FedAvg establishes that the exponential history weighting introduces a memory lag term into the convergence bound. When α is high (heavy history weight), the server accumulates inertia from outdated aggregations. Coupled with a decaying learning rate schedule η_t = η₀/(1+t), the system can enter a *stagnation loop* — the model becomes locked into an outdated trajectory and loses the gradient momentum needed to correct for accumulated client drift. This is analytically consistent with the observed early-peak-and-decay phenomenon in the experimental curves.

The convergence rate of EMA-FedAvg is thus **slower** than standard FedAvg in the non-IID regime, but the smoothed trajectory offers better stability against round-to-round variance from partial participation.

**FedProx — Faster Convergence via Proximal Bounding:**
FedProx introduces a proximal regularization term μ/2 · ||w - w_t||² to each client's local objective. This mathematically restricts how far a client's weights can drift from the global model during local training. Under the convergence analysis of Li et al., this produces a tighter bound on the gradient dissimilarity across clients — directly translating into faster and more stable convergence compared to standard FedAvg on non-IID distributions.

**FedAvgM — Fastest Empirical Convergence:**
Server-side momentum accumulates a velocity vector V_t = β·V_{t-1} + pseudo_gradient across rounds, allowing the global model to build directional momentum and roll past saddle points and local minima that trap standard FedAvg. Under high client drift (E=15 local epochs), FedAvgM achieves the best convergence trajectory of all tested algorithms.

---

## Key Results

### Phase A — Convex Baseline (N=100, Non-IID, E=15, C=0.2)

| Algorithm | Final Global Accuracy | Communication Overhead |
|---|---|---|
| Standard FedAvg | ~82% | 5.99 MB |
| EMA-FedAvg (α=0.3) | ~84% | 5.99 MB |
| FedProx (μ=1.0) | ~85% | 5.99 MB |
| **FedAvgM (β=0.9)** | **88.37%** | 5.99 MB |
| DP-FedAvg | 88.04% | 5.99 MB |

> All advanced variants maintain an identical per-client uplink footprint of **30.66 KB/round** — demonstrating that algorithmic improvements do not require increased communication costs.

**Note on DP-FedAvg convergence behaviour:** DP-FedAvg exhibits a characteristic two-phase curve. In early rounds, aggressive L₂ clipping and Gaussian noise injection blunt the feature-learning vectors at a stage when the global model parameters are still near zero — causing it to lag behind the other algorithms initially. However, as communication rounds progress, the aggregate population consensus trends emerge despite the per-round perturbations, and DP-FedAvg climbs to slightly exceed FedProx and match FedAvgM by the final rounds (88.04%). This late-round recovery is mathematically expected: the noise averages out across many clients and rounds, while the clipping acts as an implicit regularizer that actually helps generalisation under non-IID conditions.

### Continual Learning — EWC vs Catastrophic Forgetting (Task shift: Digits 0–4 → 5–9)

| Method | Task 1 Retention After Task 2 Training |
|---|---|
| Standard (no protection) | 8.33% — Catastrophic Forgetting |
| **EWC Shield (λ=8.0)** | **92.88% — Knowledge Conserved** |

**9× performance retention** demonstrated by EWC regularization under sequential domain shift.

### Phase B — Deep U-Net Extension (PyTorch, FedAvgM, β=0.9)

| Model | Parameters | Per-Client Uplink |
|---|---|---|
| Phase A (softmax) | 7,850 | 30.66 KB |
| Phase B (U-Net) | 101,182 | 395.24 KB |

The deep model plateaus at ~38% under extreme non-IID conditions — intentionally demonstrating the non-convex client drift problem at scale. The convex model's 87% vs the deep model's 38% is a direct empirical illustration of the difference between convex and non-convex optimization surfaces in distributed settings.

---

## Non-IID Data Partitioning

Each of the 100 simulated edge nodes is assigned **only 2 out of 10 digit classes** — an extreme statistical skew matching the experimental setup of McMahan et al. (2017) Section 6. Partial client participation (C=0.2, 20 clients/round) is also simulated, consistent with the convergence framework of Li, Huang, Yang, Wang & Zhang.

---

## Real-World Deep FL Deployment Patterns

The Phase B results (38% accuracy under extreme non-IID) raise a natural question: how do production systems actually deploy deep models in federated settings? The project documents four standard industry approaches:

**FedProx Regularization Leashing** — Rather than letting clients train freely, production systems add a proximal penalty to the local loss function that fires if a client's weights drift too far from the global model. This forces each client to find a solution that is good for its local data but still close to the global consensus — preventing catastrophic blurring during aggregation.

**Pre-training + Transfer Learning** — Deep models in production are almost never randomly initialized on edge devices. Engineers first train the convolutional encoder layers on a large centralized dataset (e.g. ImageNet) in a standard non-federated way, then ship those pre-trained weights to edge devices. The federated workload is then reduced to fine-tuning only the final classification layers — dramatically narrowing the non-convex search space each client has to navigate.

**Clustered Federated Learning (CFL)** — Instead of forcing one global model to reconcile 100 conflicting data distributions, the server groups statistically similar clients into clusters and maintains a separate global model per cluster. This eliminates non-IID conflicts by acknowledging that different edge populations have fundamentally different data realities.

**Split Learning** — For deep networks too large to run on constrained edge hardware, the network is split: the client runs only the lightweight early encoder layers to compress input into abstract features, transmits those features to a powerful cloud server, and the server handles the heavy forward pass and backpropagation — returning only gradients. This allows deep model training without exhausting device memory or battery.

---

## Repository Structure

```
federated-learning-project/
│
├── src/
│   ├── EMA-FedAvg_vs_Standard_FedAvg_baseline.py
│   ├── EMA-FedAvg_vs_Standard_FedAvg_with_Partial_Client_Participation.py
│   ├── fedavg_ema_vs_fedavg_standard_correction_of_early_peak_decay.py
│   ├── fedavg_ema_vs_fedavg_variation_of_parameters_plots.py
│   ├── FedProx_Integration_&_Comparative_Benchmarking_Suite.py
│   ├── Server_Side_Momentum_FedAvgM_Implementation_Suite.py
│   ├── Differential_Privacy_Framework_Implementation.py
│   ├── benchmark_analytics_communication_client_cost_per_round_and_payload.py
│   ├── comprehensive_federated_learning_benchmark_suite.py
│   ├── EWC_consolidated_Federated_continual_learning_eval_wrt_catastrophic_forgetting.py
│   └── federated_deep_learning_involving_mini_UNET_based_structure.py
│
└── plots/
    ├── fedavg_vs_ema_comparison.png
    ├── fedavg_ema_fedprox_comparison.png
    ├── partial_client_participation.png
    ├── high_drift_comparison_using_FedAvgM.png
    ├── noise_injestion_privacy_utility_tradeoff.png
    ├── benchmark_analytics.png
    ├── comprehensive_fl_benchmark.png
    ├── federated_comprehensive_evaluation.png
    ├── fixed_federated_evaluation.png
    ├── EWC_consolidated_FCL_vs_catastrophic_forgetting.png
    └── federated_deep_learning_involving_mini_UNET_based_structure.png
```

---

## Setup

```bash
pip install numpy matplotlib tensorflow torch
```

Phase A runs entirely on NumPy — no GPU required.  
Phase B uses PyTorch — CPU is sufficient for the lightweight U-Net variant.

---

## Running the Experiments

```bash
# Full benchmark suite (all algorithms, communication audit)
python src/comprehensive_federated_learning_benchmark_suite.py

# Continual learning / EWC evaluation
python src/EWC_consolidated_Federated_continual_learning_eval_wrt_catastrophic_forgetting.py

# Deep U-Net federated extension
python src/federated_deep_learning_involving_mini_UNET_based_structure.py
```

---

## References

- McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). **Communication-Efficient Learning of Deep Networks from Decentralized Data.** *AISTATS 2017.*
- Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). **Federated Optimization in Heterogeneous Networks.** *MLSys 2020.*
- Li, X., Huang, K., Yang, W., Wang, S., & Zhang, Z. (2020). **On the Convergence of FedAvg on Non-IID Data.** *ICLR 2020.*
- Geyer, R. C., Klein, T., & Nabi, M. (2017). **Differentially Private Federated Learning: A Client Level Perspective.** *arXiv:1712.07557.* (DP-FedAvg)
- Kirkpatrick, J. et al. (2017). **Overcoming Catastrophic Forgetting in Neural Networks.** *PNAS.*
