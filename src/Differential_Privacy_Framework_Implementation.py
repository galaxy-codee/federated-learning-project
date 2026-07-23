"""
STEP 4: Differential Privacy (DP) Framework Implementation
===========================================================
Adds a Local Differential Privacy backend featuring L2 gradient clipping
and calibrated Gaussian noise injection to simulate secure edge networks.
"""

import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1. Data Setup
# ──────────────────────────────────────────────

def load_mnist():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    return x_train, y_train.astype(np.int32), x_test.reshape(-1, 784).astype(np.float32) / 255.0, y_test.astype(np.int32)

def partition_non_iid(x, y, N=100, seed=42):
    rng = np.random.default_rng(seed)
    num_classes = 10
    class_indices = [np.where(y == c)[0] for c in range(num_classes)]
    for c in range(num_classes): rng.shuffle(class_indices[c])

    digit_pairs = [(i % num_classes, (i + 1) % num_classes) for i in range(N)]
    class_ptr = [0] * num_classes
    samples_per_digit = max(10, min(len(class_indices[c]) for c in range(num_classes)) // (N // num_classes + 1))

    devices = []
    for k in range(N):
        d1, d2 = digit_pairs[k]
        n1 = min(samples_per_digit, len(class_indices[d1]) - class_ptr[d1])
        n2 = min(samples_per_digit, len(class_indices[d2]) - class_ptr[d2])
        idx1 = class_indices[d1][class_ptr[d1]: class_ptr[d1] + n1]
        idx2 = class_indices[d2][class_ptr[d2]: class_ptr[d2] + n2]
        class_ptr[d1] += n1; class_ptr[d2] += n2
        idx = np.concatenate([idx1, idx2])
        rng.shuffle(idx)
        devices.append((x[idx], y[idx]))

    sizes = np.array([len(d[1]) for d in devices], dtype=float)
    return devices, sizes / sizes.sum()

# ──────────────────────────────────────────────
# 2. Optimization Utilities
# ──────────────────────────────────────────────

D_IN, N_CLASS = 784, 10
D_PARAM = D_IN * N_CLASS + N_CLASS

def unpack(w): return w[:7840].reshape(784, 10), w[7840:]
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

def cross_entropy_loss(w, X, y, lam=1e-4):
    W, b = unpack(w)
    probs = softmax(X @ W + b)
    return -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)) + (lam / 2) * np.dot(w[:7840], w[:7840])

def cross_entropy_grad(w, X_batch, y_batch, lam=1e-4):
    W, b = unpack(w)
    probs = softmax(X_batch @ W + b)
    probs[np.arange(len(y_batch)), y_batch] -= 1
    probs /= len(y_batch)
    return np.concatenate([(X_batch.T @ probs + lam * W).ravel(), probs.sum(axis=0)])

def accuracy(w, X, y):
    W, b = unpack(w)
    return np.mean((X @ W + b).argmax(axis=1) == y)

# ──────────────────────────────────────────────
# 3. Local Training Loops (Standard vs Private)
# ──────────────────────────────────────────────

def local_sgd(w_t, X, y, E, eta_round, mu=0.0, batch_size=64, lam=1e-4, rng=None):
    w = w_t.copy()
    for _ in range(E):
        idx = rng.choice(len(y), size=min(batch_size, len(y)), replace=False)
        grad = cross_entropy_grad(w, X[idx], y[idx], lam)
        if mu > 0.0: grad += mu * (w - w_t)
        w -= eta_round * grad
    return w

def local_sgd_private(w_t, X, y, E, eta_round, clip_thresh=1.0, noise_scale=0.01, batch_size=64, lam=1e-4, rng=None):
    w = w_t.copy()
    for _ in range(E):
        idx = rng.choice(len(y), size=min(batch_size, len(y)), replace=False)
        grad = cross_entropy_grad(w, X[idx], y[idx], lam)
        w -= eta_round * grad

    # NEW FEATURE: Local Differential Privacy Post-Processing
    local_update = w - w_t

    # 1. L2 Gradient Clipping
    l2_norm = np.linalg.norm(local_update)
    if l2_norm > clip_thresh:
        local_update = local_update * (clip_thresh / l2_norm)

    # 2. Gaussian Noise Addition
    noise = rng.normal(0, noise_scale, size=local_update.shape)

    return w_t + local_update + noise

# ──────────────────────────────────────────────
# 4. Centralized Framework Runner
# ──────────────────────────────────────────────

def run_experiment(mode, devices, p, n_rounds, E, eta0, C=0.2, alpha=None, mu=0.0, beta=0.9,
                   clip_thresh=1.0, noise_scale=0.01, x_test=None, y_test=None, seed=42):
    rng = np.random.default_rng(seed)
    N = len(devices)
    w = np.zeros(D_PARAM)
    loss_curve, acc_curve = [], []
    m = max(1, int(C * N))
    v_velocity = np.zeros(D_PARAM)

    for r in range(n_rounds):
        eta_round = eta0 if r < (n_rounds // 2) else eta0 * 0.5
        active_indices = rng.choice(N, size=m, replace=False)
        active_p_norm = p[active_indices] / p[active_indices].sum()

        v = []
        for idx in active_indices:
            X_k, y_k = devices[idx]
            if mode == 'dp_fedavg':
                v_k = local_sgd_private(w, X_k, y_k, E, eta_round, clip_thresh, noise_scale, 64, 1e-4, rng)
            else:
                v_k = local_sgd(w, X_k, y_k, E, eta_round, mu, 64, 1e-4, rng)
            v.append(v_k)

        current_agg = sum(active_p_norm[i] * v[i] for i in range(m))

        if mode == 'ema':
            w = current_agg if r == 0 else alpha * current_agg + (1 - alpha) * w
        elif mode == 'fedavgm':
            pseudo_grad = w - current_agg
            v_velocity  = beta * v_velocity + pseudo_grad
            w           = w - v_velocity
        else:
            w = current_agg

        loss_curve.append(cross_entropy_loss(w, x_test, y_test))
        acc_curve.append(accuracy(w, x_test, y_test))

        if (r + 1) % 20 == 0:
            print(f"  [{mode.upper()}] Round {r+1:3d}/{n_rounds} | Accuracy: {acc_curve[-1]*100:.2f}%")

    return loss_curve, acc_curve

# ──────────────────────────────────────────────
# 5. Pipeline Execution
# ──────────────────────────────────────────────

if __name__ == "__main__":
    N, n_rounds, eta0, C, E = 100, 100, 0.05, 0.2, 15
    print("="*60); print(f"  STEP 4 DEPLOYED: Differential Privacy Suite (E={E})"); print("="*60)

    x_train, y_train, x_test, y_test = load_mnist()
    devices, p = partition_non_iid(x_train, y_train, N=N)

    print("\nRunning Standard FedAvg Baseline...")
    _, acc_fed = run_experiment('fedavg', devices, p, n_rounds, E, eta0, C, x_test=x_test, y_test=y_test)

    print("\nRunning FedProx Regularization (mu=1.0)...")
    _, acc_prox = run_experiment('fedprox', devices, p, n_rounds, E, eta0, C, mu=1.0, x_test=x_test, y_test=y_test)

    print("\nRunning Server-Side Momentum (FedAvgM, beta=0.9)...")
    _, acc_fedm = run_experiment('fedavgm', devices, p, n_rounds, E, eta0, C, beta=0.9, x_test=x_test, y_test=y_test)

    print("\nRunning Private FedAvg Implementation (Clip=0.5, Noise=0.02)...")
    _, acc_dp = run_experiment('dp_fedavg', devices, p, n_rounds, E, eta0, C, clip_thresh=0.5, noise_scale=0.02, x_test=x_test, y_test=y_test)

    # Plot results
    rounds = np.arange(1, n_rounds + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(rounds, np.array(acc_fed) * 100, label='Standard FedAvg', color='#1f77b4')
    plt.plot(rounds, np.array(acc_prox) * 100, label='FedProx (μ=1.0)', color='#2ca02c')
    plt.plot(rounds, np.array(acc_fedm) * 100, label='FedAvgM (β=0.9)', color='#d62728', linestyle=':')
    plt.plot(rounds, np.array(acc_dp) * 100, label='DP-FedAvg (Private)', color='#9467bd', linewidth=2, linestyle='--')

    plt.title('Global Accuracy with Privacy Constraints', fontsize=12, fontweight='bold')
    plt.xlabel('Communication Rounds'); plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle=':', alpha=0.6); plt.legend(); plt.tight_layout(); plt.show()
    plt.savefig('noise_injestion_privacy_utility_tradeoff.png', dpi=300)
