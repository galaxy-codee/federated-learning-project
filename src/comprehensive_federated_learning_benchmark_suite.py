"""
Comprehensive Federated Learning Benchmark Suite
================================================
Generates an empirical 4-panel analysis tracking the impacts of Alpha,
Local Steps (E), Client Population (N), and Statistical Divergence (IID vs Non-IID).
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# ──────────────────────────────────────────────
# 1. Core Data & Math Setup
# ──────────────────────────────────────────────

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    x_test  = x_test.reshape(-1, 784).astype(np.float32) / 255.0
    return x_train, y_train.astype(np.int32), x_test, y_test.astype(np.int32)

def partition_data(x, y, N=100, iid=False, seed=42):
    rng = np.random.default_rng(seed)
    if iid:
        indices = np.arange(len(x))
        rng.shuffle(indices)
        chunks = np.array_split(indices, N)
        devices = [(x[chunk], y[chunk]) for chunk in chunks]
    else:
        # Non-IID: 2 digits per device
        class_indices = [np.where(y == c)[0] for c in range(10)]
        for c in range(10):
            rng.shuffle(class_indices[c])

        digit_pairs = [(list(range(10))[i % 10], list(range(10))[(i + 1) % 10]) for i in range(N)]
        class_ptr = [0] * 10
        samples_per_digit = max(10, min(len(class_indices[c]) for c in range(10)) // (N // 10 + 1))

        devices = []
        for k in range(N):
            d1, d2 = digit_pairs[k]
            n1 = min(samples_per_digit, len(class_indices[d1]) - class_ptr[d1])
            n2 = min(samples_per_digit, len(class_indices[d2]) - class_ptr[d2])
            n1, n2 = max(1, n1), max(1, n2)

            idx1 = class_indices[d1][class_ptr[d1]: class_ptr[d1] + n1]
            idx2 = class_indices[d2][class_ptr[d2]: class_ptr[d2] + n2]
            class_ptr[d1] += n1
            class_ptr[d2] += n2

            idx = np.concatenate([idx1, idx2])
            rng.shuffle(idx)
            devices.append((x[idx], y[idx]))

    sizes = np.array([len(d[1]) for d in devices], dtype=float)
    return devices, sizes / sizes.sum()

# Optimization Functions
D_IN, N_CLASS = 784, 10
D_PARAM = D_IN * N_CLASS + N_CLASS

def unpack(w):
    return w[:7840].reshape(784, 10), w[7840:]

def softmax(z):
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_grad(w, X, y, lam=1e-4):
    W, b = unpack(w)
    n = len(y)
    probs = softmax(X @ W + b)
    probs[np.arange(n), y] -= 1
    probs /= n
    return np.concatenate([(X.T @ probs + lam * W).ravel(), probs.sum(axis=0)])

def evaluate_accuracy(w, X, y):
    W, b = unpack(w)
    return np.mean((X @ W + b).argmax(axis=1) == y)

def local_sgd(w_t, X, y, E, eta, batch_size=64, lam=1e-4, rng=None):
    w = w_t.copy()
    n = len(y)
    for _ in range(E):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        w -= eta * cross_entropy_grad(w, X[idx], y[idx], lam)
    return w

# ──────────────────────────────────────────────
# 2. Generalized Training Engine
# ──────────────────────────────────────────────

def run_federated_experiment(devices, p, n_rounds, E, eta0, alpha=None, x_test=None, y_test=None):
    rng = np.random.default_rng(0)
    w = np.zeros(D_PARAM)
    acc_curve = []

    for r in range(n_rounds):
        eta_round = eta0 if r < (n_rounds // 2) else eta0 * 0.5
        v = [local_sgd(w, dev[0], dev[1], E, eta_round, 64, 1e-4, rng) for dev in devices]
        current_agg = sum(p[k] * v[k] for k in range(len(devices)))

        if alpha is None:
            w = current_agg
        else:
            w = current_agg if r == 0 else alpha * current_agg + (1 - alpha) * w

        acc_curve.append(evaluate_accuracy(w, x_test, y_test))
    return np.array(acc_curve) * 100

# ──────────────────────────────────────────────
# 3. Execution Pipeline & Visual Generator
# ──────────────────────────────────────────────

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    n_rounds = 60
    eta0 = 0.05

    plt.figure(figsize=(15, 10))

    # ── PANEL 1: Impact of Alpha (Non-IID, N=50, E=10) ──
    print("Simulating Panel 1: Varying Alpha...")
    devs_p1, p_p1 = partition_data(x_train, y_train, N=50, iid=False)
    acc_std_p1 = run_federated_experiment(devs_p1, p_p1, n_rounds, E=10, eta0=eta0, x_test=x_test, y_test=y_test)
    plt.subplot(2, 2, 1)
    plt.plot(range(1, n_rounds+1), acc_std_p1, label='Standard FedAvg', color='#1f77b4', linewidth=2)
    for a in [0.2, 0.5, 0.7]:
        acc_ema = run_federated_experiment(devs_p1, p_p1, n_rounds, E=10, eta0=eta0, alpha=a, x_test=x_test, y_test=y_test)
        plt.plot(range(1, n_rounds+1), acc_ema, label=f'EMA-FedAvg (α={a})', linestyle='--')
    plt.title('Impact of Alpha (Non-IID, N=50, E=10)', fontweight='bold')
    plt.xlabel('Communication Rounds'); plt.ylabel('Accuracy (%)'); plt.grid(True, linestyle=':'); plt.legend()

    # ── PANEL 2: Impact of Local Steps E (Non-IID, N=50) ──
    print("Simulating Panel 2: Varying Local Steps...")
    plt.subplot(2, 2, 2)
    colors_p2 = {2: '#1f77b4', 10: '#2ca02c', 25: '#9467bd'}
    for e_val in [2, 10, 25]:
        acc_std = run_federated_experiment(devs_p1, p_p1, n_rounds, E=e_val, eta0=eta0, x_test=x_test, y_test=y_test)
        acc_ema = run_federated_experiment(devs_p1, p_p1, n_rounds, E=e_val, eta0=eta0, alpha=0.3, x_test=x_test, y_test=y_test)
        plt.plot(range(1, n_rounds+1), acc_std, label=f'Standard (E={e_val})', color=colors_p2[e_val], linewidth=2)
        plt.plot(range(1, n_rounds+1), acc_ema, label=f'EMA (E={e_val}, α=0.3)', color=colors_p2[e_val], linestyle='--')
    plt.title('Impact of Local Steps E (Non-IID, N=50)', fontweight='bold')
    plt.xlabel('Communication Rounds'); plt.ylabel('Accuracy (%)'); plt.grid(True, linestyle=':'); plt.legend()

    # ── PANEL 3: Scaling Device Population N (Non-IID, E=2) ──
    print("Simulating Panel 3: Scaling Device Population...")
    plt.subplot(2, 2, 3)
    colors_p3 = {20: '#1f77b4', 60: '#d62728', 100: '#9467bd'}
    for n_val in [20, 60, 100]:
        devs_p3, p_p3 = partition_data(x_train, y_train, N=n_val, iid=False)
        acc_std = run_federated_experiment(devs_p3, p_p3, n_rounds, E=2, eta0=eta0, x_test=x_test, y_test=y_test)
        acc_ema = run_federated_experiment(devs_p3, p_p3, n_rounds, E=2, eta0=eta0, alpha=0.3, x_test=x_test, y_test=y_test)
        plt.plot(range(1, n_rounds+1), acc_std, label=f'Standard (N={n_val})', color=colors_p3[n_val], linewidth=2)
        plt.plot(range(1, n_rounds+1), acc_ema, label=f'EMA (N={n_val}, α=0.3)', color=colors_p3[n_val], linestyle='--')
    plt.title('Scaling Device Population N (Non-IID, E=2)', fontweight='bold')
    plt.xlabel('Communication Rounds'); plt.ylabel('Accuracy (%)'); plt.grid(True, linestyle=':'); plt.legend()

    # ── PANEL 4: Statistical Divergence: IID vs Non-IID (N=60, E=2) ──
    print("Simulating Panel 4: IID vs Non-IID Trends...")
    plt.subplot(2, 2, 4)
    # Non-IID
    devs_non_iid, p_non_iid = partition_data(x_train, y_train, N=60, iid=False)
    acc_std_niid = run_federated_experiment(devs_non_iid, p_non_iid, n_rounds, E=2, eta0=eta0, x_test=x_test, y_test=y_test)
    acc_ema_niid = run_federated_experiment(devs_non_iid, p_non_iid, n_rounds, E=2, eta0=eta0, alpha=0.3, x_test=x_test, y_test=y_test)
    # IID
    devs_iid, p_iid = partition_data(x_train, y_train, N=60, iid=True)
    acc_std_iid = run_federated_experiment(devs_iid, p_iid, n_rounds, E=2, eta0=eta0, x_test=x_test, y_test=y_test)
    acc_ema_iid = run_federated_experiment(devs_iid, p_iid, n_rounds, E=2, eta0=eta0, alpha=0.3, x_test=x_test, y_test=y_test)

    plt.plot(range(1, n_rounds+1), acc_std_niid, label='Non-IID: Standard', color='#d62728', linewidth=2)
    plt.plot(range(1, n_rounds+1), acc_ema_niid, label='Non-IID: EMA (α=0.3)', color='#d62728', linestyle='--')
    plt.plot(range(1, n_rounds+1), acc_std_iid, label='IID Split: Standard', color='#2ca02c', linewidth=2)
    plt.plot(range(1, n_rounds+1), acc_ema_iid, label='IID Split: EMA (α=0.3)', color='#2ca02c', linestyle='--')
    plt.title('Statistical Divergence: IID vs Non-IID (N=60, E=2)', fontweight='bold')
    plt.xlabel('Communication Rounds'); plt.ylabel('Accuracy (%)'); plt.grid(True, linestyle=':'); plt.legend()

    plt.suptitle('Empirical Analysis: EMA-FedAvg vs Standard FedAvg Performance Profiles', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('comprehensive_fl_benchmark.png', dpi=300)
    plt.show()
