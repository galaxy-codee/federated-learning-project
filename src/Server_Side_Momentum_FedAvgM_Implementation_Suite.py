"""
STEP 3: Server-Side Momentum (FedAvgM) Implementation Suite
============================================================
Deploys FedAvgM alongside FedProx, EMA, and Standard FedAvg.
Uses E=15 to induce heavy client drift, forcing the optimization
paths to visually separate and demonstrate their unique characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1. Load MNIST
# ──────────────────────────────────────────────

def load_mnist():
    print("Fetching MNIST dataset via Keras...")
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    return x_train, y_train.astype(np.int32), x_test.reshape(-1, 784).astype(np.float32) / 255.0, y_test.astype(np.int32)


# ──────────────────────────────────────────────
# 2. Non-IID partitioning
# ──────────────────────────────────────────────

def partition_non_iid(x, y, N=100, seed=42):
    rng = np.random.default_rng(seed)
    num_classes = 10
    class_indices = [np.where(y == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(class_indices[c])

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
# 3. Model Engine Setup
# ──────────────────────────────────────────────

D_IN, N_CLASS = 784, 10
D_PARAM = D_IN * N_CLASS + N_CLASS

def unpack(w):
    return w[:7840].reshape(784, 10), w[7840:]

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(w, X, y, lam=1e-4):
    W, b = unpack(w)
    probs = softmax(X @ W + b)
    n = len(y)
    return -np.mean(np.log(probs[np.arange(n), y] + 1e-12)) + (lam / 2) * np.dot(w[:7840], w[:7840])

def cross_entropy_grad(w, X_batch, y_batch, lam=1e-4):
    W, b = unpack(w)
    n = len(y_batch)
    probs = softmax(X_batch @ W + b)
    probs[np.arange(n), y_batch] -= 1
    probs /= n
    return np.concatenate([(X_batch.T @ probs + lam * W).ravel(), probs.sum(axis=0)])

def accuracy(w, X, y):
    W, b = unpack(w)
    return np.mean((X @ W + b).argmax(axis=1) == y)


# ──────────────────────────────────────────────
# 4. Generalized Local SGD Engine
# ──────────────────────────────────────────────

def local_sgd(w_t, X, y, E, eta_round, mu=0.0, batch_size=64, lam=1e-4, rng=None):
    w = w_t.copy()
    n = len(y)
    for _ in range(E):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        grad = cross_entropy_grad(w, X[idx], y[idx], lam)

        if mu > 0.0:
            grad += mu * (w - w_t)

        w -= eta_round * grad
    return w


# ──────────────────────────────────────────────
# 5. Centralized Framework Runner (Supporting FedAvgM)
# ──────────────────────────────────────────────

def run_experiment(mode, devices, p, n_rounds, E, eta0, C=0.2, alpha=None, mu=0.0, beta=0.9,
                   x_test=None, y_test=None, seed=42):
    rng = np.random.default_rng(seed)
    N = len(devices)
    w = np.zeros(D_PARAM)

    loss_curve, acc_curve = [], []
    m = max(1, int(C * N))

    # Initialize Server Momentum Velocity Vector
    v_velocity = np.zeros(D_PARAM)

    for r in range(n_rounds):
        eta_round = eta0 if r < (n_rounds // 2) else eta0 * 0.5
        active_indices = rng.choice(N, size=m, replace=False)
        active_p_norm = p[active_indices] / p[active_indices].sum()

        v = []
        for idx in active_indices:
            X_k, y_k = devices[idx]
            v_k = local_sgd(w, X_k, y_k, E, eta_round, mu, batch_size=64, lam=1e-4, rng=rng)
            v.append(v_k)

        current_agg = sum(active_p_norm[i] * v[i] for i in range(m))

        # Core Server Aggregation Logic States
        if mode == 'ema':
            w = current_agg if r == 0 else alpha * current_agg + (1 - alpha) * w
        elif mode == 'fedavgm':
            # NEW ADVANCED INTEGRATION: Global Update Vector Vectorization
            pseudo_grad = w - current_agg
            v_velocity  = beta * v_velocity + pseudo_grad
            w           = w - v_velocity
        else:
            w = current_agg

        loss = cross_entropy_loss(w, x_test, y_test)
        acc  = accuracy(w, x_test, y_test)
        loss_curve.append(loss)
        acc_curve.append(acc)

        if (r + 1) % 20 == 0:
            print(f"  [{mode.upper()}] Round {r+1:3d}/{n_rounds} | Loss: {loss:.4f} | Acc: {acc*100:.2f}%")

    return loss_curve, acc_curve


# ──────────────────────────────────────────────
# 6. Pipeline Execution
# ──────────────────────────────────────────────

if __name__ == "__main__":
    N, n_rounds, eta0, C = 100, 100, 0.05, 0.2
    E = 15     # High local steps to break lines apart cleanly

    print("="*60)
    print(f"  STEP 3 DEPLOYED: High Drift Simulation Engine (E={E})")
    print("="*60)

    x_train, y_train, x_test, y_test = load_mnist()
    devices, p = partition_non_iid(x_train, y_train, N=N)

    print("\nRunning Standard FedAvg Baseline...")
    loss_fed, acc_fed = run_experiment('fedavg', devices, p, n_rounds, E, eta0, C, x_test=x_test, y_test=y_test)

    print("\nRunning EMA-FedAvg Tweak (alpha=0.3)...")
    loss_ema, acc_ema = run_experiment('ema', devices, p, n_rounds, E, eta0, C, alpha=0.3, x_test=x_test, y_test=y_test)

    print("\nRunning FedProx Regularization Deployment (mu=1.0)...")
    loss_prox, acc_prox = run_experiment('fedprox', devices, p, n_rounds, E, eta0, C, mu=1.0, x_test=x_test, y_test=y_test)

    print("\nRunning Server-Side Momentum Execution (FedAvgM, beta=0.9)...")
    loss_fedm, acc_fedm = run_experiment('fedavgm', devices, p, n_rounds, E, eta0, C, beta=0.9, x_test=x_test, y_test=y_test)

    # Multi-Variant Output Presentation
    rounds = np.arange(1, n_rounds + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rounds, loss_fed, label='Standard FedAvg', color='#1f77b4', linewidth=2)
    plt.plot(rounds, loss_ema, label='EMA-FedAvg (α=0.3)', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.plot(rounds, loss_prox, label='FedProx (μ=1.0)', color='#2ca02c', linewidth=2)
    plt.plot(rounds, loss_fedm, label='FedAvgM (β=0.9)', color='#d62728', linewidth=2, linestyle=':')
    plt.title('Global Test Loss (High Drift Workspace)', fontsize=12, fontweight='bold')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Loss')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rounds, np.array(acc_fed) * 100, label='Standard FedAvg', color='#1f77b4', linewidth=2)
    plt.plot(rounds, np.array(acc_ema) * 100, label='EMA-FedAvg (α=0.3)', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.plot(rounds, np.array(acc_prox) * 100, label='FedProx (μ=1.0)', color='#2ca02c', linewidth=2)
    plt.plot(rounds, np.array(acc_fedm) * 100, label='FedAvgM (β=0.9)', color='#d62728', linewidth=2, linestyle=':')
    plt.title('Global Test Accuracy (High Drift Workspace)', fontsize=12, fontweight='bold')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig('high_drift_comparison_using_FedAvgM.png', dpi=300)
    plt.show()
