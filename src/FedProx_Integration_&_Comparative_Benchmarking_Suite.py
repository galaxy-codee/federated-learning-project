"""
STEP 2: FedProx Integration & Comparative Benchmarking Suite
=============================================================
Deploys FedProx proximal regularization to the local training engine to counteract
client drift caused by partial client participation on non-IID data distributions.

Dataset     : MNIST, N=100 devices, non-IID (2 digits per device)
Model       : Multinomial logistic regression
New Feature : FedProx Optimizer Engine (mu=1.0) compared alongside baseline loops.
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
    y_train = y_train.astype(np.int32)

    x_test  = x_test.reshape(-1, 784).astype(np.float32) / 255.0
    y_test  = y_test.astype(np.int32)

    return x_train, y_train, x_test, y_test


# ──────────────────────────────────────────────
# 2. Non-IID partitioning
# ──────────────────────────────────────────────

def partition_non_iid(x, y, N=100, balanced=True, seed=42):
    rng = np.random.default_rng(seed)
    num_classes = 10

    class_indices = [np.where(y == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    digit_pairs = []
    all_digits = list(range(num_classes))
    for i in range(N):
        d1 = all_digits[i % num_classes]
        d2 = all_digits[(i + 1) % num_classes]
        digit_pairs.append((d1, d2))

    class_ptr = [0] * num_classes
    samples_per_digit = min(len(class_indices[c]) for c in range(num_classes)) // (N // num_classes + 1)
    samples_per_digit = max(10, samples_per_digit)

    devices = []
    for k in range(N):
        d1, d2 = digit_pairs[k]
        n1 = n2 = samples_per_digit

        n1 = min(n1, len(class_indices[d1]) - class_ptr[d1])
        n2 = min(n2, len(class_indices[d2]) - class_ptr[d2])
        n1, n2 = max(1, n1), max(1, n2)

        idx1 = class_indices[d1][class_ptr[d1]: class_ptr[d1] + n1]
        idx2 = class_indices[d2][class_ptr[d2]: class_ptr[d2] + n2]
        class_ptr[d1] += n1
        class_ptr[d2] += n2

        idx = np.concatenate([idx1, idx2])
        rng.shuffle(idx)
        devices.append((x[idx], y[idx]))

    sizes = np.array([len(d[1]) for d in devices], dtype=float)
    p = sizes / sizes.sum()
    return devices, p


# ──────────────────────────────────────────────
# 3. Multinomial logistic regression
# ──────────────────────────────────────────────

D_IN    = 784
N_CLASS = 10
D_PARAM = D_IN * N_CLASS + N_CLASS

def unpack(w):
    W = w[:D_IN * N_CLASS].reshape(D_IN, N_CLASS)
    b = w[D_IN * N_CLASS:]
    return W, b

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(w, X, y, lam=1e-4):
    W, b = unpack(w)
    logits = X @ W + b
    probs  = softmax(logits)
    n      = len(y)
    loss   = -np.mean(np.log(probs[np.arange(n), y.astype(int)] + 1e-12))
    loss  += (lam / 2) * np.dot(w[:D_IN*N_CLASS], w[:D_IN*N_CLASS])
    return loss

def cross_entropy_grad(w, X_batch, y_batch, lam=1e-4):
    W, b   = unpack(w)
    n      = len(y_batch)
    logits = X_batch @ W + b
    probs  = softmax(logits)
    probs[np.arange(n), y_batch.astype(int)] -= 1
    probs /= n

    dW = X_batch.T @ probs + lam * W
    db = probs.sum(axis=0)
    return np.concatenate([dW.ravel(), db])

def accuracy(w, X, y):
    W, b   = unpack(w)
    logits = X @ W + b
    preds  = logits.argmax(axis=1)
    return np.mean(preds == y)


# ──────────────────────────────────────────────
# 4. Generalized Local SGD Engine (Supporting FedProx)
# ──────────────────────────────────────────────

def local_sgd(w_t, X, y, E, eta_round, mu=0.0, batch_size=64, lam=1e-4, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    w = w_t.copy()
    n = len(y)
    for _ in range(E):
        idx   = rng.choice(n, size=min(batch_size, n), replace=False)
        grad  = cross_entropy_grad(w, X[idx], y[idx], lam)

        # FIXED EXTENSION: Add the FedProx proximal penalty term if mu is active
        if mu > 0.0:
            grad += mu * (w - w_t)

        w     = w - eta_round * grad
    return w


# ──────────────────────────────────────────────
# 5. Centralized Framework Experiment Runner
# ──────────────────────────────────────────────

def run_experiment(mode, devices, p, n_rounds, E, eta0, C=0.2, alpha=None, mu=0.0,
                   x_test=None, y_test=None, seed=0):
    rng = np.random.default_rng(seed)
    N = len(devices)
    w = np.zeros(D_PARAM)

    loss_curve = []
    acc_curve  = []
    m = max(1, int(C * N))

    for r in range(n_rounds):
        eta_round = eta0 if r < (n_rounds // 2) else eta0 * 0.5

        # Client selection matching step 1 constraints
        active_indices = rng.choice(N, size=m, replace=False)
        active_p_norm = p[active_indices] / p[active_indices].sum()

        v = []
        for idx in active_indices:
            X_k, y_k = devices[idx]
            # Pass the mu variable directly to the local optimizer
            v_k = local_sgd(w, X_k, y_k, E, eta_round, mu, batch_size=64, lam=1e-4, rng=rng)
            v.append(v_k)

        current_agg = sum(active_p_norm[i] * v[i] for i in range(m))

        if mode == 'ema':
            w = current_agg if r == 0 else alpha * current_agg + (1 - alpha) * w
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
    N          = 100
    n_rounds   = 100
    E          = 2
    eta0       = 0.05
    C          = 0.2       # 20% client selection
    alpha      = 0.3

    print("="*60)
    print("  STEP 2 COMPLETE: Deploying FedProx Multi-Variant Suite")
    print("="*60)

    x_train, y_train, x_test, y_test = load_mnist()
    devices, p = partition_non_iid(x_train, y_train, N=N, balanced=True)

    print("\nRunning Standard FedAvg Baseline (mu=0.0)...")
    loss_fed, acc_fed = run_experiment('fedavg', devices, p, n_rounds, E, eta0, C, mu=0.0, x_test=x_test, y_test=y_test)

    print("\nRunning EMA-FedAvg Execution (alpha=0.3)...")
    loss_ema, acc_ema = run_experiment('ema', devices, p, n_rounds, E, eta0, C, alpha=alpha, x_test=x_test, y_test=y_test)

    print("\nRunning FedProx Deployment (mu=1.0)...")
    loss_prox, acc_prox = run_experiment('fedprox', devices, p, n_rounds, E, eta0, C, mu=1.0, x_test=x_test, y_test=y_test)

    # Visualization Setup
    rounds = np.arange(1, n_rounds + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rounds, loss_fed, label='Standard FedAvg', color='#1f77b4', linewidth=2)
    plt.plot(rounds, loss_ema, label=f'EMA-FedAvg (α={alpha})', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.plot(rounds, loss_prox, label='FedProx (μ=1.0)', color='#2ca02c', linewidth=2)
    plt.title('Global Test Loss Comparison', fontsize=12, fontweight='bold')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Loss')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rounds, np.array(acc_fed) * 100, label='Standard FedAvg', color='#1f77b4', linewidth=2)
    plt.plot(rounds, np.array(acc_ema) * 100, label=f'EMA-FedAvg (α={alpha})', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.plot(rounds, np.array(acc_prox) * 100, label='FedProx (μ=1.0)', color='#2ca02c', linewidth=2)
    plt.title('Global Test Accuracy Comparison', fontsize=12, fontweight='bold')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig('fedavg_ema_fedprox_comparison.png', dpi=300)
    plt.show()
