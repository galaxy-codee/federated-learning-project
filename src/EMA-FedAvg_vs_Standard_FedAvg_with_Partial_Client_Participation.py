"""
STEP 1: EMA-FedAvg vs Standard FedAvg with Partial Client Participation
========================================================================
Implements a realistic federated edge environment where only a fraction (C)
of the total device population checks in and trains per communication round.

Dataset     : MNIST, N=100 devices, non-IID (2 digits per device)
Model       : Multinomial logistic regression
New Feature : Dynamic client sampling fraction C=0.2 (20% participation/round)
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
# 4. Local SGD
# ──────────────────────────────────────────────

def local_sgd(w_t, X, y, E, eta_round, batch_size=64, lam=1e-4, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    w = w_t.copy()
    n = len(y)
    for _ in range(E):
        idx   = rng.choice(n, size=min(batch_size, n), replace=False)
        grad  = cross_entropy_grad(w, X[idx], y[idx], lam)
        w     = w - eta_round * grad
    return w


# ──────────────────────────────────────────────
# 5. Standard FedAvg with Dynamic Sampling
# ──────────────────────────────────────────────

def fedavg(devices, p, n_rounds, E, eta0, C=0.2, lam=1e-4, batch_size=64,
           x_test=None, y_test=None, seed=0):
    rng   = np.random.default_rng(seed)
    N     = len(devices)
    w     = np.zeros(D_PARAM)

    loss_curve = []
    acc_curve  = []

    # Determine the number of active clients checking in per round
    m = max(1, int(C * N))

    for r in range(n_rounds):
        eta_round = eta0 if r < (n_rounds // 2) else eta0 * 0.5

        # 1. SAMPLE: Randomly choose a subset of unique client indices
        active_indices = rng.choice(N, size=m, replace=False)

        # 2. RE-WEIGHT: Extract and re-normalize weights to ensure they sum to 1.0
        active_p = p[active_indices]
        active_p_normalized = active_p / active_p.sum()

        # 3. COMPUTE: Train locally only on chosen devices
        v = []
        for idx in active_indices:
            X_k, y_k = devices[idx]
            v_k = local_sgd(w, X_k, y_k, E, eta_round, batch_size, lam, rng)
            v.append(v_k)

        # 4. AGGREGATE: Recombine parameters using normalized local contributions
        w = sum(active_p_normalized[i] * v[i] for i in range(m))

        loss = cross_entropy_loss(w, x_test, y_test, lam)
        acc  = accuracy(w, x_test, y_test)
        loss_curve.append(loss)
        acc_curve.append(acc)

        if (r + 1) % 10 == 0:
            print(f"  [FedAvg]      Round {r+1:3d}/{n_rounds} | Active Devices: {m}/{N} | Loss: {loss:.4f} | Acc: {acc*100:.2f}%")

    return w, loss_curve, acc_curve


# ──────────────────────────────────────────────
# 6. EMA-FedAvg with Dynamic Sampling
# ──────────────────────────────────────────────

def ema_fedavg(devices, p, n_rounds, E, eta0, alpha, C=0.2, lam=1e-4, batch_size=64,
               x_test=None, y_test=None, seed=0):
    rng   = np.random.default_rng(seed)
    N     = len(devices)
    w     = np.zeros(D_PARAM)

    loss_curve = []
    acc_curve  = []

    m = max(1, int(C * N))

    for r in range(n_rounds):
        eta_round = eta0 if r < (n_rounds // 2) else eta0 * 0.5

        # 1. SAMPLE: Select active entities
        active_indices = rng.choice(N, size=m, replace=False)

        # 2. RE-WEIGHT: Set subset weights cleanly
        active_p = p[active_indices]
        active_p_normalized = active_p / active_p.sum()

        # 3. COMPUTE: Edge device execution
        v = []
        for idx in active_indices:
            X_k, y_k = devices[idx]
            v_k = local_sgd(w, X_k, y_k, E, eta_round, batch_size, lam, rng)
            v.append(v_k)

        current_agg = sum(active_p_normalized[i] * v[i] for i in range(m))

        # 4. EMA UPDATE TRACK
        if r == 0:
            w = current_agg
        else:
            w = alpha * current_agg + (1 - alpha) * w

        loss = cross_entropy_loss(w, x_test, y_test, lam)
        acc  = accuracy(w, x_test, y_test)
        loss_curve.append(loss)
        acc_curve.append(acc)

        if (r + 1) % 10 == 0:
            print(f"  [EMA-FedAvg] Round {r+1:3d}/{n_rounds} | Active Devices: {m}/{N} | Loss: {loss:.4f} | Acc: {acc*100:.2f}%")

    return w, loss_curve, acc_curve


# ──────────────────────────────────────────────
# 7. Main Execution & Plotting
# ──────────────────────────────────────────────

if __name__ == "__main__":
    N          = 100       # Total fleet size
    n_rounds   = 100
    E          = 2
    eta0       = 0.05
    lam        = 1e-4
    batch_size = 64
    alpha      = 0.3
    C          = 0.2       # New Feature: Only 20% of clients participate each round

    print("="*60)
    print("  STEP 1 COMPLETE: Federated Learning with Client Sampling")
    print("="*60)
    print(f"  Total Fleet Size (N)   : {N}")
    print(f"  Sampling Fraction (C)  : {C} ({int(C*N)} clients/round)")
    print(f"  Communication Rounds   : {n_rounds}")
    print(f"  Local Steps per Round  : {E}")

    x_train, y_train, x_test, y_test = load_mnist()
    devices, p = partition_non_iid(x_train, y_train, N=N, balanced=True)

    print("\nRunning Standard FedAvg with Client Sampling...")
    _, loss_fed, acc_fed = fedavg(devices, p, n_rounds, E, eta0, C, lam, batch_size, x_test, y_test, seed=0)

    print("\nRunning EMA-FedAvg with Client Sampling...")
    _, loss_ema, acc_ema = ema_fedavg(devices, p, n_rounds, E, eta0, alpha, C, lam, batch_size, x_test, y_test, seed=0)

    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"  Standard FedAvg | Final Loss: {loss_fed[-1]:.4f} | Final Acc: {acc_fed[-1]*100:.2f}%")
    print(f"  EMA-FedAvg      | Final Loss: {loss_ema[-1]:.4f} | Final Acc: {acc_ema[-1]*100:.2f}%")
    print("="*60 + "\n")

    # Visualizing the Output
    rounds = np.arange(1, n_rounds + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rounds, loss_fed, label=f'Standard FedAvg (C={C})', color='#1f77b4', linewidth=2)
    plt.plot(rounds, loss_ema, label=f'EMA-FedAvg (C={C}, α={alpha})', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.title('Global Test Loss (Partial Participation)', fontsize=12, fontweight='bold')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Loss')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rounds, np.array(acc_fed) * 100, label=f'Standard FedAvg (C={C})', color='#1f77b4', linewidth=2)
    plt.plot(rounds, np.array(acc_ema) * 100, label=f'EMA-FedAvg (C={C}, α={alpha})', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.title('Global Test Accuracy (Partial Participation)', fontsize=12, fontweight='bold')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig('partial_client_participation.png', dpi=300)
    plt.show()
