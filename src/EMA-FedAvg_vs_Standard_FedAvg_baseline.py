"""
EMA-FedAvg vs Standard FedAvg on MNIST
=======================================
Based on: "On the Convergence of FedAvg on Non-IID Data" (Li et al., ICLR 2020)

Dataset     : MNIST, N=100 devices, each device holds only 2 digits (non-IID)
Model       : Multinomial logistic regression (softmax + cross-entropy + L2)
LR schedule : eta_t = eta0 / (1 + t)  [decaying, as required by the paper]
Outputs     : Round-by-round console logs and immediate comparison graphs.
"""

import numpy as np
import os
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1. Load MNIST (Modified for Google Colab)
# ──────────────────────────────────────────────

def load_mnist():
    print("Fetching MNIST dataset via Keras...")
    from tensorflow.keras.datasets import mnist

    # Automatically downloads dataset into the Colab instance
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten the 28x28 images into 784 vectors and normalize pixel values to [0, 1]
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

    # group indices by digit
    class_indices = [np.where(y == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    # assign 2 digits to each device by cycling through all digit combinations
    digit_pairs = []
    all_digits = list(range(num_classes))
    for i in range(N):
        d1 = all_digits[i % num_classes]
        d2 = all_digits[(i + 1) % num_classes]
        digit_pairs.append((d1, d2))

    # pointers into each class's index list
    class_ptr = [0] * num_classes

    if balanced:
        samples_per_digit = min(len(class_indices[c]) for c in range(num_classes)) // (N // num_classes + 1)
        samples_per_digit = max(10, samples_per_digit)
    else:
        weights = np.array([(k+1)**(-0.5) for k in range(N)])
        weights /= weights.sum()

    devices = []
    for k in range(N):
        d1, d2 = digit_pairs[k]
        if balanced:
            n1 = n2 = samples_per_digit
        else:
            total_k = max(10, int(weights[k] * len(x)))
            n1 = total_k // 2
            n2 = total_k - n1

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
# 3. Multinomial logistic regression (Classic ML)
# ──────────────────────────────────────────────

D_IN    = 784
N_CLASS = 10
D_PARAM = D_IN * N_CLASS + N_CLASS   # 7850 parameters

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
    loss   = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
    loss  += (lam / 2) * np.dot(w[:D_IN*N_CLASS], w[:D_IN*N_CLASS])
    return loss

def cross_entropy_grad(w, X_batch, y_batch, lam=1e-4):
    W, b   = unpack(w)
    n      = len(y_batch)
    logits = X_batch @ W + b
    probs  = softmax(logits)
    probs[np.arange(n), y_batch] -= 1
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

def local_sgd(w_t, X, y, E, eta_t, batch_size=64, lam=1e-4, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    w = w_t.copy()
    n = len(y)
    for _ in range(E):
        idx   = rng.choice(n, size=min(batch_size, n), replace=False)
        grad  = cross_entropy_grad(w, X[idx], y[idx], lam)
        w     = w - eta_t * grad
    return w


# ──────────────────────────────────────────────
# 5. Original FedAvg
# ──────────────────────────────────────────────

def fedavg(devices, p, T, E, eta0, lam=1e-4, batch_size=64,
           x_test=None, y_test=None, seed=0):
    rng      = np.random.default_rng(seed)
    N        = len(devices)
    w        = np.zeros(D_PARAM)
    n_rounds = T // E

    loss_curve = []
    acc_curve  = []

    for r in range(n_rounds):
        t     = r * E
        eta_t = eta0 / (1 + t)

        v = []
        for k in range(N):
            X_k, y_k = devices[k]
            v_k = local_sgd(w, X_k, y_k, E, eta_t, batch_size, lam, rng)
            v.append(v_k)

        w = sum(p[k] * v[k] for k in range(N))

        loss = cross_entropy_loss(w, x_test, y_test, lam)
        acc  = accuracy(w, x_test, y_test)
        loss_curve.append(loss)
        acc_curve.append(acc)

        if (r + 1) % 10 == 0:
            print(f"  [FedAvg]     Round {r+1:3d}/{n_rounds} | Loss: {loss:.4f} | Acc: {acc*100:.2f}%")

    return w, loss_curve, acc_curve


# ──────────────────────────────────────────────
# 6. EMA-FedAvg
# ──────────────────────────────────────────────

def ema_fedavg(devices, p, T, E, eta0, alpha, lam=1e-4, batch_size=64,
               x_test=None, y_test=None, seed=0):
    rng      = np.random.default_rng(seed)
    N        = len(devices)
    w        = np.zeros(D_PARAM)
    n_rounds = T // E

    loss_curve = []
    acc_curve  = []

    for r in range(n_rounds):
        t     = r * E
        eta_t = eta0 / (1 + t)

        v = []
        for k in range(N):
            X_k, y_k = devices[k]
            v_k = local_sgd(w, X_k, y_k, E, eta_t, batch_size, lam, rng)
            v.append(v_k)

        current_agg = sum(p[k] * v[k] for k in range(N))
        w = (1 - alpha) * current_agg + alpha * w

        loss = cross_entropy_loss(w, x_test, y_test, lam)
        acc  = accuracy(w, x_test, y_test)
        loss_curve.append(loss)
        acc_curve.append(acc)

        if (r + 1) % 10 == 0:
            print(f"  [EMA-FedAvg] Round {r+1:3d}/{n_rounds} | Loss: {loss:.4f} | Acc: {acc*100:.2f}%")

    return w, loss_curve, acc_curve


# ──────────────────────────────────────────────
# 7. Main Execution & Plotting
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Hyperparameters
    N          = 100       # number of devices
    T          = 1000      # total local SGD steps
    E          = 10        # local steps per round → 100 communication rounds
    eta0       = 0.1       # initial learning rate
    lam        = 1e-4      # L2 regularization
    batch_size = 64        # mini-batch size
    alpha      = 0.3       # EMA decay factor

    print("="*60)
    print("  EMA-FedAvg vs Standard FedAvg on MNIST (Non-IID)")
    print("="*60)
    print(f"  N={N} devices | T={T} | E={E} | eta0={eta0} | alpha={alpha}")
    print(f"  Communication rounds: {T//E}\n")

    # Load data
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"  Train samples: {x_train.shape[0]} | Test samples: {x_test.shape[0]}\n")

    # Non-IID partitioning
    print("Partitioning data (non-IID: 2 digits per device)...")
    devices, p = partition_non_iid(x_train, y_train, N=N, balanced=True)
    sizes = [len(d[1]) for d in devices]
    print(f"  Samples/device: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}\n")

    # Run standard FedAvg
    print("Running Standard FedAvg...")
    _, loss_fed, acc_fed = fedavg(
        devices, p, T, E, eta0, lam, batch_size, x_test, y_test, seed=0)

    # Run EMA-FedAvg
    print("\nRunning EMA-FedAvg...")
    _, loss_ema, acc_ema = ema_fedavg(
        devices, p, T, E, eta0, alpha, lam, batch_size, x_test, y_test, seed=0)

    # Print Final Numerical Results
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"  Standard FedAvg | Final Loss: {loss_fed[-1]:.4f} | Final Acc: {acc_fed[-1]*100:.2f}%")
    print(f"  EMA-FedAvg      | Final Loss: {loss_ema[-1]:.4f} | Final Acc: {acc_ema[-1]*100:.2f}%")
    print("="*60 + "\n")

    # Generate performance graph lines mapped against round numbers
    print("Generating performance graphs...")
    rounds = np.arange(1, len(loss_fed) + 1)

    plt.figure(figsize=(14, 5))

    # Subplot A: Loss Curves
    plt.subplot(1, 2, 1)
    plt.plot(rounds, loss_fed, label='Standard FedAvg', color='#1f77b4', linewidth=2)
    plt.plot(rounds, loss_ema, label=f'EMA-FedAvg (alpha={alpha})', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.title('Global Test Loss vs. Rounds', fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Communication Rounds', fontsize=11)
    plt.ylabel('Cross-Entropy Loss', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='upper right')

    # Subplot B: Accuracy Curves
    plt.subplot(1, 2, 2)
    plt.plot(rounds, np.array(acc_fed) * 100, label='Standard FedAvg', color='#1f77b4', linewidth=2)
    plt.plot(rounds, np.array(acc_ema) * 100, label=f'EMA-FedAvg (alpha={alpha})', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.title('Global Test Accuracy vs. Rounds', fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Communication Rounds', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='lower right')

    plt.tight_layout()
    plt.savefig('fedavg_vs_ema_comparison.png', dpi=300)
    plt.show()
