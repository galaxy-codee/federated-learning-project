"""
EMA-FedAvg vs Standard FedAvg on MNIST
=======================================
Based on: "On the Convergence of FedAvg on Non-IID Data" (Li et al., ICLR 2020)

Dataset     : MNIST, N=100 devices, each device holds only 2 digits (non-IID)
Model       : Multinomial logistic regression (softmax + cross-entropy + L2)
LR schedule : eta_t = eta0 / (1 + t)  [decaying, as required by the paper]
Outputs     : Round-by-round console logs and immediate comparison graphs.
(Multiple plots by changing alpha, Epoch, No. of clients, IID vs Non-IID case)
"""
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1. Load MNIST (Via Keras)
# ──────────────────────────────────────────────

def load_mnist():
    print("Fetching MNIST dataset via Keras...")
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten and normalize
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    y_train = y_train.astype(np.int32)

    x_test  = x_test.reshape(-1, 784).astype(np.float32) / 255.0
    y_test  = y_test.astype(np.int32)

    return x_train, y_train, x_test, y_test


# ──────────────────────────────────────────────
# 2. Partitioning Data (Non-IID and IID)
# ──────────────────────────────────────────────

def partition_non_iid(x, y, N=100, seed=42):
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
    p = sizes / sizes.sum()
    return devices, p


def partition_iid(x, y, N=100, seed=42):
    """Uniformly distributed random splits across devices."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)

    chunks = np.array_split(indices, N)
    devices = []
    for chunk in chunks:
        devices.append((x[chunk], y[chunk]))

    sizes = np.array([len(d[1]) for d in devices], dtype=float)
    p = sizes / sizes.sum()
    return devices, p


# ──────────────────────────────────────────────
# 3. Multinomial Logistic Regression Utilities
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
# 4. Parametric Federated Engine
# ──────────────────────────────────────────────

def run_federated_experiment(devices, p, T, E, eta0, alpha, lam=1e-4,
                             batch_size=64, x_test=None, y_test=None, seed=0):
    """
    Unified engine running either standard FedAvg (alpha=0) or EMA-FedAvg (alpha > 0).
    Tracks accuracy curves mapped against uniform structural intervals.
    """
    rng = np.random.default_rng(seed)
    N = len(devices)
    w = np.zeros(D_PARAM)
    n_rounds = T // E

    acc_curve = []

    for r in range(n_rounds):
        t = r * E
        eta_t = eta0 / (1 + t)

        v = []
        for k in range(N):
            X_k, y_k = devices[k]
            v_k = local_sgd(w, X_k, y_k, E, eta_t, batch_size, lam, rng)
            v.append(v_k)

        current_agg = sum(p[k] * v[k] for k in range(N))

        # EMA weight modification rule
        w = (1 - alpha) * current_agg + alpha * w

        acc = accuracy(w, x_test, y_test)
        acc_curve.append(acc)

    return acc_curve


# ──────────────────────────────────────────────
# 5. Experiment Suite Execution & Plotting
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Baseline Hyperparameters
    T_total    = 600      # Shortened slightly to process many variations gracefully
    eta0       = 0.1
    lam        = 1e-4
    batch_size = 64

    x_train, y_train, x_test, y_test = load_mnist()

    # Initialize plotting layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Empirical Analysis: EMA-FedAvg vs Standard FedAvg Performance Profiles',
                 fontsize=16, fontweight='bold', y=0.98)

    # ---------------------------------------------------------
    # Test Suite 1: Effects of changing Hyperparameter Alpha
    # ---------------------------------------------------------
    print("\n>>> Running Experiment 1: Scanning Alpha Values...")
    ax1 = axes[0, 0]
    N_exp1, E_exp1 = 50, 10
    dev_non_iid, p_non_iid = partition_non_iid(x_train, y_train, N=N_exp1)
    rounds_exp1 = np.arange(1, (T_total // E_exp1) + 1)

    for alpha_val in [0.0, 0.2, 0.5, 0.7]:
        label_text = 'Standard FedAvg' if alpha_val == 0.0 else f'EMA-FedAvg (α={alpha_val})'
        style = '-' if alpha_val == 0.0 else '--'
        curves = run_federated_experiment(dev_non_iid, p_non_iid, T_total, E_exp1, eta0, alpha_val,
                                           lam, batch_size, x_test, y_test)
        ax1.plot(rounds_exp1, np.array(curves) * 100, label=label_text, linestyle=style, lw=2)

    ax1.set_title(f'Impact of Alpha (Non-IID, N={N_exp1}, E={E_exp1})', fontweight='bold')
    ax1.set_xlabel('Communication Rounds')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # ---------------------------------------------------------
    # Test Suite 2: Effects of changing Local Step Intervals (E)
    # ---------------------------------------------------------
    print("\n>>> Running Experiment 2: Scanning Local Steps (E)...")
    ax2 = axes[0, 1]
    N_exp2, alpha_exp2 = 50, 0.3
    dev_non_iid, p_non_iid = partition_non_iid(x_train, y_train, N=N_exp2)

    for E_val in [5, 15, 30]:
        rounds_exp2 = np.arange(1, (T_total // E_val) + 1)
        # Standard
        c_std = run_federated_experiment(dev_non_iid, p_non_iid, T_total, E_val, eta0, 0.0,
                                         lam, batch_size, x_test, y_test)
        ax2.plot(rounds_exp2, np.array(c_std) * 100, label=f'Standard (E={E_val})', lw=1.5)
        # EMA
        c_ema = run_federated_experiment(dev_non_iid, p_non_iid, T_total, E_val, eta0, alpha_exp2,
                                         lam, batch_size, x_test, y_test)
        ax2.plot(rounds_exp2, np.array(c_ema) * 100, label=f'EMA (E={E_val}, α={alpha_exp2})', linestyle='--', lw=2)

    ax2.set_title(f'Impact of Local Steps E (Non-IID, N={N_exp2})', fontweight='bold')
    ax2.set_xlabel('Communication Rounds')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    # ---------------------------------------------------------
    # Test Suite 3: Scaling Population Sizes (N)
    # ---------------------------------------------------------
    print("\n>>> Running Experiment 3: Scaling Network Population Size (N)...")
    ax3 = axes[1, 0]
    E_exp3, alpha_exp3 = 10, 0.3
    rounds_exp3 = np.arange(1, (T_total // E_exp3) + 1)

    for N_val in [20, 60, 100]:
        dev_non_iid, p_non_iid = partition_non_iid(x_train, y_train, N=N_val)
        # Standard
        c_std = run_federated_experiment(dev_non_iid, p_non_iid, T_total, E_exp3, eta0, 0.0,
                                         lam, batch_size, x_test, y_test)
        ax3.plot(rounds_exp3, np.array(c_std) * 100, label=f'Standard (N={N_val})', lw=1.5)
        # EMA
        c_ema = run_federated_experiment(dev_non_iid, p_non_iid, T_total, E_exp3, eta0, alpha_exp3,
                                         lam, batch_size, x_test, y_test)
        ax3.plot(rounds_exp3, np.array(c_ema) * 100, label=f'EMA (N={N_val}, α={alpha_exp3})', linestyle='--', lw=2)

    ax3.set_title(f'Scaling Device Population N (Non-IID, E={E_exp3})', fontweight='bold')
    ax3.set_xlabel('Communication Rounds')
    ax3.set_ylabel('Accuracy (%)')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend()

    # ---------------------------------------------------------
    # Test Suite 4: Non-IID vs IID Statistical Data Split Matchup
    # ---------------------------------------------------------
    print("\n>>> Running Experiment 4: Non-IID vs IID Structural Matchup...")
    ax4 = axes[1, 1]
    N_exp4, E_exp4, alpha_exp4 = 60, 10, 0.3
    rounds_exp4 = np.arange(1, (T_total // E_exp4) + 1)

    # Prepare data distributions
    dev_non_iid, p_non_iid = partition_non_iid(x_train, y_train, N=N_exp4)
    dev_iid, p_iid = partition_iid(x_train, y_train, N=N_exp4)

    # 1. Non-IID: Standard vs EMA
    c_non_iid_std = run_federated_experiment(dev_non_iid, p_non_iid, T_total, E_exp4, eta0, 0.0, lam, batch_size, x_test, y_test)
    c_non_iid_ema = run_federated_experiment(dev_non_iid, p_non_iid, T_total, E_exp4, eta0, alpha_exp4, lam, batch_size, x_test, y_test)

    # 2. IID: Standard vs EMA
    c_iid_std = run_federated_experiment(dev_iid, p_iid, T_total, E_exp4, eta0, 0.0, lam, batch_size, x_test, y_test)
    c_iid_ema = run_federated_experiment(dev_iid, p_iid, T_total, E_exp4, eta0, alpha_exp4, lam, batch_size, x_test, y_test)

    ax4.plot(rounds_exp4, np.array(c_non_iid_std) * 100, label='Non-IID: Standard', color='red', lw=1.5)
    ax4.plot(rounds_exp4, np.array(c_non_iid_ema) * 100, label=f'Non-IID: EMA (α={alpha_exp4})', color='red', linestyle='--', lw=2)
    ax4.plot(rounds_exp4, np.array(c_iid_std) * 100, label='IID Split: Standard', color='green', lw=1.5)
    ax4.plot(rounds_exp4, np.array(c_iid_ema) * 100, label=f'IID Split: EMA (α={alpha_exp4})', color='green', linestyle='--', lw=2)

    ax4.set_title(f'Statistical Divergence: IID vs Non-IID (N={N_exp4}, E={E_exp4})', fontweight='bold')
    ax4.set_xlabel('Communication Rounds')
    ax4.set_ylabel('Accuracy (%)')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend()

    # Adjust layout and render
    plt.tight_layout()
    plt.savefig('federated_comprehensive_evaluation.png', dpi=300)
    print("\n[Done] Comparative figures generated successfully as 'federated_comprehensive_evaluation.png'.")
    plt.show()
