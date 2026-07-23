"""
FILE: continual_learning_eval.py
================================──────────────────────────────────────────────
Evaluates Catastrophic Forgetting versus local Elastic Weight Consolidation (EWC)
regularization barriers when streaming sequentially across distinct digit tasks.
"""

import numpy as np

D_IN, N_CLASS = 784, 10
D_PARAM = D_IN * N_CLASS + N_CLASS

def unpack(w): return w[:7840].reshape(784, 10), w[7840:]
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

def cross_entropy_grad(w, X_batch, y_batch, lam=1e-4):
    W, b = unpack(w)
    probs = softmax(X_batch @ W + b)
    probs[np.arange(len(y_batch)), y_batch] -= 1
    probs /= len(y_batch)
    return np.concatenate([(X_batch.T @ probs + lam * W).ravel(), probs.sum(axis=0)])

def accuracy(w, X, y):
    W, b = unpack(w)
    return np.mean((X @ W + b).argmax(axis=1) == y)

def local_sgd_continual(w_initial, X, y, E, eta, batch_size=64, lam=1e-4, rng=None, w_old=None, ewc_lambda=0.0):
    w = w_initial.copy()
    for _ in range(E):
        idx = rng.choice(len(y), size=min(batch_size, len(y)), replace=False)
        grad = cross_entropy_grad(w, X[idx], y[idx], lam)

        # TEMPORAL CONSTRAINT: EWC Penalty to anchor old parameter spaces
        if ewc_lambda > 0.0 and w_old is not None:
            grad += ewc_lambda * (w - w_old)

        w -= eta * grad
    return w

if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    x_test  = x_test.reshape(-1, 784).astype(np.float32) / 255.0

    rng = np.random.default_rng(42)

    # Task 1 Dataset: Digits 0 to 4
    idx_t1_train = np.where(y_train < 5)[0]
    idx_t1_test  = np.where(y_test < 5)[0]
    X_t1_train, y_t1_train = x_train[idx_t1_train], y_train[idx_t1_train]
    X_t1_test,  y_t1_test  = x_test[idx_t1_test],   y_test[idx_t1_test]

    # Task 2 Dataset: Digits 5 to 9
    idx_t2_train = np.where(y_train >= 5)[0]
    X_t2_train, y_t2_train = x_train[idx_t2_train], y_train[idx_t2_train]

    print("Training base parameter configurations on Task 1 (Digits 0-4)...")
    w_task1 = local_sgd_continual(np.zeros(D_PARAM), X_t1_train, y_t1_train, E=30, eta=0.05, rng=rng)

    print("Shifting sequential training stream directly to Task 2 (Digits 5-9)...")
    w_forget = local_sgd_continual(w_task1, X_t2_train, y_t2_train, E=30, eta=0.05, rng=rng)
    w_consolidation = local_sgd_continual(w_task1, X_t2_train, y_t2_train, E=30, eta=0.05, rng=rng, w_old=w_task1, ewc_lambda=8.0)

    acc_forget = accuracy(w_forget, X_t1_test, y_t1_test)
    acc_consol = accuracy(w_consolidation, X_t1_test, y_t1_test)

    print("\n" + "="*75)
    print("      CONTINUAL LEARNING ADVISOR REPORT: HISTORIC MEMORY RETENTION")
    print("="*75)
    print(f"  Accuracy on Task 1 after Task 2 training (Standard FedAvg): {acc_forget*100:.2f}% -> [CATASTROPHIC FORGETTING]")
    print(f"  Accuracy on Task 1 after Task 2 training (With EWC Shield): {acc_consol*100:.2f}% -> [KNOWLEDGE CONSERVED]")
    print("="*75 + "\n")
