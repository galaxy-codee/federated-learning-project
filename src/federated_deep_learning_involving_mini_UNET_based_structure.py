"""
FILE: federated_deep_learning.py
================================────────────────────────────────==============
Implements a True Federated Deep Learning Framework using a Lightweight U-Net
Encoder-Decoder Architecture across Non-IID Edge Clients with Server Momentum.
================================────────────────────────────────==============
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set deterministic seeds for validation stability
torch.manual_seed(101)
np.random.seed(101)

# ──────────────────────────────────────────────────────────────────────────────
# 1. THE DEEP NETWORK ARCHITECTURE (Lightweight Convolutional U-Net Variant)
# ──────────────────────────────────────────────────────────────────────────────

class FederatedUNetClassifier(nn.Module):
    def __init__(self):
        super(FederatedUNetClassifier, self).__init__()
        # Contracting Path (Encoder)
        self.enc1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)  # Input: 1x28x28 -> 4x28x28
        self.pool1 = nn.MaxPool2d(2, 2)                        # 4x28x28 -> 4x14x14

        # Bottleneck
        self.bottleneck = nn.Conv2d(4, 8, kernel_size=3, padding=1) # 8x14x14

        # Expansive Path (Decoder) with Skip Connections
        self.up1 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2) # 4x28x28

        # Linear classification head mapping deep feature representations to 10 classes
        # Note: We output RAW LOGITS. No Softmax layer at the end!
        self.flat_dim = 4 * 28 * 28
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # Explicitly reshape flat input vectors to spatial 2D images
        x = x.view(-1, 1, 28, 28)

        # Forward pass through structural convolutions
        s1 = self.relu(self.enc1(x))
        p1 = self.pool1(s1)
        b  = self.relu(self.bottleneck(p1))

        # Decode and pass the skip connection tensor
        d1 = self.relu(self.up1(b))
        d1 = d1 + s1  # Residual structural skip connection alignment

        # Flatten and classify to output raw scores (logits)
        flat = d1.view(-1, self.flat_dim)
        return self.classifier(flat)

# ──────────────────────────────────────────────────────────────────────────────
# 2. EDGE SIMULATION: NON-IID DATA ENGINE (FIXED FEATURE SCALING)
# ──────────────────────────────────────────────────────────────────────────────

def create_federated_non_iid_ecosystem(num_clients=100, samples_per_client=120):
    """Simulates high-contrast structural digit signals to provide clean gradients"""
    clients_data = {}

    # Generate structured geometric features instead of purely random noise
    X_pool = np.zeros((num_clients * samples_per_client, 784), dtype=np.float32)
    y_pool = np.random.randint(0, 10, size=(num_clients * samples_per_client))

    # Build actual high-contrast digit masks so the convolutions can extract real filters
    for i in range(len(y_pool)):
        img = np.zeros((28, 28), dtype=np.float32)
        # Draw a structural box unique to each class to provide an obvious gradient signal
        digit_offset = y_pool[i] % 5
        img[2 + digit_offset : 12 + digit_offset, 2:24] = 1.0
        X_pool[i] = img.flatten()

    # Sort into shards to enforce extreme Non-IID conditions
    indices = np.argsort(y_pool)
    X_sorted, y_sorted = X_pool[indices], y_pool[indices]

    shard_size = samples_per_client // 2
    total_shards = (num_clients * samples_per_client) // shard_size

    for client_id in range(num_clients):
        shard_idx1 = (client_id * 2) % total_shards
        shard_idx2 = (client_id * 2 + 1) % total_shards

        start1, end1 = shard_idx1 * shard_size, (shard_idx1 + 1) * shard_size
        start2, end2 = shard_idx2 * shard_size, (shard_idx2 + 1) * shard_size

        X_c = np.concatenate([X_sorted[start1:end1], X_sorted[start2:end2]], axis=0)
        y_c = np.concatenate([y_sorted[start1:end1], y_sorted[start2:end2]], axis=0)

        clients_data[client_id] = (torch.tensor(X_c), torch.tensor(y_c, dtype=torch.long))

    # Build validation set mapping identical spatial patterns
    X_test_np = np.zeros((200, 784), dtype=np.float32)
    y_test_np = np.random.randint(0, 10, size=200)
    for i in range(len(y_test_np)):
        img = np.zeros((28, 28), dtype=np.float32)
        digit_offset = y_test_np[i] % 5
        img[2 + digit_offset : 12 + digit_offset, 2:24] = 1.0
        X_test_np[i] = img.flatten()

    return clients_data, torch.tensor(X_test_np), torch.tensor(y_test_np, dtype=torch.long)

# ──────────────────────────────────────────────────────────────────────────────
# 3. FEDERATED OPTIMIZATION CONTROLLER WITH SERVER MOMENTUM (FedAvgM)
# ──────────────────────────────────────────────────────────────────────────────

def get_flat_parameters(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()]).clone()

def set_flat_parameters(model, flat_params):
    index = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[index:index + numel].view(p.shape))
        index += numel

def run_deep_federated_learning(rounds=15, client_fraction=0.1, local_epochs=5):
    print("="*70)
    print("       INITIALIZING DEEP FEDERATED LEARNING PIPELINE (U-NET VARIANT)")
    print("="*70)

    clients_data, X_test, y_test = create_federated_non_iid_ecosystem()
    global_model = FederatedUNetClassifier()
    criterion = nn.CrossEntropyLoss()

    # Initialize weights using standard uniform variance scaling
    for layer in global_model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    total_params = sum(p.numel() for p in global_model.parameters())
    bytes_per_param = 4
    client_cost_kb = (total_params * bytes_per_param) / 1024

    beta = 0.9
    server_velocity = torch.zeros(total_params)
    num_active_per_round = max(1, int(client_fraction * 100))

    print(f"  Total Parameters in Deep CNN Block: {total_params:,}")
    print(f"  Calculated Per-Client Uplink Cost:   {client_cost_kb:.2f} KB/round")
    print(f"  Active Node Count Per Round:        {num_active_per_round} Clients")
    print("-"*70)

    # Global Coordination Loop
    for r in range(rounds):
        global_weights = get_flat_parameters(global_model)
        active_clients = np.random.choice(100, size=num_active_per_round, replace=False)

        local_updates = []

        for client_id in active_clients:
            X_local, y_local = clients_data[client_id]

            local_model = FederatedUNetClassifier()
            set_flat_parameters(local_model, global_weights)

            # Use an adaptive optimizer (Adam) locally to handle non-convex updates easily
            optimizer = optim.Adam(local_model.parameters(), lr=0.01)

            local_model.train()
            for epoch in range(local_epochs):
                optimizer.zero_grad()
                pred = local_model(X_local)
                loss = criterion(pred, y_local)
                loss.backward()
                optimizer.step()

            local_updates.append(get_flat_parameters(local_model))

        # FedAvg Aggregation step
        avg_local_weights = torch.stack(local_updates).mean(dim=0)

        # Apply Server Momentum (FedAvgM)
        pseudo_gradient = global_weights - avg_local_weights
        server_velocity = beta * server_velocity + pseudo_gradient
        updated_global_weights = global_weights - server_velocity

        set_flat_parameters(global_model, updated_global_weights)

        # Round Telemetry Evaluation
        global_model.eval()
        with torch.no_grad():
            test_preds = global_model(X_test)
            acc = (test_preds.argmax(dim=1) == y_test).float().mean().item() * 100

        print(f"  [Round {r+1:2d}/{rounds}] Federated Deep Model Validation Accuracy: {acc:.2f}%")

    total_fleet_mb = (rounds * num_active_per_round * total_params * bytes_per_param) / (1024 * 1024)
    print("-"*70)
    print(f"  Total Network Traffic Generated: {total_fleet_mb:.2f} MB")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_deep_federated_learning()
