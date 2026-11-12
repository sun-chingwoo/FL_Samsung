"""
Minimal Federated Learning (FedAvg) simulation - Phase 1.
Fashion-MNIST, 10 clients, IID split, plain PyTorch.

Install:  pip install torch torchvision matplotlib
Run:      python fl_simple.py
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

# ---------------- config ----------------
NUM_CLIENTS = 5
ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

# ---------------- data ----------------
def load_data():
    train = datasets.FashionMNIST("./data", train=True, download=True)
    test = datasets.FashionMNIST("./data", train=False, download=True)
    # scale to [0,1], normalize, add channel dim: (N, 1, 28, 28)
    prep = lambda d: ((d.data.float() / 255.0) - 0.2860) / 0.3530
    xtr, ytr = prep(train).unsqueeze(1), train.targets
    xte, yte = prep(test).unsqueeze(1), test.targets
    return xtr, ytr, xte, yte

def split_iid(x, y, num_clients):
    """Shuffle and split equally. Returns a list of (x, y) per client."""
    idx = torch.randperm(len(x))
    chunks = torch.chunk(idx, num_clients)
    return [(x[c], y[c]) for c in chunks]

# ---------------- model ----------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))

# ---------------- client ----------------
def local_train(global_model, x, y):
    """Copy the global model, train on local data, return new weights."""
    model = copy.deepcopy(global_model).to(DEVICE)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    for _ in range(LOCAL_EPOCHS):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            xb, yb = x[b].to(DEVICE), y[b].to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    return {k: v.cpu() for k, v in model.state_dict().items()}

# ---------------- server ----------------
def fedavg(weights_list, sizes):
    """Weighted average of client weights (weighted by dataset size)."""
    total = sum(sizes)
    avg = {}
    for k in weights_list[0]:
        avg[k] = sum(w[k] * (n / total) for w, n in zip(weights_list, sizes))
    return avg

@torch.no_grad()
def evaluate(model, x, y):
    model.eval().to(DEVICE)
    correct = 0
    for i in range(0, len(x), 1000):
        pred = model(x[i:i + 1000].to(DEVICE)).argmax(1).cpu()
        correct += (pred == y[i:i + 1000]).sum().item()
    return correct / len(x)

# ---------------- main ----------------
def main():
    xtr, ytr, xte, yte = load_data()
    clients = split_iid(xtr, ytr, NUM_CLIENTS)
    global_model = CNN()
    history = []

    # Phase 2 hook: put malicious client ids here, e.g. {0, 1}
    malicious = set()

    for rnd in range(1, ROUNDS + 1):
        updates, sizes = [], []
        for cid, (cx, cy) in enumerate(clients):
            if cid in malicious:
                pass  # Phase 2: poison cx/cy or the update here
            updates.append(local_train(global_model, cx, cy))
            sizes.append(len(cx))
        global_model.load_state_dict(fedavg(updates, sizes))
        acc = evaluate(global_model, xte, yte)
        history.append(acc)
        print(f"Round {rnd:2d} | test accuracy: {acc:.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.plot(range(1, ROUNDS + 1), history, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Test accuracy")
        plt.title("FedAvg on Fashion-MNIST (IID, 10 clients)")
        plt.grid(True)
        plt.savefig("accuracy.png", dpi=150)
        print("Saved accuracy.png")
    except ImportError:
        print("matplotlib not installed, skipping plot")

if __name__ == "__main__":
    main()