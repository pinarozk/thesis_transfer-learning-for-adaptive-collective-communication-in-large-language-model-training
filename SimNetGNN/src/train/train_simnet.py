import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

from src.models.simnet_gnn import SimNetGNN


DATA_PATH = Path("data/processed/simnet_pyg.pt")

BATCH_SIZE = 32
EPOCHS = 130
LR = 5e-4


def load_data():
    graphs = torch.load(DATA_PATH, weights_only=False)

    train, temp = train_test_split(graphs, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    return train, val, test


def get_loaders(train, val, test):
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.u)
        target = batch.y.view(-1)

        loss = criterion(pred, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.u)
        target = batch.y.view(-1)

        loss = criterion(pred, target)
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_original_scale(model, loader, device):
    model.eval()

    preds = []
    targets = []

    for batch in loader:
        batch = batch.to(device)

        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.u)

        preds.append(pred.cpu())
        targets.append(batch.y.view(-1).cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    preds = np.exp(preds)
    targets = np.exp(targets)

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    return mae, rmse


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val, test = load_data()
    train_loader, val_loader, test_loader = get_loaders(train, val, test)

    sample = train[0]

    model = SimNetGNN(
        node_in_dim=sample.x.shape[1],
        edge_in_dim=sample.edge_attr.shape[1],
        global_dim=sample.u.shape[1],
        hidden_dim=128,
        heads=4,
        dropout=0.10,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    mae, rmse = evaluate_original_scale(model, test_loader, device)

    print("\nFinal Evaluation:")
    print("MAE :", mae)
    print("RMSE:", rmse)

    # save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/simnet_best.pt")

    print("\nModel saved to models/simnet_best.pt")


if __name__ == "__main__":
    main()