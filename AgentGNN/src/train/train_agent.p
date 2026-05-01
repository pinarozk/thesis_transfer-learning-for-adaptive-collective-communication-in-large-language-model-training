import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.models.agent_gnn import StrongAgentGNN


DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def compute_agent_loss(model, batch, routing_criterion, scheduling_criterion, lambda_sched):
    routing_logits, scheduling_pred = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
    )

    y_routing = batch.y_routing.float()
    y_scheduling = batch.y_scheduling.float()

    routing_loss = routing_criterion(routing_logits, y_routing)

    used_mask = y_routing == 1

    if used_mask.sum() > 0:
        scheduling_loss = scheduling_criterion(
            scheduling_pred[used_mask],
            y_scheduling[used_mask],
        )
    else:
        scheduling_loss = torch.tensor(0.0, device=batch.x.device)

    total_loss = routing_loss + lambda_sched * scheduling_loss

    return total_loss, routing_loss, scheduling_loss


def train_epoch(model, loader, optimizer, routing_criterion, scheduling_criterion, lambda_sched, device):
    model.train()

    total_loss = 0.0
    total_route = 0.0
    total_sched = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        loss, route_loss, sched_loss = compute_agent_loss(
            model,
            batch,
            routing_criterion,
            scheduling_criterion,
            lambda_sched,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()
        total_route += route_loss.item()
        total_sched += sched_loss.item()

    n = len(loader)
    return total_loss / n, total_route / n, total_sched / n


@torch.no_grad()
def eval_epoch(model, loader, routing_criterion, scheduling_criterion, lambda_sched, device):
    model.eval()

    total_loss = 0.0
    total_route = 0.0
    total_sched = 0.0

    for batch in loader:
        batch = batch.to(device)

        loss, route_loss, sched_loss = compute_agent_loss(
            model,
            batch,
            routing_criterion,
            scheduling_criterion,
            lambda_sched,
        )

        total_loss += loss.item()
        total_route += route_loss.item()
        total_sched += sched_loss.item()

    n = len(loader)
    return total_loss / n, total_route / n, total_sched / n


@torch.no_grad()
def evaluate_routing_scheduling(model, loader, device, threshold=0.5):
    model.eval()

    total_correct = 0
    total_edges = 0
    tp = fp = fn = 0

    total_abs_error = 0.0
    total_sched_edges = 0

    for batch in loader:
        batch = batch.to(device)

        routing_logits, scheduling_pred = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
        )

        pred = (torch.sigmoid(routing_logits) > threshold).float()
        y = batch.y_routing.float()

        total_correct += (pred == y).sum().item()
        total_edges += y.numel()

        tp += ((pred == 1) & (y == 1)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()

        used_mask = y == 1

        if used_mask.sum() > 0:
            total_abs_error += torch.abs(
                scheduling_pred[used_mask] - batch.y_scheduling[used_mask]
            ).sum().item()

            total_sched_edges += used_mask.sum().item()

    accuracy = total_correct / total_edges
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    sched_mae = total_abs_error / (total_sched_edges + 1e-8)

    return accuracy, precision, recall, sched_mae


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_data = torch.load(DATA_DIR / "agent_train_pyg.pt", weights_only=False)
    val_data = torch.load(DATA_DIR / "agent_val_pyg.pt", weights_only=False)
    test_data = torch.load(DATA_DIR / "agent_test_pyg.pt", weights_only=False)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, pin_memory=pin_memory)

    first_graph = train_data[0]

    model = StrongAgentGNN(
        node_in_dim=first_graph.x.shape[1],
        edge_in_dim=first_graph.edge_attr.shape[1],
        hidden_dim=96,
        heads=4,
        dropout=0.15,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    routing_criterion = nn.BCEWithLogitsLoss()
    scheduling_criterion = nn.MSELoss()

    lambda_sched = 1.0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
    )

    num_epochs = 80
    patience = 8
    patience_counter = 0

    best_val_loss = float("inf")
    best_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_routing_loss": [],
        "train_scheduling_loss": [],
        "val_routing_loss": [],
        "val_scheduling_loss": [],
    }

    for epoch in range(1, num_epochs + 1):
        train_loss, train_route_loss, train_sched_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            routing_criterion,
            scheduling_criterion,
            lambda_sched,
            device,
        )

        val_loss, val_route_loss, val_sched_loss = eval_epoch(
            model,
            val_loader,
            routing_criterion,
            scheduling_criterion,
            lambda_sched,
            device,
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_routing_loss"].append(train_route_loss)
        history["train_scheduling_loss"].append(train_sched_loss)
        history["val_routing_loss"].append(val_route_loss)
        history["val_scheduling_loss"].append(val_sched_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 5 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d} | "
                f"Train: {train_loss:.6f} "
                f"(R: {train_route_loss:.6f}, S: {train_sched_loss:.6f}) | "
                f"Val: {val_loss:.6f} "
                f"(R: {val_route_loss:.6f}, S: {val_sched_loss:.6f}) | "
                f"LR: {lr:.6f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    test_loss, test_route_loss, test_sched_loss = eval_epoch(
        model,
        test_loader,
        routing_criterion,
        scheduling_criterion,
        lambda_sched,
        device,
    )

    acc, prec, rec, sched_mae = evaluate_routing_scheduling(
        model,
        test_loader,
        device,
    )

    print("\nBest validation loss:", best_val_loss)

    print("\nTest Loss:")
    print("Total:", test_loss)
    print("Routing:", test_route_loss)
    print("Scheduling:", test_sched_loss)

    print("\nRouting Metrics:")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)

    print("\nScheduling Metrics:")
    print("MAE:", sched_mae)

    torch.save(model.state_dict(), MODEL_DIR / "agent_gnn_best.pt")
    torch.save(history, MODEL_DIR / "agent_training_history.pt")

    print("\nSaved model:")
    print(MODEL_DIR / "agent_gnn_best.pt")


if __name__ == "__main__":
    main()
    