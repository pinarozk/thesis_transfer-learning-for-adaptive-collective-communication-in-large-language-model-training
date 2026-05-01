# src/eval/evaluate_agent.py

import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.models.agent_gnn import StrongAgentGNN


DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
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


@torch.no_grad()
def visualize_single_graph(model, data, device, threshold=0.5, name="agent_example"):
    model.eval()
    data = data.to(device)

    routing_logits, scheduling_pred = model(
        data.x,
        data.edge_index,
        data.edge_attr,
    )

    routing_prob = torch.sigmoid(routing_logits).cpu()
    routing_pred = (routing_prob > threshold).float()
    routing_true = data.y_routing.cpu()

    scheduling_pred = scheduling_pred.cpu()
    scheduling_true = data.y_scheduling.cpu()

    edge_index = data.edge_index.cpu()
    n_nodes = data.x.shape[0]

    true_route_mat = torch.zeros((n_nodes, n_nodes))
    pred_route_mat = torch.zeros((n_nodes, n_nodes))
    true_sched_mat = torch.zeros((n_nodes, n_nodes))
    pred_sched_mat = torch.zeros((n_nodes, n_nodes))

    for e in range(edge_index.shape[1]):
        i = edge_index[0, e]
        j = edge_index[1, e]

        true_route_mat[i, j] = routing_true[e]
        pred_route_mat[i, j] = routing_pred[e]

        true_sched_mat[i, j] = scheduling_true[e]
        pred_sched_mat[i, j] = scheduling_pred[e]

    # Routing comparison
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(true_route_mat)
    plt.title("True Routing")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(pred_route_mat)
    plt.title("Predicted Routing")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}_routing.png", dpi=200)
    plt.show()

    # Scheduling comparison
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(true_sched_mat)
    plt.title("True Scheduling")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(pred_sched_mat)
    plt.title("Predicted Scheduling")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}_scheduling.png", dpi=200)
    plt.show()

    print("Saved figures:")
    print(FIG_DIR / f"{name}_routing.png")
    print(FIG_DIR / f"{name}_scheduling.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    test_data = torch.load(DATA_DIR / "agent_test_pyg.pt", weights_only=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    first_graph = test_data[0]

    model = StrongAgentGNN(
        node_in_dim=first_graph.x.shape[1],
        edge_in_dim=first_graph.edge_attr.shape[1],
        hidden_dim=96,
        heads=4,
        dropout=0.15,
    ).to(device)

    model.load_state_dict(
        torch.load(MODEL_DIR / "agent_gnn_best.pt", map_location=device)
    )

    acc, prec, rec, sched_mae = evaluate(model, test_loader, device)

    print("\nEvaluation Results")
    print("Routing Accuracy:", acc)
    print("Routing Precision:", prec)
    print("Routing Recall:", rec)
    print("Scheduling MAE:", sched_mae)

    visualize_single_graph(
        model=model,
        data=test_data[0],
        device=device,
        threshold=0.5,
        name="test_sample_0",
    )


if __name__ == "__main__":
    main()
diff = torch.abs(true_route_mat - pred_route_mat)

