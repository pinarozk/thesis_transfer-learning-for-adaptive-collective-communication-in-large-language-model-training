# src/data/agent_pyg_dataset.py

from pathlib import Path
import torch
from torch_geometric.data import Data


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"


def matrix_to_edge_targets(matrix, edge_index):
    """
    matrix: [N, N]
    edge_index: [2, E]
    return: [E]
    """
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    return matrix[src_nodes, dst_nodes]


def convert_raw_agent_sample(sample):
    edge_routing_target = matrix_to_edge_targets(
        sample["routing_target"],
        sample["edge_index"]
    )

    edge_scheduling_target = matrix_to_edge_targets(
        sample["scheduling_target"],
        sample["edge_index"]
    )

    return {
        "source": sample.get("source"),
        "topology_name": sample["topology_name"],
        "chassis": sample["chassis"],
        "collective": sample["collective"],
        "mode": sample["mode"],
        "message_size": sample["message_size"],
        "node_feat": sample["node_feat"],
        "edge_index": sample["edge_index"],
        "edge_attr": sample["edge_attr"],
        "edge_routing_target": edge_routing_target.float(),
        "edge_scheduling_target": edge_scheduling_target.float(),
    }


def make_agent_pyg_data(sample):
    return Data(
        x=sample["node_feat"],
        edge_index=sample["edge_index"],
        edge_attr=sample["edge_attr"],

        y_routing=sample["edge_routing_target"].float(),
        y_scheduling=sample["edge_scheduling_target"].float(),
    )


def build_agent_pyg_splits():
    raw_train = torch.load(DATA_DIR / "agent_train.pt", weights_only=False)
    raw_val = torch.load(DATA_DIR / "agent_val.pt", weights_only=False)
    raw_test = torch.load(DATA_DIR / "agent_test.pt", weights_only=False)

    agent_train = [convert_raw_agent_sample(s) for s in raw_train]
    agent_val = [convert_raw_agent_sample(s) for s in raw_val]
    agent_test = [convert_raw_agent_sample(s) for s in raw_test]

    agent_train_data = [make_agent_pyg_data(s) for s in agent_train]
    agent_val_data = [make_agent_pyg_data(s) for s in agent_val]
    agent_test_data = [make_agent_pyg_data(s) for s in agent_test]

    torch.save(agent_train_data, DATA_DIR / "agent_train_pyg.pt")
    torch.save(agent_val_data, DATA_DIR / "agent_val_pyg.pt")
    torch.save(agent_test_data, DATA_DIR / "agent_test_pyg.pt")

    print("Num train graphs:", len(agent_train_data))
    print("Num val graphs:", len(agent_val_data))
    print("Num test graphs:", len(agent_test_data))

    if agent_train_data:
        d = agent_train_data[0]
        print("\nExample graph:")
        print("x:", d.x.shape)
        print("edge_index:", d.edge_index.shape)
        print("edge_attr:", d.edge_attr.shape)
        print("routing:", d.y_routing.shape)
        print("scheduling:", d.y_scheduling.shape)
        print("used routing edges:", int(d.y_routing.sum().item()))

    print("\nSaved:")
    print(DATA_DIR / "agent_train_pyg.pt")
    print(DATA_DIR / "agent_val_pyg.pt")
    print(DATA_DIR / "agent_test_pyg.pt")

    return agent_train_data, agent_val_data, agent_test_data


if __name__ == "__main__":
    build_agent_pyg_splits()