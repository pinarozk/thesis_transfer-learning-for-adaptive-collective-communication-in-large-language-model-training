# src/data/build_agent_dataset.py

import sys
import json
import re
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "external" / "TE-CCL"))
sys.path.append(str(BASE_DIR / "external" / "ilp_ccl_tree"))

from src.data.build_metadata import build_all_metadata

from teccl.input_data import (
    UserInputParams,
    ObjectiveType,
    SolutionMethod,
    Collective,
    EpochType,
)
from teccl.scheduler import TECCLSolver

from tree_generate import generate_and_select_trees
from topo_generate import generate_delay_and_capacity


def build_collective_vocab(metadata_rows):
    collective_vocab = sorted(set(row["collective"] for row in metadata_rows))
    collective_to_idx = {c: i for i, c in enumerate(collective_vocab)}
    return collective_vocab, collective_to_idx


def grouped_split(dataset, group_fields, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    group_to_items = defaultdict(list)

    for item in dataset:
        key = tuple(item.get(f) for f in group_fields)
        group_to_items[key].append(item)

    train, val, test = [], [], []

    for items in group_to_items.values():
        items = list(items)
        rng.shuffle(items)

        n = len(items)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    return train, val, test


def build_teccl_topology_graph(
    topology_name,
    chassis,
    collective_name,
    collective_vocab,
    collective_to_idx,
):
    user_input = UserInputParams()
    user_input.topology.name = topology_name
    user_input.topology.chassis = chassis

    user_input.instance.collective = Collective.ALLGATHER
    user_input.instance.objective_type = ObjectiveType.PAPER
    user_input.instance.solution_method = SolutionMethod.ONE_SHOT
    user_input.instance.epoch_type = EpochType.FASTEST_LINK

    solver = TECCLSolver(user_input)
    topo = solver.topology_obj

    cap = np.array(topo.capacity, dtype=float)
    alp = np.array(topo.alpha, dtype=float)

    n_nodes = cap.shape[0]
    switch_set = set(topo.switch_indices)

    edge_pairs = []
    edge_features = []

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and cap[i, j] > 0:
                edge_pairs.append((i, j))
                edge_features.append([
                    cap[i, j],
                    alp[i, j] if alp[i, j] >= 0 else 0.0,
                    float(i in switch_set),
                    float(j in switch_set),
                ])

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    edge_attr_norm = edge_attr.clone()
    if edge_attr_norm[:, 0].max() > 0:
        edge_attr_norm[:, 0] /= edge_attr_norm[:, 0].max()
    if edge_attr_norm[:, 1].max() > 0:
        edge_attr_norm[:, 1] /= edge_attr_norm[:, 1].max()

    collective_onehot = np.zeros(len(collective_vocab), dtype=float)
    collective_onehot[collective_to_idx[collective_name]] = 1.0

    node_feat = torch.tensor(
        [[float(i in switch_set)] + collective_onehot.tolist() for i in range(n_nodes)],
        dtype=torch.float,
    )

    return {
        "N": n_nodes,
        "node_feat": node_feat,
        "edge_index": edge_index,
        "edge_attr": edge_attr_norm,
    }


def extract_teccl_routing_scheduling_and_time(schedule_path, expected_n):
    with open(schedule_path, "r") as f:
        schedule_data = json.load(f)

    if "7-Flows" in schedule_data:
        flows = schedule_data["7-Flows"]
    elif "6-Flows" in schedule_data:
        flows = schedule_data["6-Flows"]
    else:
        raise KeyError(f"No flow key found in {schedule_path}")

    routing_matrix = np.zeros((expected_n, expected_n), dtype=float)
    scheduling_matrix = np.full((expected_n, expected_n), -1.0, dtype=float)

    pattern = r"traveled over (\d+)->(\d+) in epoch (\d+)"
    max_epoch = 0

    for flow_str in flows:
        match = re.search(pattern, flow_str)
        if not match:
            continue

        src = int(match.group(1))
        dst = int(match.group(2))
        epoch = int(match.group(3))

        if src < expected_n and dst < expected_n:
            routing_matrix[src, dst] = 1.0

            if scheduling_matrix[src, dst] < 0:
                scheduling_matrix[src, dst] = float(epoch)
            else:
                scheduling_matrix[src, dst] = min(scheduling_matrix[src, dst], float(epoch))

            max_epoch = max(max_epoch, epoch)

    if max_epoch > 0:
        scheduling_matrix = np.where(
            scheduling_matrix >= 0,
            scheduling_matrix / max_epoch,
            0.0,
        )
    else:
        scheduling_matrix = np.where(
            scheduling_matrix >= 0,
            scheduling_matrix,
            0.0,
        )

    return (
        torch.tensor(routing_matrix, dtype=torch.float),
        torch.tensor(scheduling_matrix, dtype=torch.float),
    )


def build_ccl_synthetic_sample(
    collective_name,
    collective_vocab,
    collective_to_idx,
    tree_number=3,
):
    n_nodes = 8

    topo_demo = [[0] * n_nodes for _ in range(n_nodes)]

    for i in range(0, 4):
        for j in range(0, 4):
            if i != j:
                topo_demo[i][j] = 1

    for i in range(4, 8):
        for j in range(4, 8):
            if i != j:
                topo_demo[i][j] = 1

    cross_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in cross_edges:
        topo_demo[i][j] = 1
        topo_demo[j][i] = 1

    transfer_delay, capacity = generate_delay_and_capacity(topo_demo)

    edge_pairs = []
    edge_features = []

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and topo_demo[i][j] == 1:
                cap = capacity[i, j]
                delay = transfer_delay[i, j]

                edge_pairs.append((i, j))
                edge_features.append([
                    cap,
                    delay if np.isfinite(delay) else 0.0,
                    0.0,
                    0.0,
                ])

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    edge_attr_norm = edge_attr.clone()
    if edge_attr_norm[:, 0].max() > 0:
        edge_attr_norm[:, 0] /= edge_attr_norm[:, 0].max()
    if edge_attr_norm[:, 1].max() > 0:
        edge_attr_norm[:, 1] /= edge_attr_norm[:, 1].max()

    collective_onehot = np.zeros(len(collective_vocab), dtype=float)
    collective_onehot[collective_to_idx[collective_name]] = 1.0

    node_feat = torch.tensor(
        [[0.0] + collective_onehot.tolist() for _ in range(n_nodes)],
        dtype=torch.float,
    )

    _, selected_trees = generate_and_select_trees(
        topo_demo,
        select_n=tree_number,
        enumeration_limit=2000,
        sample_tries=500,
        verbose=False,
        do_plot=False,
    )

    routing_matrix = np.zeros((n_nodes, n_nodes), dtype=float)
    scheduling_matrix = np.zeros((n_nodes, n_nodes), dtype=float)

    for (_, _), edges in selected_trees.items():
        for depth, (i, j) in enumerate(edges):
            routing_matrix[i, j] = 1.0
            if scheduling_matrix[i, j] == 0:
                scheduling_matrix[i, j] = depth + 1
            else:
                scheduling_matrix[i, j] = min(scheduling_matrix[i, j], depth + 1)

    if scheduling_matrix.max() > 0:
        scheduling_matrix = scheduling_matrix / scheduling_matrix.max()

    return {
        "source": "ccl",
        "topology_name": "custom",
        "chassis": None,
        "collective": collective_name,
        "mode": "tree_generated",
        "message_size": "synthetic",
        "node_feat": node_feat,
        "edge_index": edge_index,
        "edge_attr": edge_attr_norm,
        "routing_target": torch.tensor(routing_matrix, dtype=torch.float),
        "scheduling_target": torch.tensor(scheduling_matrix, dtype=torch.float),
    }


def build_agent_dataset():
    metadata_rows = build_all_metadata()
    collective_vocab, collective_to_idx = build_collective_vocab(metadata_rows)

    print("Collective vocab:", collective_to_idx)

    agent_dataset = []
    topology_cache = {}
    skipped_files = []

    for row in metadata_rows:
        try:
            if row["source"] == "teccl":
                topo_key = (
                    row["topology_name"],
                    row["chassis"],
                    row["collective"],
                )

                if topo_key not in topology_cache:
                    topology_cache[topo_key] = build_teccl_topology_graph(
                        topology_name=row["topology_name"],
                        chassis=row["chassis"],
                        collective_name=row["collective"],
                        collective_vocab=collective_vocab,
                        collective_to_idx=collective_to_idx,
                    )

                graph_data = topology_cache[topo_key]

                routing_target, scheduling_target = extract_teccl_routing_scheduling_and_time(
                    row["path"],
                    expected_n=graph_data["N"],
                )

                agent_dataset.append({
                    **row,
                    "node_feat": graph_data["node_feat"],
                    "edge_index": graph_data["edge_index"],
                    "edge_attr": graph_data["edge_attr"],
                    "routing_target": routing_target,
                    "scheduling_target": scheduling_target,
                })

            elif row["source"] == "ccl":
                ccl_sample = build_ccl_synthetic_sample(
                    collective_name=row["collective"],
                    collective_vocab=collective_vocab,
                    collective_to_idx=collective_to_idx,
                    tree_number=3,
                )
                agent_dataset.append(ccl_sample)

        except Exception as e:
            skipped_files.append((row.get("path", row["source"]), str(e)))

    print("Agent dataset size:", len(agent_dataset))
    print("Skipped files:", len(skipped_files))

    if agent_dataset:
        first = agent_dataset[0]
        print("First agent sample:")
        print({
            "source": first.get("source"),
            "topology_name": first["topology_name"],
            "collective": first["collective"],
            "routing_target_shape": first["routing_target"].shape,
            "scheduling_target_shape": first["scheduling_target"].shape,
        })

    if skipped_files[:5]:
        print("First skipped files:")
        for path, err in skipped_files[:5]:
            print(path, "->", err)

    output_dir = BASE_DIR / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(agent_dataset, output_dir / "agent_dataset.pt")

    group_fields = ["source", "topology_name", "chassis", "collective", "mode"]
    agent_train, agent_val, agent_test = grouped_split(agent_dataset, group_fields)

    torch.save(agent_train, output_dir / "agent_train.pt")
    torch.save(agent_val, output_dir / "agent_val.pt")
    torch.save(agent_test, output_dir / "agent_test.pt")

    print("Agent split:", len(agent_train), len(agent_val), len(agent_test))
    print("Saved:")
    print(output_dir / "agent_dataset.pt")
    print(output_dir / "agent_train.pt")
    print(output_dir / "agent_val.pt")
    print(output_dir / "agent_test.pt")

    return agent_dataset


if __name__ == "__main__":
    build_agent_dataset()