import math
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT_DIR = Path(__file__).resolve().parents[2]
SIMULATOR_DIR = ROOT_DIR / "CCL_Simulator"
sys.path.append(str(SIMULATOR_DIR))

RAW_PATH = Path("data/raw/simnet_samples.pt")
SAVE_PATH = Path("data/processed/simnet_pyg.pt")


def build_node_mapping(topo):
    nodes = list(topo.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    return nodes, node_to_idx


def build_node_features(topo, nodes):
    x = []

    for node in nodes:
        node_data = topo.nodes[node]

        is_gpu = 1.0 if node_data.get("type", "gpu") == "gpu" else 0.0
        num_qps = float(node_data.get("num_qps", 1))
        quantum_packets = float(node_data.get("quantum_packets", 1))
        tx_proc_delay = float(node_data.get("tx_proc_delay", 0.0))
        gpu_store_delay = float(node_data.get("gpu_store_delay", 0.0))

        x.append([
            is_gpu,
            num_qps,
            quantum_packets,
            tx_proc_delay,
            gpu_store_delay,
        ])

    return torch.tensor(x, dtype=torch.float)


def aggregate_policy_on_edges(topo, policy):
    edge_stats = {}

    for src, dst in topo.edges():
        edge_stats[(src, dst)] = {
            "num_transfers": 0.0,
            "total_bytes": 0.0,
            "total_time": 0.0,
            "max_time": 0.0,
            "dependency_count": 0.0,
        }

    for entry in policy:
        path = entry.path
        chunk_size = float(entry.chunk_size_bytes)
        trigger_time = float(entry.time)

        dependency = getattr(entry, "dependency", [])
        dep_count = len(dependency) if dependency is not None else 0

        for u, v in zip(path[:-1], path[1:]):
            if (u, v) not in edge_stats:
                raise ValueError(f"Policy uses edge not in topology: {(u, v)}")

            edge_stats[(u, v)]["num_transfers"] += 1.0
            edge_stats[(u, v)]["total_bytes"] += chunk_size
            edge_stats[(u, v)]["total_time"] += trigger_time
            edge_stats[(u, v)]["max_time"] = max(edge_stats[(u, v)]["max_time"], trigger_time)
            edge_stats[(u, v)]["dependency_count"] += dep_count

    return edge_stats


def build_edge_tensors(topo, node_to_idx, policy):
    edge_stats = aggregate_policy_on_edges(topo, policy)

    edge_index = []
    edge_attr = []

    link_rates = []
    prop_delays = []

    for src, dst, data in topo.edges(data=True):
        stat = edge_stats[(src, dst)]

        num_transfers = stat["num_transfers"]
        total_bytes = stat["total_bytes"]

        if num_transfers > 0:
            avg_trigger_time = stat["total_time"] / num_transfers
            avg_dependency_count = stat["dependency_count"] / num_transfers
        else:
            avg_trigger_time = 0.0
            avg_dependency_count = 0.0

        max_trigger_time = stat["max_time"]

        link_rate = float(data.get("link_rate_bps", 0.0))
        prop_delay = float(data.get("prop_delay", 0.0))

        link_rates.append(link_rate)
        prop_delays.append(prop_delay)

        edge_index.append([node_to_idx[src], node_to_idx[dst]])

        edge_attr.append([
            math.log10(link_rate + 1.0),
            prop_delay,
            num_transfers,
            math.log10(total_bytes + 1.0),
            avg_trigger_time,
            max_trigger_time,
            avg_dependency_count,
        ])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return edge_index, edge_attr, link_rates, prop_delays


def build_global_features(sample, topo, link_rates, prop_delays):
    chunk_size = float(sample.get("chunk_size_bytes", 1.0))
    num_nodes = float(topo.number_of_nodes())
    num_edges = float(topo.number_of_edges())

    max_possible_directed_edges = max(num_nodes * (num_nodes - 1), 1.0)
    density = num_edges / max_possible_directed_edges

    avg_log_link_rate = sum(math.log10(r + 1.0) for r in link_rates) / max(len(link_rates), 1)
    avg_prop_delay = sum(prop_delays) / max(len(prop_delays), 1)

    u = torch.tensor([[
        math.log2(chunk_size + 1.0),
        num_nodes,
        num_edges,
        density,
        avg_log_link_rate,
        avg_prop_delay,
    ]], dtype=torch.float)

    return u


def make_pyg_graph(sample):
    topo = sample["topology"]
    policy = sample["policy"]
    completion_time = float(sample["completion_time"])

    nodes, node_to_idx = build_node_mapping(topo)

    x = build_node_features(topo, nodes)
    edge_index, edge_attr, link_rates, prop_delays = build_edge_tensors(topo, node_to_idx, policy)
    u = build_global_features(sample, topo, link_rates, prop_delays)

    y = torch.tensor([math.log(max(completion_time, 1e-12))], dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        u=u,
        y=y,
    )

    data.raw_completion_time = completion_time
    data.topology_type = sample.get("topology_type", "unknown")
    data.policy_type = sample.get("policy_type", "unknown")
    data.chunk_mb = sample.get("chunk_mb", -1)
    data.num_nodes_original = topo.number_of_nodes()
    data.num_edges_original = topo.number_of_edges()
    data.num_policy_entries = len(policy)

    return data


def main():
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_samples = torch.load(RAW_PATH, weights_only=False)

    pyg_graphs = []
    skipped = 0

    for i, sample in enumerate(raw_samples):
        try:
            graph = make_pyg_graph(sample)
            pyg_graphs.append(graph)
        except Exception as e:
            skipped += 1
            print(f"[{i}] skipped: {e}")

    torch.save(pyg_graphs, SAVE_PATH)

    print(f"Saved {len(pyg_graphs)} PyG graphs to {SAVE_PATH}")
    print(f"Skipped: {skipped}")

    if len(pyg_graphs) > 0:
        g = pyg_graphs[0]
        print("\nExample graph:")
        print("x:", g.x.shape)
        print("edge_index:", g.edge_index.shape)
        print("edge_attr:", g.edge_attr.shape)
        print("u:", g.u.shape)
        print("y:", g.y)
        print("raw_completion_time:", g.raw_completion_time)
        print("topology_type:", g.topology_type)
        print("policy_type:", g.policy_type)


if __name__ == "__main__":
    main()