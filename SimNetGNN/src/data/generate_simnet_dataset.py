import random
from pathlib import Path

import torch
import networkx as nx

from simulator_wrapper import run_simulator
from simcore import PolicyEntry


NUM_SAMPLES = 1000
SAVE_PATH = Path("data/raw/simnet_samples.pt")


print("RUNNING FILE:", __file__)
print("NUM_SAMPLES:", NUM_SAMPLES)
print("SAVE_PATH:", SAVE_PATH.resolve())


LINK_RATES = [25e9, 50e9, 100e9, 200e9]
CHUNK_MB_OPTIONS = [1, 4, 16, 64, 128]


def add_gpu_nodes(G, n):
    for i in range(n):
        G.add_node(
            f"GPU{i}",
            type="gpu",
            num_qps=random.choice([1, 2, 4]),
            quantum_packets=random.choice([1, 2, 4]),
            tx_proc_delay=0.0,
            gpu_store_delay=0.0,
        )


def add_edge_pair(G, u, v, rate=None, delay=None):
    rate = rate if rate is not None else random.choice(LINK_RATES)
    delay = delay if delay is not None else random.uniform(0.0, 0.002)

    G.add_edge(u, v, link_rate_bps=rate, prop_delay=delay)
    G.add_edge(v, u, link_rate_bps=rate, prop_delay=delay)


def build_line_topology(n):
    G = nx.DiGraph()
    add_gpu_nodes(G, n)

    for i in range(n - 1):
        add_edge_pair(G, f"GPU{i}", f"GPU{i+1}")

    G.graph["topology_type"] = "line"
    return G


def build_ring_topology(n):
    G = build_line_topology(n)
    add_edge_pair(G, f"GPU{n-1}", "GPU0")
    G.graph["topology_type"] = "ring"
    return G


def build_star_topology(n):
    G = nx.DiGraph()
    add_gpu_nodes(G, n)

    center = "GPU0"
    for i in range(1, n):
        add_edge_pair(G, center, f"GPU{i}")

    G.graph["topology_type"] = "star"
    return G


def build_bottleneck_topology(n):
    G = nx.DiGraph()
    add_gpu_nodes(G, n)

    split = n // 2

    # left cluster
    for i in range(split - 1):
        add_edge_pair(G, f"GPU{i}", f"GPU{i+1}", rate=100e9)

    # right cluster
    for i in range(split, n - 1):
        add_edge_pair(G, f"GPU{i}", f"GPU{i+1}", rate=100e9)

    # bottleneck bridge
    add_edge_pair(G, f"GPU{split-1}", f"GPU{split}", rate=25e9, delay=random.uniform(0.001, 0.004))

    G.graph["topology_type"] = "bottleneck"
    return G


def build_random_topology(n):
    G = nx.DiGraph()
    add_gpu_nodes(G, n)

    # ensure connected backbone
    for i in range(n - 1):
        add_edge_pair(G, f"GPU{i}", f"GPU{i+1}")

    # add random extra links
    possible = [
        (f"GPU{i}", f"GPU{j}")
        for i in range(n)
        for j in range(i + 1, n)
        if abs(i - j) > 1
    ]

    random.shuffle(possible)
    extra_edges = random.randint(1, min(len(possible), n))

    for u, v in possible[:extra_edges]:
        add_edge_pair(G, u, v)

    G.graph["topology_type"] = "random"
    return G


def random_topology():
    n = random.choice([4, 6, 8])
    topo_type = random.choice(["line", "ring", "star", "bottleneck", "random"])

    if topo_type == "line":
        return build_line_topology(n)
    if topo_type == "ring":
        return build_ring_topology(n)
    if topo_type == "star":
        return build_star_topology(n)
    if topo_type == "bottleneck":
        return build_bottleneck_topology(n)
    return build_random_topology(n)


def shortest_path_policy_entry(chunk_id, topo, src, dst, chunk_size, trigger_time, dependency=None):
    path = nx.shortest_path(topo, source=src, target=dst)

    return PolicyEntry(
        chunk_id,
        src,
        dst,
        qpid=random.randint(0, 1),
        rate="Max",
        chunk_size_bytes=chunk_size,
        path=path,
        time=trigger_time,
        dependency=dependency or [],
    )


def generate_random_policy(topo, chunk_size):
    nodes = list(topo.nodes())
    num_transfers = random.randint(2, min(10, len(nodes) * 2))

    policy = []

    for i in range(num_transfers):
        src = random.choice(nodes)
        dst = random.choice([n for n in nodes if n != src])

        entry = shortest_path_policy_entry(
            f"R{i}",
            topo,
            src,
            dst,
            chunk_size,
            trigger_time=random.uniform(0.0, 1.0),
        )
        policy.append(entry)

    return policy


def generate_fanout_policy(topo, chunk_size):
    nodes = list(topo.nodes())
    src = random.choice(nodes)
    dsts = random.sample([n for n in nodes if n != src], k=random.randint(2, min(4, len(nodes) - 1)))

    policy = []
    for i, dst in enumerate(dsts):
        policy.append(
            shortest_path_policy_entry(
                f"FO{i}",
                topo,
                src,
                dst,
                chunk_size,
                trigger_time=0.0,
            )
        )

    return policy


def generate_fanin_policy(topo, chunk_size):
    nodes = list(topo.nodes())
    dst = random.choice(nodes)
    srcs = random.sample([n for n in nodes if n != dst], k=random.randint(2, min(4, len(nodes) - 1)))

    policy = []
    for i, src in enumerate(srcs):
        policy.append(
            shortest_path_policy_entry(
                f"FI{i}",
                topo,
                src,
                dst,
                chunk_size,
                trigger_time=0.0,
            )
        )

    return policy


def generate_dependency_chain_policy(topo, chunk_size):
    nodes = list(topo.nodes())
    chain_len = random.randint(3, min(5, len(nodes)))
    chain_nodes = random.sample(nodes, chain_len)

    policy = []
    prev_chunk = None

    for i in range(chain_len - 1):
        chunk_id = f"CH{i}"
        dependency = [prev_chunk] if prev_chunk is not None else []

        policy.append(
            shortest_path_policy_entry(
                chunk_id,
                topo,
                chain_nodes[i],
                chain_nodes[i + 1],
                chunk_size,
                trigger_time=0.0,
                dependency=dependency,
            )
        )

        prev_chunk = chunk_id

    return policy


def generate_policy(topo, chunk_size):
    policy_type = random.choice(["random", "fanout", "fanin", "chain"])

    if policy_type == "fanout":
        policy = generate_fanout_policy(topo, chunk_size)
    elif policy_type == "fanin":
        policy = generate_fanin_policy(topo, chunk_size)
    elif policy_type == "chain":
        policy = generate_dependency_chain_policy(topo, chunk_size)
    else:
        policy = generate_random_policy(topo, chunk_size)

    return policy, policy_type


def main():
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = []
    attempts = 0
    max_attempts = NUM_SAMPLES * 3

    while len(dataset) < NUM_SAMPLES and attempts < max_attempts:
        attempts += 1

        topo = random_topology()

        chunk_mb = random.choice(CHUNK_MB_OPTIONS)
        chunk_size = chunk_mb * 1024 * 1024

        policy, policy_type = generate_policy(topo, chunk_size)

        try:
            makespan, tx_times = run_simulator(topo, policy)

            sample = {
                "topology": topo,
                "policy": policy,
                "completion_time": makespan,
                "tx_times": tx_times,
                "topology_type": topo.graph.get("topology_type", "unknown"),
                "policy_type": policy_type,
                "num_nodes": topo.number_of_nodes(),
                "num_edges": topo.number_of_edges(),
                "chunk_mb": chunk_mb,
                "chunk_size_bytes": chunk_size,
            }

            dataset.append(sample)

            if len(dataset) % 100 == 0:
                print(f"Generated {len(dataset)}/{NUM_SAMPLES}")

        except Exception as e:
            print(f"Skipped sample due to error: {e}")

    torch.save(dataset, SAVE_PATH)

    print(f"\nSaved {len(dataset)} samples to {SAVE_PATH}")
    print(f"Attempts: {attempts}")

if __name__ == "__main__":
    main()
