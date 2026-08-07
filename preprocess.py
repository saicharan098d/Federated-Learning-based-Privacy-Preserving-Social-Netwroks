"""
preprocess.py
─────────────
Step 1 of the FL-PPSN pipeline.

Downloads (if needed) and preprocesses the Facebook Page-Page Network
from SNAP, then splits into federated client subgraphs.

Usage:
    python preprocess.py                    # expects data/ folder with CSV/JSON files
    python preprocess.py --synthetic        # generates a synthetic graph (no download needed)
    python preprocess.py --clients 10       # number of federated clients

Output:
    data/client_0_data.pkl  ...  data/client_N_data.pkl
    data/global_graph.pkl
    data/label_encoder.pkl
"""

import os
import sys
import json
import pickle
import argparse
import random
import logging
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Argument parser
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default="data",  help="Folder with SNAP files")
    parser.add_argument("--clients",   type=int, default=5, help="Number of FL clients")
    parser.add_argument("--overlap",   type=float, default=0.1, help="Overlap fraction 0-1")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic graph instead of SNAP")
    parser.add_argument("--seed",      type=int, default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
#  Loaders
# ─────────────────────────────────────────────────────────────

def load_facebook_snap(data_dir):
    """
    Load the Facebook Page-Page Network from SNAP.
    Expected files in data_dir/:
        musae_facebook_edges.csv     - columns: id_1, id_2
        musae_facebook_features.json - dict {node_id: [feature list]}
        musae_facebook_target.csv    - columns: id, page_type
    """
    edges_path    = os.path.join(data_dir, "musae_facebook_edges.csv")
    features_path = os.path.join(data_dir, "musae_facebook_features.json")
    target_path   = os.path.join(data_dir, "musae_facebook_target.csv")

    missing = [p for p in [edges_path, features_path, target_path] if not os.path.exists(p)]
    if missing:
        print(f"\n⚠️  Missing dataset files: {missing}")
        print("   Download from: https://snap.stanford.edu/data/facebook-large-page-page-network.html")
        print("   Place in:", data_dir)
        print("   Falling back to SYNTHETIC graph...\n")
        return None, None, None

    print("📂 Loading Facebook Page-Page Network (SNAP)...")
    edges_df = pd.read_csv(edges_path)
    with open(features_path) as f:
        features_raw = json.load(f)
    target_df = pd.read_csv(target_path)

    # Build graph
    G = nx.from_pandas_edgelist(edges_df, "id_1", "id_2")
    print(f"   Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

    # Features — each node has a sparse binary feature vector
    # Determine max feature dimension
    all_nodes = sorted(G.nodes())
    all_feat_ids = set()
    for v in features_raw.values():
        all_feat_ids.update(v)
    feat_dim = max(all_feat_ids) + 1 if all_feat_ids else 128

    X = np.zeros((len(all_nodes), feat_dim), dtype=np.float32)
    node2idx = {n: i for i, n in enumerate(all_nodes)}
    for node_str, feat_list in features_raw.items():
        nid = int(node_str)
        if nid in node2idx:
            for fid in feat_list:
                if fid < feat_dim:
                    X[node2idx[nid], fid] = 1.0

    # Labels
    le = LabelEncoder()
    label_dict = dict(zip(target_df["id"], target_df["page_type"]))
    labels_raw = [label_dict.get(n, "unknown") for n in all_nodes]
    y = le.fit_transform(labels_raw)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # Re-index graph nodes 0..N-1
    G_reindexed = nx.relabel_nodes(G, node2idx)

    print(f"   Feature dim: {feat_dim} | Classes: {len(le.classes_)}")
    return G_reindexed, X, y, le


def load_synthetic_graph(num_nodes=1000, num_classes=4, feat_dim=64, seed=42):
    """
    Synthetic Barabasi-Albert graph for testing without downloading SNAP.
    """
    print("🔧 Generating synthetic social graph (Barabasi-Albert)...")
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(num_nodes, m=3, seed=seed)

    # Community-based labels
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        y = np.zeros(num_nodes, dtype=int)
        for cid, comm in enumerate(communities):
            for node in comm:
                y[node] = cid % num_classes
    except ImportError:
        logger.warning("Community detection unavailable; using random labels")
        y = rng.integers(0, num_classes, size=num_nodes)

    # Features correlated with labels
    X = np.zeros((num_nodes, feat_dim), dtype=np.float32)
    for node in range(num_nodes):
        class_offset = y[node] * (feat_dim // num_classes)
        X[node, class_offset:class_offset + feat_dim // num_classes] = 1.0
        X[node] += rng.normal(0, 0.3, feat_dim).astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    le = LabelEncoder()
    le.fit([str(c) for c in range(num_classes)])

    print(f"   Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | Classes: {num_classes}")
    return G, X, y, le


# ─────────────────────────────────────────────────────────────
#  Federated partitioner
# ─────────────────────────────────────────────────────────────

def partition_federated(G, X, y, num_clients=5, overlap_rate=0.1, seed=42):
    """
    Partition the global graph into client subgraphs.
    Uses community detection for realistic non-IID split.
    Injects overlap_rate fraction of shared nodes across clients.
    """
    random.seed(seed)
    np.random.seed(seed)

    nodes = list(G.nodes())
    N = len(nodes)

    # Community-based partition (non-IID)
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        communities = sorted(communities, key=len, reverse=True)
        client_nodes = defaultdict(set)
        for i, comm in enumerate(communities):
            cid = i % num_clients
            client_nodes[cid].update(comm)
        print(f"   Community partition: {len(communities)} communities → {num_clients} clients")
    except ImportError:
        logger.warning("Community detection unavailable; using random partition")
        # Fallback: random partition
        random.shuffle(nodes)
        client_nodes = defaultdict(set)
        chunk = N // num_clients
        for cid in range(num_clients):
            start = cid * chunk
            end = start + chunk if cid < num_clients - 1 else N
            client_nodes[cid].update(nodes[start:end])
        print("   Random partition (community detection failed)")

    # Inject overlapping nodes
    n_overlap = int(N * overlap_rate)
    overlap_pool = random.sample(nodes, min(n_overlap, N))
    overlap_set = set(overlap_pool)

    for node in overlap_pool:
        # Find owner
        owner = next((cid for cid, ns in client_nodes.items() if node in ns), None)
        if owner is None:
            continue
        others = [c for c in range(num_clients) if c != owner]
        if others:
            target = random.choice(others)
            client_nodes[target].add(node)

    # Build per-client data dicts
    client_data_list = []
    for cid in range(num_clients):
        cnodes = sorted(client_nodes[cid])
        subG = G.subgraph(cnodes).copy()

        # Local index mapping
        local2global = {i: n for i, n in enumerate(cnodes)}
        global2local = {n: i for i, n in local2global.items()}

        # Remap edges to local indices
        edges = [(global2local[u], global2local[v]) for u, v in subG.edges()
                 if u in global2local and v in global2local]

        X_c = X[cnodes]
        y_c = y[cnodes]

        # Train/val/test masks (60/20/20)
        nc = len(cnodes)
        perm = np.random.permutation(nc)
        n_train = int(0.6 * nc)
        n_val   = int(0.2 * nc)
        train_mask = np.zeros(nc, dtype=bool)
        val_mask   = np.zeros(nc, dtype=bool)
        test_mask  = np.zeros(nc, dtype=bool)
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_train+n_val]] = True
        test_mask[perm[n_train+n_val:]] = True

        overlap_local = {global2local[n] for n in cnodes if n in overlap_set}

        client_data = {
            "cid": cid,
            "nodes_global": cnodes,
            "num_nodes": nc,
            "edges": edges,
            "X": X_c,
            "y": y_c,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "overlap_local": overlap_local,
            "local2global": local2global,
            "global2local": global2local,
            "in_dim": X_c.shape[1],
            "num_classes": int(y.max()) + 1,
        }
        client_data_list.append(client_data)

    return client_data_list, overlap_set


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    print("\n" + "═"*55)
    print("  FL-PPSN: Data Preprocessing")
    print("═"*55)

    # Load data
    if args.synthetic:
        G, X, y, le = load_synthetic_graph(seed=args.seed)
    else:
        result = load_facebook_snap(args.data_dir)
        if result[0] is None:
            G, X, y, le = load_synthetic_graph(seed=args.seed)
        else:
            G, X, y, le = result

    # Partition
    print(f"\n🔀 Partitioning into {args.clients} federated clients (overlap={args.overlap*100:.0f}%)...")
    client_data_list, overlap_set = partition_federated(
        G, X, y,
        num_clients=args.clients,
        overlap_rate=args.overlap,
        seed=args.seed,
    )

    # Print stats
    print(f"\n{'─'*55}")
    print(f"  {'Client':<10} {'Nodes':>8} {'Edges':>8} {'Overlap':>8} {'Train':>8} {'Test':>8}")
    print(f"{'─'*55}")
    for d in client_data_list:
        print(f"  Client {d['cid']:<4} {d['num_nodes']:>8} {len(d['edges']):>8} "
              f"{len(d['overlap_local']):>8} {d['train_mask'].sum():>8} {d['test_mask'].sum():>8}")
    print(f"{'─'*55}")
    print(f"  Total overlap nodes: {len(overlap_set)}")

    # Save client data
    for d in client_data_list:
        path = os.path.join(args.data_dir, f"client_{d['cid']}_data.pkl")
        with open(path, "wb") as f:
            pickle.dump(d, f)

    # Save global graph summary
    global_summary = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_classes": int(y.max()) + 1,
        "in_dim": X.shape[1],
        "num_clients": args.clients,
        "label_classes": le.classes_.tolist() if hasattr(le, "classes_") else [],
    }
    with open(os.path.join(args.data_dir, "global_summary.json"), "w") as f:
        json.dump(global_summary, f, indent=2)

    print(f"\n✅ Saved {args.clients} client datasets → {args.data_dir}/")
    print(f"   Global summary → {args.data_dir}/global_summary.json")
    print("\n   Next step: python train_federated.py\n")


if __name__ == "__main__":
    main()
