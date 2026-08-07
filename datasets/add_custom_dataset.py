"""
datasets/add_custom_dataset.py
────────────────────────────────
Add ANY custom social network dataset to FL-PPSN.

Supports three input formats:
  Format A: CSV edges + CSV/JSON features + CSV labels
  Format B: Single adjacency matrix (CSV or NumPy .npy)
  Format C: GraphML / GML / GEXF files (via NetworkX)

Usage:
    # Format A (most common):
    python datasets/add_custom_dataset.py \\
        --name mynet \\
        --edges path/to/edges.csv \\
        --features path/to/features.csv \\
        --labels path/to/labels.csv \\
        --label-col category \\
        --clients 5

    # Format C (GraphML):
    python datasets/add_custom_dataset.py \\
        --name mynet \\
        --graphml path/to/graph.graphml \\
        --label-attr community \\
        --clients 5

    # Check format requirements:
    python datasets/add_custom_dataset.py --show-formats
"""

import os
import sys
import json
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configure logging
logger = logging.getLogger(__name__)
from collections import defaultdict
import random

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# ══════════════════════════════════════════════════════════════
#  Format A: edges CSV + features CSV/JSON + labels CSV
# ══════════════════════════════════════════════════════════════

def load_format_a(edges_path, features_path, labels_path, label_col=None,
                  node_id_col=None):
    """
    Load dataset from separate CSV/JSON files.

    edges_path   : CSV with two columns (source_node, target_node)
    features_path: CSV (node_id, f1, f2, ...) OR JSON {node_id: [feat_ids]}
    labels_path  : CSV (node_id, label_column)
    """
    print("  📂 Loading Format A (CSV/JSON)...")

    # ── Edges ────────────────────────────────────────────────
    edges_df = pd.read_csv(edges_path)
    c1, c2 = edges_df.columns[0], edges_df.columns[1]
    G = nx.from_pandas_edgelist(edges_df, c1, c2)
    print(f"     Edges loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    all_nodes = sorted(G.nodes(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    node2idx  = {n: i for i, n in enumerate(all_nodes)}
    N = len(all_nodes)

    # ── Features ─────────────────────────────────────────────
    ext = os.path.splitext(features_path)[1].lower()
    if ext == ".json":
        with open(features_path) as f:
            feat_raw = json.load(f)
        # Sparse binary features
        all_fids = set()
        for v in feat_raw.values():
            all_fids.update(v if isinstance(v, list) else [])
        feat_dim = (max(all_fids) + 1) if all_fids else 64
        feat_dim = min(feat_dim, 1024)
        X = np.zeros((N, feat_dim), dtype=np.float32)
        for nid_str, flist in feat_raw.items():
            nid = int(nid_str) if str(nid_str).isdigit() else nid_str
            if nid in node2idx:
                for fid in (flist if isinstance(flist, list) else []):
                    if fid < feat_dim:
                        X[node2idx[nid], fid] = 1.0
    else:
        feat_df = pd.read_csv(features_path)
        id_col_f = node_id_col or feat_df.columns[0]
        feat_df  = feat_df.set_index(id_col_f)
        feat_cols = feat_df.columns.tolist()
        feat_dim  = len(feat_cols)
        X = np.zeros((N, feat_dim), dtype=np.float32)
        for node, idx in node2idx.items():
            if node in feat_df.index:
                X[idx] = feat_df.loc[node, feat_cols].values.astype(np.float32)

    print(f"     Features: {N} × {feat_dim}")

    # ── Labels ───────────────────────────────────────────────
    labels_df = pd.read_csv(labels_path)
    id_col_l  = node_id_col or labels_df.columns[0]
    if label_col is None:
        label_col = [c for c in labels_df.columns if c != id_col_l][0]
    label_map = dict(zip(labels_df[id_col_l], labels_df[label_col]))
    y_raw = [label_map.get(n, "unknown") for n in all_nodes]
    le    = LabelEncoder()
    y     = le.fit_transform(y_raw).astype(np.int32)
    print(f"     Labels: {len(le.classes_)} classes → {le.classes_.tolist()[:10]}")

    # Normalise
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    G_reindexed = nx.relabel_nodes(G, node2idx)
    return G_reindexed, X, y, le


# ══════════════════════════════════════════════════════════════
#  Format B: adjacency matrix
# ══════════════════════════════════════════════════════════════

def load_format_b(adj_path, features_path=None, labels_path=None,
                  label_col=None):
    """
    Load from adjacency matrix (CSV or .npy).
    """
    print("  📂 Loading Format B (Adjacency Matrix)...")
    ext = os.path.splitext(adj_path)[1].lower()
    if ext == ".npy":
        A = np.load(adj_path)
    else:
        A = pd.read_csv(adj_path, index_col=0).values

    N = A.shape[0]
    G = nx.from_numpy_array(A)
    print(f"     Graph: {N} nodes, {G.number_of_edges()} edges")

    if features_path:
        X = pd.read_csv(features_path, index_col=0).values.astype(np.float32)
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    else:
        # Degree-based features
        deg = np.array([G.degree(n) for n in range(N)], dtype=np.float32)
        X   = deg.reshape(-1, 1)

    if labels_path:
        ldf   = pd.read_csv(labels_path)
        lc    = label_col or ldf.columns[-1]
        y_raw = ldf[lc].values
        le    = LabelEncoder()
        y     = le.fit_transform(y_raw).astype(np.int32)
    else:
        # Community detection as pseudo-labels
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G))
        y = np.zeros(N, dtype=np.int32)
        for cid, comm in enumerate(comms):
            for n in comm: y[n] = cid
        le = LabelEncoder()
        le.fit([str(c) for c in np.unique(y)])

    print(f"     Features: {X.shape} | Classes: {len(np.unique(y))}")
    return G, X, y, le


# ══════════════════════════════════════════════════════════════
#  Format C: GraphML / GML / GEXF
# ══════════════════════════════════════════════════════════════

def load_format_c(graph_path, label_attr=None, feature_attrs=None):
    """
    Load from GraphML, GML, or GEXF file.
    Node attributes become features; label_attr becomes target.
    """
    print("  📂 Loading Format C (GraphML/GML/GEXF)...")
    ext = os.path.splitext(graph_path)[1].lower()
    if ext == ".graphml":
        G_raw = nx.read_graphml(graph_path)
    elif ext == ".gml":
        G_raw = nx.read_gml(graph_path)
    elif ext in (".gexf", ".gexf.gz"):
        G_raw = nx.read_gexf(graph_path)
    else:
        raise ValueError(f"Unsupported graph format: {ext}")

    G_raw = nx.convert_node_labels_to_integers(G_raw)
    G     = G_raw.to_undirected()
    N     = G.number_of_nodes()
    print(f"     Graph: {N} nodes, {G.number_of_edges()} edges")

    # Extract node attributes
    node_data = [G.nodes[i] for i in range(N)]
    all_attrs  = set()
    for d in node_data: all_attrs.update(d.keys())

    if label_attr and label_attr in all_attrs:
        all_attrs.discard(label_attr)

    feat_attrs = feature_attrs or sorted(all_attrs)

    if feat_attrs:
        rows = []
        for i in range(N):
            row = []
            for a in feat_attrs:
                v = G.nodes[i].get(a, 0)
                try:
                    row.append(float(v))
                except (ValueError, TypeError):
                    row.append(0.0)
            rows.append(row)
        X = np.array(rows, dtype=np.float32)
        if X.shape[1] > 0:
            scaler = StandardScaler()
            X = scaler.fit_transform(X).astype(np.float32)
    else:
        deg = np.array([G.degree(i) for i in range(N)], dtype=np.float32).reshape(-1,1)
        X   = deg

    if label_attr:
        y_raw = [str(G.nodes[i].get(label_attr, "unknown")) for i in range(N)]
    else:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G))
        y_raw = ["unknown"] * N
        for cid, comm in enumerate(comms):
            for n in comm: y_raw[n] = str(cid)

    le = LabelEncoder()
    y  = le.fit_transform(y_raw).astype(np.int32)
    print(f"     Features: {X.shape} | Classes: {len(le.classes_)}")
    return G, X, y, le


# ══════════════════════════════════════════════════════════════
#  Partition and save (same as download_datasets.py)
# ══════════════════════════════════════════════════════════════

def partition_and_save(G, X, y, le, name, num_clients=5,
                       overlap=0.1, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    nodes   = list(G.nodes())
    N       = len(nodes)
    out_dir = os.path.join(DATA_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = sorted(greedy_modularity_communities(G), key=len, reverse=True)
        client_nodes = defaultdict(set)
        for i, c in enumerate(communities):
            client_nodes[i % num_clients].update(c)
    except ImportError:
        logger.warning("Community detection unavailable; using random partition")
        random.shuffle(nodes)
        client_nodes = defaultdict(set)
        chunk = N // num_clients
        for cid in range(num_clients):
            s = cid * chunk
            e = s + chunk if cid < num_clients - 1 else N
            client_nodes[cid].update(nodes[s:e])

    # Overlap
    n_ov = int(N * overlap)
    ov_p = random.sample(nodes, min(n_ov, N))
    ov_s = set(ov_p)
    for node in ov_p:
        owner = next((c for c, ns in client_nodes.items() if node in ns), None)
        if owner is None: continue
        others = [c for c in range(num_clients) if c != owner]
        if others: client_nodes[random.choice(others)].add(node)

    num_classes = int(y.max()) + 1
    in_dim      = X.shape[1]

    print(f"\n  Saving federated splits:")
    print(f"  {'Client':<10} {'Nodes':>8} {'Edges':>8} {'Overlap':>8} {'Train':>7} {'Test':>7}")
    print(f"  {'─'*50}")

    for cid in range(num_clients):
        cnodes = sorted(client_nodes[cid])
        subG   = G.subgraph(cnodes).copy()
        l2g    = {i: n for i, n in enumerate(cnodes)}
        g2l    = {n: i for i, n in l2g.items()}
        edges  = [(g2l[u], g2l[v]) for u, v in subG.edges()
                  if u in g2l and v in g2l]
        nc  = len(cnodes)
        perm = np.random.permutation(nc)
        n_tr = int(0.6 * nc); n_val = int(0.2 * nc)
        tr = np.zeros(nc, bool); tr[perm[:n_tr]] = True
        vl = np.zeros(nc, bool); vl[perm[n_tr:n_tr+n_val]] = True
        ts = np.zeros(nc, bool); ts[perm[n_tr+n_val:]] = True
        ov_local = {g2l[n] for n in cnodes if n in ov_s}

        data = {
            "cid": cid, "nodes_global": cnodes, "num_nodes": nc, "edges": edges,
            "X": X[cnodes], "y": y[cnodes],
            "train_mask": tr, "val_mask": vl, "test_mask": ts,
            "overlap_local": ov_local, "local2global": l2g, "global2local": g2l,
            "in_dim": in_dim, "num_classes": num_classes, "dataset": name,
        }
        with open(os.path.join(out_dir, f"client_{cid}_data.pkl"), "wb") as f:
            pickle.dump(data, f)

        print(f"  Client {cid:<5} {nc:>8} {len(edges):>8} {len(ov_local):>8} {tr.sum():>7} {ts.sum():>7}")

    summary = {
        "dataset": name, "num_nodes": N,
        "num_edges": G.number_of_edges(), "num_classes": num_classes,
        "in_dim": in_dim, "num_clients": num_clients, "overlap": overlap,
        "label_classes": le.classes_.tolist() if hasattr(le,"classes_") else [],
    }
    with open(os.path.join(out_dir, "global_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ✅ Saved → data/{name}/")
    print(f"  Train with: python train_federated.py --data-dir data/{name} --clients {num_clients}")
    return out_dir


# ══════════════════════════════════════════════════════════════
#  Show format requirements
# ══════════════════════════════════════════════════════════════

def show_formats():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  FL-PPSN: Custom Dataset Format Requirements                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  FORMAT A: CSV/JSON (recommended)                            ║
║  ─────────────────────────────────                           ║
║  edges.csv:                                                  ║
║      src_node, dst_node                                      ║
║      0, 1                                                    ║
║      0, 2                                                    ║
║                                                              ║
║  features.csv:                                               ║
║      node_id, feat_1, feat_2, feat_3, ...                    ║
║      0,       0.5,    1.2,    0.0,   ...                     ║
║                                                              ║
║  features.json (alternative, sparse binary):                 ║
║      {"0": [3, 17, 42], "1": [0, 5, 99], ...}               ║
║                                                              ║
║  labels.csv:                                                 ║
║      node_id, category                                       ║
║      0,       sports                                         ║
║      1,       news                                           ║
║                                                              ║
║  FORMAT B: Adjacency Matrix                                  ║
║  ──────────────────────────────                              ║
║  adj.csv or adj.npy   → N×N adjacency matrix                ║
║  features.csv         → N×D feature matrix (optional)        ║
║  labels.csv           → N×1 label column   (optional)        ║
║                                                              ║
║  FORMAT C: Graph files                                       ║
║  ─────────────────────────────                               ║
║  graph.graphml / graph.gml / graph.gexf                      ║
║  Node attributes = features + label                          ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  Example commands:                                           ║
║                                                              ║
║  python datasets/add_custom_dataset.py \\                    ║
║      --name mydata \\                                        ║
║      --edges mydata/edges.csv \\                             ║
║      --features mydata/features.csv \\                       ║
║      --labels mydata/labels.csv \\                           ║
║      --label-col category \\                                 ║
║      --clients 5                                             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name",          required=False, help="Dataset name (used as folder)")
    p.add_argument("--edges",         help="Path to edges CSV")
    p.add_argument("--features",      help="Path to features CSV or JSON")
    p.add_argument("--labels",        help="Path to labels CSV")
    p.add_argument("--label-col",     help="Label column name in labels CSV")
    p.add_argument("--adj",           help="Path to adjacency matrix (Format B)")
    p.add_argument("--graphml",       help="Path to GraphML/GML/GEXF file (Format C)")
    p.add_argument("--label-attr",    help="Node attribute to use as label (Format C)")
    p.add_argument("--feat-attrs",    nargs="+", help="Node attributes to use as features (Format C)")
    p.add_argument("--clients",       type=int,   default=5)
    p.add_argument("--overlap",       type=float, default=0.1)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--show-formats",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.show_formats:
        show_formats()
        return

    if not args.name:
        print("  ✗ Please provide --name for your dataset")
        show_formats()
        sys.exit(1)

    if args.graphml:
        G, X, y, le = load_format_c(args.graphml, args.label_attr, args.feat_attrs)
    elif args.adj:
        G, X, y, le = load_format_b(args.adj, args.features, args.labels, args.label_col)
    elif args.edges:
        if not args.features or not args.labels:
            print("  ✗ Format A requires --edges, --features, and --labels")
            sys.exit(1)
        G, X, y, le = load_format_a(args.edges, args.features, args.labels,
                                     args.label_col)
    else:
        print("  ✗ Provide one of: --edges, --adj, or --graphml")
        show_formats()
        sys.exit(1)

    partition_and_save(G, X, y, le, args.name, args.clients, args.overlap, args.seed)


if __name__ == "__main__":
    main()
