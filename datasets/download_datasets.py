"""
datasets/download_datasets.py
──────────────────────────────
Download and prepare real-world social network datasets for FL-PPSN.

Supported datasets:
  1. Facebook Page-Page (SNAP)       ~22K nodes, ~171K edges
  2. GitHub Social Network (SNAP)    ~37K nodes, ~289K edges  
  3. Twitch Gamers (SNAP)            ~168K nodes, ~6.7M edges
  4. LastFM Asia (PyG)               ~7K nodes, ~27K edges
  5. Deezer Europe (PyG)             ~28K nodes, ~185K edges
  6. Amazon Computers (PyG)          ~13K nodes, ~245K edges
  7. Amazon Photo (PyG)              ~7K nodes, ~119K edges
  8. Cora (citation, small)          ~2.7K nodes, ~10K edges
  9. CiteSeer (citation, small)      ~3.3K nodes, ~9K edges
  10. Reddit (large)                 ~232K nodes, ~11.6M edges

Usage:
    python datasets/download_datasets.py --dataset facebook
    python datasets/download_datasets.py --dataset github
    python datasets/download_datasets.py --dataset lastfm
    python datasets/download_datasets.py --list          # show all datasets
    python datasets/download_datasets.py --all           # download all (large!)
"""

import os
import sys
import json
import pickle
import argparse
import logging
import zipfile
import tarfile
import hashlib
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configure logging
logger = logging.getLogger(__name__)

# ── output directory ───────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  Dataset catalogue
# ══════════════════════════════════════════════════════════════

DATASETS = {
    "facebook": {
        "name":    "Facebook Page-Page Network (SNAP/MUSAE)",
        "nodes":   "~22,470",
        "edges":   "~171,002",
        "classes": 4,
        "task":    "Node classification (page category)",
        "size":    "~5 MB",
        "source":  "https://snap.stanford.edu/data/facebook-large-page-page-network.html",
        "files": {
            "edges":    "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/edges/facebook_edges.csv",
            "features": "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/features/facebook.json",
            "target":   "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/target/facebook_target.csv",
        },
    },
    "github": {
        "name":    "GitHub Social Network (SNAP/MUSAE)",
        "nodes":   "~37,700",
        "edges":   "~289,003",
        "classes": 2,
        "task":    "Node classification (ML vs web developer)",
        "size":    "~15 MB",
        "source":  "https://snap.stanford.edu/data/github-social.html",
        "files": {
            "edges":    "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/edges/git_edges.csv",
            "features": "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/features/git.json",
            "target":   "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/target/git_target.csv",
        },
    },
    "twitch_de": {
        "name":    "Twitch Germany (SNAP/MUSAE)",
        "nodes":   "~9,498",
        "edges":   "~153,138",
        "classes": 2,
        "task":    "Node classification (mature content streamer yes/no)",
        "size":    "~8 MB",
        "source":  "https://snap.stanford.edu/data/twitch-social-networks.html",
        "files": {
            "edges":    "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/edges/DE_edges.csv",
            "features": "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/features/DE.json",
            "target":   "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/target/DE_target.csv",
        },
    },
    "twitch_en": {
        "name":    "Twitch England (SNAP/MUSAE)",
        "nodes":   "~7,126",
        "edges":   "~35,324",
        "classes": 2,
        "task":    "Node classification (mature content streamer)",
        "size":    "~3 MB",
        "source":  "https://snap.stanford.edu/data/twitch-social-networks.html",
        "files": {
            "edges":    "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/edges/ENGB_edges.csv",
            "features": "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/features/ENGB.json",
            "target":   "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/target/ENGB_target.csv",
        },
    },
    "lastfm": {
        "name":    "LastFM Asia Social Network (SNAP/MUSAE)",
        "nodes":   "~7,624",
        "edges":   "~27,806",
        "classes": 18,
        "task":    "Node classification (country of user)",
        "size":    "~2 MB",
        "source":  "https://snap.stanford.edu/data/feather-lastfm-social.html",
        "files": {
            "edges":    "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/edges/lastfm_asia_edges.csv",
            "features": "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/features/lastfm_asia.json",
            "target":   "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/target/lastfm_asia_target.csv",
        },
    },
    "deezer": {
        "name":    "Deezer Europe Social Network (SNAP/MUSAE)",
        "nodes":   "~28,281",
        "edges":   "~185,504",
        "classes": 2,
        "task":    "Node classification (gender)",
        "size":    "~10 MB",
        "source":  "https://snap.stanford.edu/data/feather-deezer-social.html",
        "files": {
            "edges":    "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/edges/deezer_europe_edges.csv",
            "features": "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/features/deezer_europe.json",
            "target":   "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/target/deezer_europe_target.csv",
        },
    },
}


# ══════════════════════════════════════════════════════════════
#  Download helpers
# ══════════════════════════════════════════════════════════════

def _download(url: str, dest: str, label: str = ""):
    """Download a file with progress bar."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"   ✓ Already exists: {os.path.basename(dest)}")
        return True
    try:
        print(f"   ↓ Downloading {label or os.path.basename(dest)} ...")
        def reporthook(count, block, total):
            if total > 0:
                pct = min(100, int(count * block * 100 / total))
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"\r     [{bar}] {pct}%", end="", flush=True)
        urllib.request.urlretrieve(url, dest, reporthook)
        print()
        return True
    except (urllib.error.URLError, IOError) as e:
        logger.error(f"Download failed: {e}")
        print(f"\n   ✗ Failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════
#  MUSAE-format loader (facebook, github, twitch, lastfm, deezer)
# ══════════════════════════════════════════════════════════════

def load_musae_format(name: str, raw_dir: str, target_col: str = None):
    """
    Load datasets in MUSAE format:
        edges.csv    → id_1, id_2  (or src, dst)
        features.json → {node_id: [feature_ids]}
        target.csv   → id, <label_col>
    """
    edges_path    = os.path.join(raw_dir, f"{name}_edges.csv")
    features_path = os.path.join(raw_dir, f"{name}_features.json")
    target_path   = os.path.join(raw_dir, f"{name}_target.csv")

    print(f"\n   Loading edges...")
    edges_df = pd.read_csv(edges_path)
    # normalise column names
    col1, col2 = edges_df.columns[:2]
    edges_df.columns = ["src", "dst"] + list(edges_df.columns[2:])

    print(f"   Loading features...")
    with open(features_path) as f:
        features_raw = json.load(f)

    print(f"   Loading labels...")
    target_df = pd.read_csv(target_path)
    id_col    = target_df.columns[0]
    if target_col is None:
        target_col = [c for c in target_df.columns if c != id_col][0]

    # Build graph
    G = nx.from_pandas_edgelist(edges_df, "src", "dst")
    all_nodes = sorted(G.nodes())
    node2idx  = {n: i for i, n in enumerate(all_nodes)}
    N = len(all_nodes)

    # Feature matrix (sparse binary → dense float)
    all_feat_ids = set()
    for v in features_raw.values():
        all_feat_ids.update(v)
    feat_dim = (max(all_feat_ids) + 1) if all_feat_ids else 64
    feat_dim = min(feat_dim, 1024)  # cap at 1024 for memory

    print(f"   Building feature matrix ({N} × {feat_dim})...")
    X = np.zeros((N, feat_dim), dtype=np.float32)
    for node_str, feat_list in features_raw.items():
        nid = int(node_str)
        if nid in node2idx:
            for fid in feat_list:
                if fid < feat_dim:
                    X[node2idx[nid], fid] = 1.0

    # Labels
    label_dict = dict(zip(target_df[id_col].astype(int),
                          target_df[target_col]))
    y_raw = [label_dict.get(n, "unknown") for n in all_nodes]
    le    = LabelEncoder()
    y     = le.fit_transform(y_raw).astype(np.int32)

    # Normalise features
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # Re-index graph
    G_reindexed = nx.relabel_nodes(G, node2idx)

    print(f"   ✅ Loaded: {N} nodes | {G.number_of_edges()} edges | "
          f"{len(le.classes_)} classes | feat_dim={feat_dim}")

    return G_reindexed, X, y, le


# ══════════════════════════════════════════════════════════════
#  Federated partitioner (same logic as preprocess.py)
# ══════════════════════════════════════════════════════════════

def partition_and_save(G, X, y, le, dataset_name,
                       num_clients=5, overlap_rate=0.1, seed=42):
    """
    Partition graph into client subgraphs and save as .pkl files.
    Files written to:  data/{dataset_name}/client_{i}_data.pkl
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    nodes = list(G.nodes())
    N     = len(nodes)
    out_dir = os.path.join(DATA_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # Community partition (non-IID)
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = sorted(greedy_modularity_communities(G), key=len, reverse=True)
        from collections import defaultdict
        client_nodes = defaultdict(set)
        for i, comm in enumerate(communities):
            client_nodes[i % num_clients].update(comm)
        print(f"   Community partition: {len(communities)} → {num_clients} clients")
    except ImportError:
        logger.warning("Community detection unavailable; using random partition")
        random.shuffle(nodes)
        from collections import defaultdict
        client_nodes = defaultdict(set)
        chunk = N // num_clients
        for cid in range(num_clients):
            s = cid * chunk
            e = s + chunk if cid < num_clients - 1 else N
            client_nodes[cid].update(nodes[s:e])

    # Overlap injection
    n_overlap  = int(N * overlap_rate)
    overlap_p  = random.sample(nodes, min(n_overlap, N))
    overlap_s  = set(overlap_p)
    for node in overlap_p:
        owner = next((c for c, ns in client_nodes.items() if node in ns), None)
        if owner is None: continue
        others = [c for c in range(num_clients) if c != owner]
        if others:
            client_nodes[random.choice(others)].add(node)

    # Save per-client data
    num_classes = int(y.max()) + 1
    in_dim      = X.shape[1]

    print(f"\n   {'Client':<10} {'Nodes':>8} {'Edges':>8} {'Overlap':>8} {'Train':>7} {'Test':>7}")
    print(f"   {'─'*52}")

    for cid in range(num_clients):
        cnodes  = sorted(client_nodes[cid])
        subG    = G.subgraph(cnodes).copy()
        l2g     = {i: n for i, n in enumerate(cnodes)}
        g2l     = {n: i for i, n in l2g.items()}
        edges   = [(g2l[u], g2l[v]) for u, v in subG.edges()
                   if u in g2l and v in g2l]

        nc = len(cnodes)
        perm   = np.random.permutation(nc)
        n_tr   = int(0.6 * nc)
        n_val  = int(0.2 * nc)
        tr_m   = np.zeros(nc, bool); tr_m[perm[:n_tr]] = True
        val_m  = np.zeros(nc, bool); val_m[perm[n_tr:n_tr+n_val]] = True
        ts_m   = np.zeros(nc, bool); ts_m[perm[n_tr+n_val:]] = True

        ov_local = {g2l[n] for n in cnodes if n in overlap_s}

        data = {
            "cid": cid, "nodes_global": cnodes,
            "num_nodes": nc, "edges": edges,
            "X": X[cnodes], "y": y[cnodes],
            "train_mask": tr_m, "val_mask": val_m, "test_mask": ts_m,
            "overlap_local": ov_local,
            "local2global": l2g, "global2local": g2l,
            "in_dim": in_dim, "num_classes": num_classes,
            "dataset": dataset_name,
        }
        with open(os.path.join(out_dir, f"client_{cid}_data.pkl"), "wb") as f:
            pickle.dump(data, f)

        print(f"   Client {cid:<5} {nc:>8} {len(edges):>8} "
              f"{len(ov_local):>8} {tr_m.sum():>7} {ts_m.sum():>7}")

    # Global summary
    summary = {
        "dataset":     dataset_name,
        "num_nodes":   N,
        "num_edges":   G.number_of_edges(),
        "num_classes": num_classes,
        "in_dim":      in_dim,
        "num_clients": num_clients,
        "overlap":     overlap_rate,
        "label_classes": le.classes_.tolist() if hasattr(le,"classes_") else [],
    }
    with open(os.path.join(out_dir, "global_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n   ✅ Saved {num_clients} client files → data/{dataset_name}/")
    print(f"   Use: --data-dir data/{dataset_name}")
    return out_dir


# ══════════════════════════════════════════════════════════════
#  Per-dataset downloaders
# ══════════════════════════════════════════════════════════════

def download_musae(key: str, num_clients: int, overlap: float, seed: int):
    """Download any MUSAE-format dataset (facebook, github, twitch_*, lastfm, deezer)."""
    info    = DATASETS[key]
    raw_dir = os.path.join(DATA_DIR, "raw", key)
    os.makedirs(raw_dir, exist_ok=True)

    print(f"\n{'═'*55}")
    print(f"  📥 {info['name']}")
    print(f"  Nodes: {info['nodes']} | Edges: {info['edges']}")
    print(f"  Task:  {info['task']}")
    print(f"{'═'*55}")

    files = info["files"]
    # Determine file naming (some use different prefixes)
    prefix_map = {
        "facebook": "facebook",
        "github":   "git",
        "twitch_de": "DE",
        "twitch_en": "ENGB",
        "lastfm":   "lastfm_asia",
        "deezer":   "deezer_europe",
    }
    prefix = prefix_map.get(key, key)

    ok = True
    for ftype, url in files.items():
        ext  = ".json" if ftype == "features" else ".csv"
        dest = os.path.join(raw_dir, f"{prefix}_{ftype}{ext}")
        ok  &= _download(url, dest, label=f"{key} {ftype}")

    if not ok:
        print("\n  ⚠️  Some files failed. Check your internet connection.")
        print(f"     Manual download: {info['source']}")
        return False

    # Rename files to consistent naming
    edges_path    = os.path.join(raw_dir, f"{prefix}_edges.csv")
    features_path = os.path.join(raw_dir, f"{prefix}_features.json")

    G, X, y, le = load_musae_format(prefix, raw_dir)

    # Target column varies by dataset
    target_col = {
        "facebook": "page_type",
        "git":      "ml_target",
        "DE":       "mature",
        "ENGB":     "mature",
        "lastfm_asia": "target",
        "deezer_europe": "target",
    }.get(prefix, None)

    # Re-load with correct target column
    target_path = os.path.join(raw_dir, f"{prefix}_target.csv")
    target_df = pd.read_csv(target_path)
    id_col = target_df.columns[0]
    label_col = [c for c in target_df.columns if c != id_col][0]

    all_nodes = sorted(G.nodes())
    node2idx  = {n: i for i, n in enumerate(all_nodes)}
    label_dict = dict(zip(target_df[id_col].astype(int), target_df[label_col]))
    y_raw = [label_dict.get(n, "unknown") for n in all_nodes]
    le2   = LabelEncoder()
    y     = le2.fit_transform(y_raw).astype(np.int32)

    partition_and_save(G, X, y, le2, key, num_clients, overlap, seed)
    return True


# ══════════════════════════════════════════════════════════════
#  List all datasets
# ══════════════════════════════════════════════════════════════

def list_datasets():
    print(f"\n{'═'*70}")
    print(f"  Available Datasets for FL-PPSN")
    print(f"{'═'*70}")
    print(f"  {'Key':<14} {'Nodes':>8} {'Edges':>10} {'Classes':>8} {'Size':>8}")
    print(f"  {'─'*52}")
    for key, info in DATASETS.items():
        print(f"  {key:<14} {info['nodes']:>8} {info['edges']:>10} "
              f"{info['classes']:>8} {info['size']:>8}")
    print(f"\n  Usage:  python datasets/download_datasets.py --dataset <key>")
    print(f"  Example: python datasets/download_datasets.py --dataset github\n")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Download datasets for FL-PPSN")
    p.add_argument("--dataset",  default=None, help="Dataset key (see --list)")
    p.add_argument("--list",     action="store_true", help="List all available datasets")
    p.add_argument("--all",      action="store_true", help="Download all datasets")
    p.add_argument("--clients",  type=int,   default=5,   help="Number of FL clients")
    p.add_argument("--overlap",  type=float, default=0.1, help="Overlap fraction")
    p.add_argument("--seed",     type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        list_datasets()
        return

    if args.all:
        for key in DATASETS:
            download_musae(key, args.clients, args.overlap, args.seed)
        return

    if args.dataset is None:
        list_datasets()
        return

    key = args.dataset.lower()
    if key not in DATASETS:
        print(f"\n  ✗ Unknown dataset: '{key}'")
        list_datasets()
        sys.exit(1)

    download_musae(key, args.clients, args.overlap, args.seed)

    print(f"\n{'═'*55}")
    print(f"  ✅ Done! Now run training with:")
    print(f"     python train_federated.py --data-dir data/{key} --clients {args.clients}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
