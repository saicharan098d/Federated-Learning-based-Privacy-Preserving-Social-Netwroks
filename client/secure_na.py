"""
client/secure_na.py
───────────────────
SecureNA: Secure Node Augmentation with Adaptive Local Differential Privacy.

Per-node privacy budget based on structural sensitivity (degree).
High-degree (more exposed) nodes get lower epsilon (more noise).
Overlap node embeddings are aggregated securely at the server.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
#  Sensitivity computation
# ─────────────────────────────────────────────────────────────

def compute_degree_sensitivity(edges, num_nodes) -> np.ndarray:
    """
    Compute normalised degree centrality for each node.
    Returns array of shape (num_nodes,) with values in [0, 1].
    """
    deg = np.zeros(num_nodes, dtype=np.float32)
    for (u, v) in edges:
        deg[u] += 1
        deg[v] += 1
    max_deg = deg.max() if deg.max() > 0 else 1.0
    return deg / max_deg


def sensitivity_to_epsilon(sensitivity: np.ndarray,
                           eps_min: float = 0.5,
                           eps_max: float = 2.0) -> np.ndarray:
    """
    Map sensitivity scores to per-node epsilon values.

    High sensitivity (score → 1)  → small ε  → more noise  (private)
    Low  sensitivity (score → 0)  → large ε  → less noise  (utility)

    ε(u) = eps_max − sensitivity(u) × (eps_max − eps_min)
    """
    eps = eps_max - sensitivity * (eps_max - eps_min)
    return np.clip(eps, eps_min, eps_max)


# ─────────────────────────────────────────────────────────────
#  Laplace mechanism
# ─────────────────────────────────────────────────────────────

def add_laplace_noise(embeddings: np.ndarray,
                      eps_per_node: np.ndarray,
                      clip: float = 1.0,
                      rng: np.random.Generator = None) -> np.ndarray:
    """
    Add per-node Laplace noise to embedding matrix.

    Args:
        embeddings   : (N, d) float array
        eps_per_node : (N,) per-node epsilon values
        clip         : L∞ clip bound for sensitivity
        rng          : numpy Generator

    Returns:
        noisy_embeddings: (N, d) array
    """
    if rng is None:
        rng = np.random.default_rng()

    # Clip embeddings to bound L1 sensitivity
    clipped = np.clip(embeddings, -clip, clip)

    noisy = clipped.copy()
    scale = clip / np.maximum(eps_per_node, 1e-8)   # Laplace scale = Δf/ε

    for i in range(len(eps_per_node)):
        noise = rng.laplace(0.0, scale[i], clipped.shape[1])
        noisy[i] += noise.astype(np.float32)

    return noisy


# ─────────────────────────────────────────────────────────────
#  Client-side SecureNA module
# ─────────────────────────────────────────────────────────────

class ClientNAEncoder:
    """
    Applies adaptive LDP to node embeddings.
    Packages noisy overlap embeddings for secure aggregation at server.
    """

    def __init__(self, cid: int, eps_min: float = 0.5, eps_max: float = 2.0,
                 clip: float = 1.0, seed: int = None):
        self.cid     = cid
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.clip    = clip
        self.rng     = np.random.default_rng(seed if seed is not None else cid * 7919)

    def encode(self, embeddings: np.ndarray, edges: list, num_nodes: int,
               overlap_local: set, global_nodes: list) -> dict:
        """
        Perturb embeddings with adaptive LDP and package overlap embeddings.

        Returns dict with:
            noisy_all         : (N, d) noisy embeddings for all nodes
            overlap_global_ids: list of global node IDs that are overlap nodes
            overlap_noisy_emb : (|overlap|, d) noisy embeddings for overlap nodes
            eps_per_node      : (N,) epsilon used per node
        """
        sensitivity = compute_degree_sensitivity(edges, num_nodes)
        eps_per_node = sensitivity_to_epsilon(sensitivity, self.eps_min, self.eps_max)
        noisy_all = add_laplace_noise(embeddings, eps_per_node, self.clip, self.rng)

        # Extract overlap node embeddings
        overlap_list = sorted(overlap_local)
        overlap_global_ids = [global_nodes[i] if i < len(global_nodes) else i
                               for i in overlap_list]
        overlap_noisy = noisy_all[overlap_list] if overlap_list else np.array([])

        return {
            "cid":               self.cid,
            "noisy_all":         noisy_all,
            "overlap_global_ids": overlap_global_ids,
            "overlap_noisy_emb": overlap_noisy,
            "eps_per_node":      eps_per_node,
            "mean_eps":          float(eps_per_node.mean()),
            "sensitivity":       sensitivity,
        }


# ─────────────────────────────────────────────────────────────
#  Server-side overlap aggregation
# ─────────────────────────────────────────────────────────────

class ServerNAAggregator:
    """
    Securely aggregates noisy overlap embeddings from multiple clients.
    Each overlap node's embedding is the average of all clients' noisy versions.
    (Since each is already LDP-perturbed, averaging is privacy-safe.)
    """

    def merge(self, na_packs: list) -> dict:
        """
        na_packs: list of dicts from ClientNAEncoder.encode()

        Returns: {global_node_id: averaged_noisy_embedding}
        """
        sums   = {}
        counts = {}

        for pack in na_packs:
            for gid, emb in zip(pack["overlap_global_ids"], pack["overlap_noisy_emb"]):
                if gid not in sums:
                    sums[gid]   = np.zeros_like(emb)
                    counts[gid] = 0
                sums[gid]   += emb
                counts[gid] += 1

        merged = {gid: sums[gid] / counts[gid] for gid in sums}
        return merged


# ─────────────────────────────────────────────────────────────
#  Privacy accounting (simple composition)
# ─────────────────────────────────────────────────────────────

class PrivacyAccountant:
    """
    Tracks cumulative privacy expenditure across federated rounds.
    Uses basic (additive) composition.
    """

    def __init__(self):
        self.rounds = 0
        self.cumulative_eps = 0.0
        self.history = []    # list of {"round": r, "mean_eps": e, "cumulative_eps": c}

    def update(self, round_num: int, eps_values: list):
        """eps_values: list of mean eps from each participating client."""
        mean_eps = float(np.mean(eps_values)) if eps_values else 0.0
        self.cumulative_eps += mean_eps
        self.rounds += 1
        self.history.append({
            "round":          round_num,
            "mean_eps":       mean_eps,
            "cumulative_eps": self.cumulative_eps,
        })

    def summary(self) -> dict:
        return {
            "rounds":          self.rounds,
            "cumulative_eps":  self.cumulative_eps,
            "avg_eps_per_rnd": self.cumulative_eps / max(self.rounds, 1),
        }
