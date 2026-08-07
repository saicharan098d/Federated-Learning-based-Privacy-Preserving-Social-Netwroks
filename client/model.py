"""
client/model.py
───────────────
Local GNN models for each federated client.
Implements GCN, GAT, and R-GCN in pure NumPy (no PyTorch required).
Each model has get_weights() / set_weights() for federated aggregation.
"""

import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple, Optional, Any


# ─── Activations ──────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation."""
    return np.maximum(0.0, x)

def leaky_relu(x: np.ndarray) -> np.ndarray:
    """Leaky ReLU activation."""
    return np.where(x >= 0, x, 0.2 * x)

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation (numerically stable)."""
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-10)

def dropout(x: np.ndarray, rate: float, training: bool, rng: np.random.Generator) -> np.ndarray:
    """Dropout layer."""
    if not training or rate == 0:
        return x
    mask = (rng.random(x.shape) > rate) / (1 - rate)
    return x * mask


# ─── Xavier init ──────────────────────────────────────────────

def xavier(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Xavier/Glorot initialization."""
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-lim, lim, (fan_in, fan_out)).astype(np.float32)


# ─── Normalised adjacency ─────────────────────────────────────

def norm_adj(edges: List[Tuple[int, int]], n: int) -> sparse.csr_matrix:
    """Symmetric normalised adjacency with self-loops (sparse CSR)."""
    if not edges:
        return sparse.eye(n, format="csr", dtype=np.float32)
    rows, cols = zip(*edges)
    r = list(rows) + list(cols) + list(range(n))
    c = list(cols) + list(rows) + list(range(n))
    A = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n), dtype=np.float32)
    A = (A > 0).astype(np.float32)
    deg = np.array(A.sum(1)).flatten()
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
    D = sparse.diags(d_inv_sqrt)
    return D @ A @ D


# ═══════════════════════════════════════════════════════════════
#  GCN
# ═══════════════════════════════════════════════════════════════

class GCN:
    """2-layer Graph Convolutional Network (Kipf & Welling, 2017)."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.5, seed: int = 42) -> None:
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.dropout: float = dropout
        self.W0: np.ndarray = xavier(in_dim, hidden_dim, self.rng)
        self.W1: np.ndarray = xavier(hidden_dim, num_classes, self.rng)
        self._A: Optional[sparse.csr_matrix] = None

    def _build_A(self, edges: List[Tuple[int, int]], n: int) -> None:
        """Build normalized adjacency matrix."""
        self._A = norm_adj(edges, n)

    def forward(self, X: np.ndarray, edges: List[Tuple[int, int]], n: int, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through the network."""
        if self._A is None or self._A.shape[0] != n:
            self._build_A(edges, n)
        A = self._A
        h = dropout(X, self.dropout, training, self.rng)
        h1 = relu(A @ h @ self.W0)                  # (N, hidden)
        h1 = dropout(h1, self.dropout, training, self.rng)
        logits = A @ h1 @ self.W1                    # (N, C)
        return softmax(logits), h1

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {"W0": self.W0.copy(), "W1": self.W1.copy()}

    def set_weights(self, d):
        self.W0 = d["W0"].copy()
        self.W1 = d["W1"].copy()
        self._A = None   # reset cached adjacency


# ═══════════════════════════════════════════════════════════════
#  GAT
# ═══════════════════════════════════════════════════════════════

class GATLayer:
    """Single GAT attention head layer."""

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.5, rng=None):
        self.heads    = heads
        self.head_dim = out_dim // heads
        self.dropout  = dropout
        self.rng      = rng or np.random.default_rng()
        self.W  = [xavier(in_dim, self.head_dim, self.rng) for _ in range(heads)]
        self.a1 = [xavier(self.head_dim, 1, self.rng) for _ in range(heads)]
        self.a2 = [xavier(self.head_dim, 1, self.rng) for _ in range(heads)]

    def forward(self, X, edges, n, training=True):
        outs = []
        for h in range(self.heads):
            Wh = X @ self.W[h]                       # (N, head_dim)
            if not edges:
                outs.append(Wh)
                continue
            src, dst = zip(*edges)
            src, dst = np.array(src), np.array(dst)
            e = leaky_relu(Wh[src] @ self.a1[h] + Wh[dst] @ self.a2[h]).flatten()
            # Sparse softmax over neighbours
            attn = np.full(n, -1e9, dtype=np.float32)
            rows = [np.where(np.array(src) == i)[0] for i in range(n)]
            H_h  = np.zeros_like(Wh)
            for i in range(n):
                nb_idx = np.where(np.array(src) == i)[0]
                nb     = dst[nb_idx] if len(nb_idx) else np.array([i])
                vals   = e[nb_idx]  if len(nb_idx) else np.array([0.0])
                vals   = vals - vals.max()
                w      = np.exp(vals) / (np.exp(vals).sum() + 1e-10)
                H_h[i] = (w[:, None] * Wh[nb]).sum(0)
            outs.append(H_h)
        return np.concatenate(outs, axis=1)

    def get_weights(self):
        return {"W": [w.copy() for w in self.W],
                "a1": [a.copy() for a in self.a1],
                "a2": [a.copy() for a in self.a2]}

    def set_weights(self, d):
        self.W  = [w.copy() for w in d["W"]]
        self.a1 = [a.copy() for a in d["a1"]]
        self.a2 = [a.copy() for a in d["a2"]]


class GAT:
    """2-layer Graph Attention Network."""

    def __init__(self, in_dim, hidden_dim, num_classes, heads=4, dropout=0.5, seed=42):
        self.rng     = np.random.default_rng(seed)
        self.dropout = dropout
        self.layer1  = GATLayer(in_dim, hidden_dim, heads, dropout, self.rng)
        self.layer2  = GATLayer(hidden_dim, num_classes, 1, dropout, self.rng)
        self.out_W   = xavier(num_classes, num_classes, self.rng)

    def forward(self, X, edges, n, training=True):
        h1  = relu(self.layer1.forward(X, edges, n, training))
        h2  = self.layer2.forward(h1, edges, n, training)
        return softmax(h2), h1

    def get_weights(self):
        return {"layer1": self.layer1.get_weights(),
                "layer2": self.layer2.get_weights(),
                "out_W":  self.out_W.copy()}

    def set_weights(self, d):
        self.layer1.set_weights(d["layer1"])
        self.layer2.set_weights(d["layer2"])
        self.out_W = d["out_W"].copy()


# ═══════════════════════════════════════════════════════════════
#  Model factory
# ═══════════════════════════════════════════════════════════════

def build_model(model_type, in_dim, hidden_dim, num_classes, dropout=0.5, heads=4, seed=42):
    if model_type == "gcn":
        return GCN(in_dim, hidden_dim, num_classes, dropout, seed)
    elif model_type == "gat":
        return GAT(in_dim, hidden_dim, num_classes, heads, dropout, seed)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
