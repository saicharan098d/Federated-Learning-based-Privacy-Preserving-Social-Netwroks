"""
server/aggregation.py
─────────────────────
All server-side model aggregation strategies:

  FedAvg       — standard weighted average
  q-FedAvg     — fairness-aware (lifts worst clients)
  KRUM         — Byzantine-robust (rejects outlier updates)
  TrimmedMean  — coordinate-wise trimmed mean
  Median       — coordinate-wise median
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Union


# ─────────────────────────────────────────────────────────────
#  Weight utilities
# ─────────────────────────────────────────────────────────────

def _flat(weights: Dict[str, Any]) -> np.ndarray:
    """Flatten nested weight dict to 1-D array."""
    parts: List[np.ndarray] = []
    for k in sorted(weights):
        v = weights[k]
        if isinstance(v, np.ndarray):
            parts.append(v.flatten())
        elif isinstance(v, list):
            for w in v:
                parts.append(w.flatten())
        elif isinstance(v, dict):
            parts.append(_flat(v))
    return np.concatenate(parts).astype(np.float32) if parts else np.array([], dtype=np.float32)


def _unflat(flat: np.ndarray, template: Dict[str, Any]) -> Dict[str, Any]:
    """Restore weight dict from flat array using a template for shapes."""
    result: Dict[str, Any] = {}
    offset: int = 0
    for k in sorted(template):
        v = template[k]
        if isinstance(v, np.ndarray):
            size = v.size
            result[k] = flat[offset:offset+size].reshape(v.shape).astype(np.float32)
            offset += size
        elif isinstance(v, list):
            result[k] = []
            for w in v:
                size = w.size
                result[k].append(flat[offset:offset+size].reshape(w.shape).astype(np.float32))
                offset += size
        elif isinstance(v, dict):
            sub_flat, sub_size = _unflat_sub(flat[offset:], v)
            result[k] = sub_flat
            offset += sub_size
    return result


def _unflat_sub(flat: np.ndarray, template: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Helper for nested dicts."""
    result = {}
    offset = 0
    for k in sorted(template):
        v = template[k]
        if isinstance(v, np.ndarray):
            size = v.size
            result[k] = flat[offset:offset+size].reshape(v.shape)
            offset += size
        elif isinstance(v, list):
            result[k] = []
            for w in v:
                size = w.size
                result[k].append(flat[offset:offset+size].reshape(w.shape))
                offset += size
    return result, offset


def _zeros_like(template: dict) -> dict:
    if isinstance(template, np.ndarray):
        return np.zeros_like(template)
    result = {}
    for k in template:
        v = template[k]
        if isinstance(v, np.ndarray):
            result[k] = np.zeros_like(v)
        elif isinstance(v, list):
            result[k] = [np.zeros_like(w) for w in v]
        elif isinstance(v, dict):
            result[k] = _zeros_like(v)
    return result


def _add(w1, w2, scale=1.0):
    """w1 + scale * w2"""
    if isinstance(w1, np.ndarray):
        return w1 + scale * w2
    result = {}
    for k in w1:
        if isinstance(w1[k], np.ndarray):
            result[k] = w1[k] + scale * w2.get(k, np.zeros_like(w1[k]))
        elif isinstance(w1[k], list):
            result[k] = [a + scale * b for a, b in zip(w1[k], w2.get(k, [np.zeros_like(x) for x in w1[k]]))]
        elif isinstance(w1[k], dict):
            result[k] = _add(w1[k], w2.get(k, _zeros_like(w1[k])), scale)
        else:
            result[k] = w1[k]
    return result


def clip_update(weights: dict, max_norm: float = 1.0) -> dict:
    """Clip weight update by global L2 norm."""
    flat = _flat(weights)
    norm = np.linalg.norm(flat)
    if norm > max_norm:
        factor = max_norm / norm
        flat   = flat * factor
        return _unflat(flat, weights)
    return weights


# ─────────────────────────────────────────────────────────────
#  Aggregation methods
# ─────────────────────────────────────────────────────────────

def fedavg(updates: list) -> dict:
    """Standard FedAvg: weighted average by number of training nodes."""
    total = sum(u["num_train"] for u in updates)
    if total == 0:
        return updates[0]["weights"]
    result = _zeros_like(updates[0]["weights"])
    for u in updates:
        w = u["num_train"] / total
        result = _add(result, u["weights"], w)
    return result


def qfedavg(updates: list, q: float = 0.3, alpha: float = 1.0, tau: float = 1e-3) -> dict:
    """
    q-FedAvg (Li et al., 2020).
    Upweights clients with higher loss → reduces performance disparity.
    w_c ∝ n_c^α · (L_c + τ)^q
    """
    raw_w = np.array([
        (u["num_train"] ** alpha) * ((u["train_loss"] + tau) ** q)
        for u in updates
    ], dtype=np.float64)
    raw_w = raw_w / (raw_w.sum() + 1e-10)

    result = _zeros_like(updates[0]["weights"])
    for u, w in zip(updates, raw_w):
        result = _add(result, u["weights"], float(w))
    return result


def krum(updates: list, f: int = 1) -> dict:
    """
    KRUM (Blanchard et al., 2017).
    Selects the update closest to its n-f-2 nearest neighbours.
    Tolerates f Byzantine clients.
    """
    n = len(updates)
    if n < 2 * f + 3:
        print(f"    [KRUM] Fallback to FedAvg (n={n} < 2f+3={2*f+3})")
        return fedavg(updates)

    vecs = [_flat(u["weights"]) for u in updates]
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            d = np.sum((vecs[i] - vecs[j]) ** 2)
            dist[i, j] = dist[j, i] = d

    k = n - f - 2
    scores = np.array([np.sort(dist[i])[1:k+1].sum() for i in range(n)])
    best   = int(np.argmin(scores))
    print(f"    [KRUM] Selected client {updates[best]['cid']} (score={scores[best]:.4f})")
    return updates[best]["weights"]


def trimmed_mean(updates: list, trim: float = 0.1) -> dict:
    """Coordinate-wise trimmed mean. Clips top/bottom trim fraction."""
    n   = len(updates)
    mat = np.stack([_flat(u["weights"]) for u in updates])   # (n, d)
    k   = max(1, int(n * trim))
    trimmed = np.sort(mat, axis=0)[k:n-k]
    mean_flat = trimmed.mean(axis=0)
    print(f"    [TrimmedMean] Trimmed {k} from each end (n={n})")
    return _unflat(mean_flat, updates[0]["weights"])


def coordinate_median(updates: list) -> dict:
    """Coordinate-wise median."""
    mat = np.stack([_flat(u["weights"]) for u in updates])
    med = np.median(mat, axis=0)
    return _unflat(med, updates[0]["weights"])


# ─────────────────────────────────────────────────────────────
#  Unified entry point
# ─────────────────────────────────────────────────────────────

METHODS = {
    "fedavg":        fedavg,
    "qfedavg":       qfedavg,
    "krum":          krum,
    "trimmed_mean":  trimmed_mean,
    "median":        coordinate_median,
}


def aggregate(updates: list, method: str = "fedavg", cfg: dict = None) -> dict:
    """
    Unified aggregation interface.

    updates : list of dicts, each with keys:
        cid, weights, num_train, train_loss, val_acc, test_acc
    method  : 'fedavg' | 'qfedavg' | 'krum' | 'trimmed_mean' | 'median'
    cfg     : extra config (q, alpha, f, trim_fraction …)
    """
    cfg = cfg or {}
    print(f"\n  [Aggregation] method={method} | clients={len(updates)}")

    if method == "fedavg":
        return fedavg(updates)
    elif method == "qfedavg":
        return qfedavg(updates,
                       q=cfg.get("q", 0.3),
                       alpha=cfg.get("alpha", 1.0),
                       tau=cfg.get("tau", 1e-3))
    elif method == "krum":
        return krum(updates, f=cfg.get("f", 1))
    elif method == "trimmed_mean":
        return trimmed_mean(updates, trim=cfg.get("trim", 0.1))
    elif method == "median":
        return coordinate_median(updates)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ─────────────────────────────────────────────────────────────
#  Fairness metrics
# ─────────────────────────────────────────────────────────────

def fairness_metrics(updates: list) -> dict:
    val_accs = [u["val_acc"] for u in updates]
    return {
        "mean_val_acc":  float(np.mean(val_accs)),
        "worst_val_acc": float(np.min(val_accs)),
        "best_val_acc":  float(np.max(val_accs)),
        "std_val_acc":   float(np.std(val_accs)),
        "gap":           float(np.max(val_accs) - np.min(val_accs)),
    }
