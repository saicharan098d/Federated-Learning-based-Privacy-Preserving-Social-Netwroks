"""
client/secure_sa.py
───────────────────
SecureSA: Secure Social Subgraph Aggregation via Bloom filters.

Client side: encode adjacency into a salted bit-vector.
Server side: bitwise-OR merge to get global adjacency approximation.
Raw edges NEVER leave the client.
"""

import os
import hashlib
import struct
import numpy as np


def _hash(salt: bytes, key: str, seed: int, m: int) -> int:
    """Deterministic hash with salt + seed mixing."""
    h = hashlib.sha256(struct.pack(">I", seed) + salt + key.encode()).digest()
    return struct.unpack(">Q", h[:8])[0] % m


class BloomSketch:
    """
    Salted Bloom filter that encodes a client's local adjacency.

    Parameters
    ----------
    m    : bit-vector size (default 8192 → ~1KB per client)
    k    : number of hash functions (default 4)
    salt : client-specific random salt so server cannot reverse hashes
    """

    def __init__(self, m: int = 8192, k: int = 4, salt: bytes = None):
        self.m    = m
        self.k    = k
        self.salt = salt or os.urandom(16)
        self.bits = np.zeros(m, dtype=np.uint8)
        self._insertions = 0

    def _positions(self, u: int, v: int):
        key = f"{min(u,v)}-{max(u,v)}"
        return [_hash(self.salt, key, i, self.m) for i in range(self.k)]

    def add_edge(self, u: int, v: int):
        for p in self._positions(u, v):
            self.bits[p] = 1
        self._insertions += 1

    def add_neighbors(self, node: int, neighbors):
        for nb in neighbors:
            self.add_edge(node, nb)

    def might_contain(self, u: int, v: int) -> bool:
        return all(self.bits[p] for p in self._positions(u, v))

    def fpr(self) -> float:
        """Theoretical false-positive rate."""
        n = self._insertions
        return (1 - np.exp(-self.k * n / self.m)) ** self.k

    def density(self) -> float:
        return float(self.bits.mean())

    def export(self) -> dict:
        return {"salt": self.salt, "bits": self.bits.copy(),
                "m": self.m, "k": self.k, "fpr": self.fpr()}

    def reset(self):
        self.bits[:] = 0
        self._insertions = 0


# ─────────────────────────────────────────────────────────────
#  Client encoder
# ─────────────────────────────────────────────────────────────

class ClientSAEncoder:
    """Builds Bloom sketch from a client's local subgraph per round."""

    def __init__(self, cid: int, m: int = 8192, k: int = 4):
        self.cid = cid
        self.m   = m
        self.k   = k
        # Deterministic client salt (seeded by cid for reproducibility)
        rng = np.random.default_rng(cid * 31337)
        self.salt = rng.bytes(16)

    def encode(self, edges, global_nodes) -> dict:
        """
        Encode edges into a Bloom sketch using GLOBAL node IDs.
        local2global is needed so the sketch can be compared across clients.
        """
        sketch = BloomSketch(m=self.m, k=self.k, salt=self.salt)
        l2g = {i: g for i, g in enumerate(global_nodes)}
        for (lu, lv) in edges:
            gu = l2g.get(lu, lu)
            gv = l2g.get(lv, lv)
            sketch.add_edge(gu, gv)
        export = sketch.export()
        export["cid"] = self.cid
        return export


# ─────────────────────────────────────────────────────────────
#  Server aggregator
# ─────────────────────────────────────────────────────────────

class ServerSAAggregator:
    """
    Merges Bloom sketches from all clients via bitwise OR.
    The merged bit-vector approximates the global adjacency without
    revealing any client's individual edges.
    """

    def __init__(self):
        self.global_bits = None
        self.m = None
        self.k = None
        self.num_merged = 0

    def merge(self, exports: list):
        if not exports:
            return
        self.m = exports[0]["m"]
        self.k = exports[0]["k"]
        self.global_bits = np.zeros(self.m, dtype=np.uint8)
        for exp in exports:
            self.global_bits = np.bitwise_or(self.global_bits, exp["bits"])
        self.num_merged = len(exports)

    def stats(self) -> dict:
        if self.global_bits is None:
            return {}
        density = float(self.global_bits.mean())
        fpr = density ** self.k if self.k else 0.0
        return {"density": density, "est_fpr": fpr,
                "num_clients": self.num_merged, "m": self.m}
