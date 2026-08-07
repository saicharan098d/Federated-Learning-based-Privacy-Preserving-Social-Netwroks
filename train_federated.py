"""
train_federated.py
──────────────────
Main federated training script.

Usage:
    # Quick test (synthetic graph)
    python train_federated.py --synthetic --rounds 10

    # Full run with Facebook SNAP dataset
    python train_federated.py --rounds 50 --clients 5 --model gat --agg qfedavg

    # With ablation study
    python train_federated.py --synthetic --rounds 10 --ablation

Results are written to:
    experiments/log_run.csv           ← Streamlit reads this
    experiments/history.json
    experiments/checkpoints/
"""

import os
import sys
import json
import pickle
import argparse
import time
import numpy as np
import pandas as pd
from collections import defaultdict

# ── path setup ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client.model    import build_model
from client.secure_sa import ClientSAEncoder, ServerSAAggregator
from client.secure_na import ClientNAEncoder, ServerNAAggregator, PrivacyAccountant
from server.aggregation import aggregate, fairness_metrics, clip_update


# ─────────────────────────────────────────────────────────────
#  Loss & accuracy helpers
# ─────────────────────────────────────────────────────────────

def cross_entropy(probs, labels, mask):
    n = mask.sum()
    if n == 0: return 0.0
    y = labels[mask].astype(int)
    p = probs[mask]
    return float(-np.log(p[np.arange(n), y] + 1e-10).mean())


def accuracy(probs, labels, mask):
    if mask.sum() == 0: return 0.0
    return float((probs[mask].argmax(1) == labels[mask]).mean())


def f1_macro(probs, labels, mask):
    if mask.sum() == 0: return 0.0
    preds = probs[mask].argmax(1)
    y     = labels[mask]
    classes = np.unique(y)
    f1s = []
    for c in classes:
        tp = ((preds == c) & (y == c)).sum()
        fp = ((preds == c) & (y != c)).sum()
        fn = ((preds != c) & (y == c)).sum()
        prec = tp / (tp + fp + 1e-10)
        rec  = tp / (tp + fn + 1e-10)
        f1s.append(2 * prec * rec / (prec + rec + 1e-10))
    return float(np.mean(f1s))


# ─────────────────────────────────────────────────────────────
#  Simple gradient update for GCN (SGD)
# ─────────────────────────────────────────────────────────────

def sgd_step(model, X, edges, n, y, mask, lr, clip_norm):
    """One SGD step using finite-difference gradient (model-agnostic)."""
    old_w = model.get_weights()

    # Forward
    probs, h1 = model.forward(X, edges, n, training=True)
    loss = cross_entropy(probs, y, mask)

    # Backward (exact for GCN W1, finite-diff for rest)
    from client.model import GCN, norm_adj
    if isinstance(model, GCN):
        m_arr = mask
        k     = mask.sum()
        A_hat = model._A if model._A is not None else norm_adj(edges, n)

        # dL/dW1
        delta = probs.copy()
        delta[m_arr, y[m_arr].astype(int)] -= 1
        delta[m_arr] /= max(k, 1)
        Ah1  = A_hat @ h1
        dW1  = Ah1.T @ delta
        norm = np.linalg.norm(dW1)
        if norm > clip_norm: dW1 *= clip_norm / norm

        # dL/dW0
        backprop = (A_hat @ delta) @ model.W1.T
        backprop *= (h1 > 0)
        AX  = A_hat @ X
        dW0 = AX.T @ backprop
        norm = np.linalg.norm(dW0)
        if norm > clip_norm: dW0 *= clip_norm / norm

        model.W0 -= lr * dW0
        model.W1 -= lr * dW1
    else:
        # Finite-difference for GAT
        eps_fd = 1e-4
        from client.model import GCN  # for isinstance checks only
        flat_old = _flatten_any(old_w)
        grad = np.zeros_like(flat_old)
        n_params = len(flat_old)
        sample_idx = np.random.choice(n_params, max(1, n_params // 10), replace=False)
        for i in sample_idx:
            flat_old[i] += eps_fd
            model.set_weights(_unflatten_any(flat_old, old_w))
            p2, _ = model.forward(X, edges, n, training=False)
            l2 = cross_entropy(p2, y, mask)
            flat_old[i] -= 2 * eps_fd
            model.set_weights(_unflatten_any(flat_old, old_w))
            p3, _ = model.forward(X, edges, n, training=False)
            l3 = cross_entropy(p3, y, mask)
            grad[i] = (l2 - l3) / (2 * eps_fd)
            flat_old[i] += eps_fd
        norm = np.linalg.norm(grad)
        if norm > clip_norm: grad *= clip_norm / norm
        flat_new = flat_old - lr * grad
        model.set_weights(_unflatten_any(flat_new, old_w))

    return loss


def _flatten_any(d):
    parts = []
    for k in sorted(d):
        v = d[k]
        if isinstance(v, np.ndarray):   parts.append(v.flatten())
        elif isinstance(v, list):       [parts.append(x.flatten()) for x in v]
        elif isinstance(v, dict):       parts.append(_flatten_any(v))
    return np.concatenate(parts) if parts else np.array([])


def _unflatten_any(flat, template):
    result = {}
    offset = 0
    for k in sorted(template):
        v = template[k]
        if isinstance(v, np.ndarray):
            result[k] = flat[offset:offset+v.size].reshape(v.shape).astype(np.float32)
            offset += v.size
        elif isinstance(v, list):
            result[k] = []
            for w in v:
                result[k].append(flat[offset:offset+w.size].reshape(w.shape).astype(np.float32))
                offset += w.size
        elif isinstance(v, dict):
            sub, n = _unflatten_sub(flat[offset:], v)
            result[k] = sub
            offset += n
    return result


def _unflatten_sub(flat, template):
    result = {}
    offset = 0
    for k in sorted(template):
        v = template[k]
        if isinstance(v, np.ndarray):
            result[k] = flat[offset:offset+v.size].reshape(v.shape)
            offset += v.size
        elif isinstance(v, list):
            result[k] = []
            for w in v:
                result[k].append(flat[offset:offset+w.size].reshape(w.shape))
                offset += w.size
    return result, offset


# ─────────────────────────────────────────────────────────────
#  Client trainer
# ─────────────────────────────────────────────────────────────

class ClientTrainer:
    def __init__(self, cid, data, model, cfg):
        self.cid   = cid
        self.data  = data
        self.model = model
        self.cfg   = cfg

        self.sa_encoder = ClientSAEncoder(cid=cid,
                                          m=cfg.get("bloom_m", 8192),
                                          k=cfg.get("bloom_k", 4))
        self.na_encoder = ClientNAEncoder(cid=cid,
                                          eps_min=cfg.get("eps_min", 0.5),
                                          eps_max=cfg.get("eps_max", 2.0),
                                          clip=cfg.get("emb_clip", 1.0),
                                          seed=cid)

    def set_weights(self, w):
        self.model.set_weights(w)

    def train(self, round_num):
        X   = self.data["X"]
        y   = self.data["y"]
        trm = self.data["train_mask"]
        vlm = self.data["val_mask"]
        tsm = self.data["test_mask"]
        edges   = self.data["edges"]
        n       = self.data["num_nodes"]
        overlap = self.data["overlap_local"]
        gnodes  = self.data["nodes_global"]

        cfg   = self.cfg
        lr    = cfg.get("lr", 0.01)
        ep    = cfg.get("local_epochs", 5)
        clip  = cfg.get("clip_norm", 1.0)

        losses = []
        for _ in range(ep):
            loss = sgd_step(self.model, X, edges, n, y, trm, lr, clip)
            losses.append(loss)

        # Final eval
        probs, emb = self.model.forward(X, edges, n, training=False)
        val_acc    = accuracy(probs, y, vlm)
        test_acc   = accuracy(probs, y, tsm)
        val_f1     = f1_macro(probs, y, vlm)
        train_loss = float(np.mean(losses))

        # SecureSA sketch
        sketch = None
        if cfg.get("secure_sa", True):
            sketch = self.sa_encoder.encode(edges, gnodes)

        # SecureNA LDP
        na_pack = None
        if cfg.get("secure_na", True):
            na_pack = self.na_encoder.encode(emb, edges, n, overlap, gnodes)
            mean_eps = na_pack["mean_eps"]
        else:
            mean_eps = cfg.get("eps_max", 2.0)

        # Weight size for comm cost estimate
        flat_w = _flatten_any(self.model.get_weights())
        comm_bytes = flat_w.nbytes
        if sketch: comm_bytes += sketch["bits"].nbytes

        print(f"    Client {self.cid:2d} | loss={train_loss:.4f} | "
              f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | "
              f"ε={mean_eps:.2f}")

        return {
            "cid":        self.cid,
            "weights":    self.model.get_weights(),
            "num_train":  int(trm.sum()),
            "train_loss": train_loss,
            "val_acc":    val_acc,
            "test_acc":   test_acc,
            "val_f1":     val_f1,
            "sketch":     sketch,
            "na_pack":    na_pack,
            "mean_eps":   mean_eps,
            "comm_bytes": comm_bytes,
        }


# ─────────────────────────────────────────────────────────────
#  Main orchestrator
# ─────────────────────────────────────────────────────────────

class FederatedOrchestrator:
    def __init__(self, cfg):
        self.cfg     = cfg
        self.sa_agg  = ServerSAAggregator()
        self.na_agg  = ServerNAAggregator()
        self.privacy = PrivacyAccountant()
        self.logs    = []
        os.makedirs("experiments/checkpoints", exist_ok=True)

    def _load_clients(self):
        data_dir = self.cfg.get("data_dir", "data")
        n_clients = self.cfg.get("num_clients", 5)
        clients = []
        for cid in range(n_clients):
            path = os.path.join(data_dir, f"client_{cid}_data.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Client data not found: {path}\n"
                    f"Run: python preprocess.py --clients {n_clients}"
                )
            with open(path, "rb") as f:
                data = pickle.load(f)

            model = build_model(
                model_type  = self.cfg.get("model", "gcn"),
                in_dim      = data["in_dim"],
                hidden_dim  = self.cfg.get("hidden_dim", 64),
                num_classes = data["num_classes"],
                dropout     = self.cfg.get("dropout", 0.5),
                heads       = self.cfg.get("heads", 4),
                seed        = self.cfg.get("seed", 42) + cid,
            )
            trainer = ClientTrainer(cid, data, model, self.cfg)
            clients.append(trainer)

        return clients

    def run(self):
        cfg = self.cfg
        clients = self._load_clients()
        n_clients = len(clients)

        # Sample fraction
        frac   = cfg.get("fraction_fit", 1.0)
        rounds = cfg.get("num_rounds", 50)
        agg_m  = cfg.get("agg_method", "fedavg")

        # Init global weights from client 0
        global_w = clients[0].model.get_weights()
        for c in clients:
            c.set_weights(global_w)

        print(f"\n{'═'*60}")
        print(f"  Federated Training ({rounds} rounds | {n_clients} clients)")
        print(f"  Model={cfg.get('model','gcn').upper()} | Agg={agg_m.upper()}")
        print(f"  SecureSA={cfg.get('secure_sa',True)} | SecureNA={cfg.get('secure_na',True)}")
        print(f"{'═'*60}\n")

        for rnd in range(1, rounds + 1):
            t0 = time.time()

            # Sample clients
            rng = np.random.default_rng(cfg.get("seed", 42) + rnd)
            n_sel = max(1, int(n_clients * frac))
            sel   = sorted(rng.choice(n_clients, n_sel, replace=False).tolist())
            print(f"[Round {rnd:3d}/{rounds}] Clients: {sel}")

            # Broadcast
            for cid in sel:
                clients[cid].set_weights(global_w)

            # Local training
            updates = [clients[cid].train(rnd) for cid in sel]

            # SecureSA
            sketches = [u["sketch"] for u in updates if u["sketch"]]
            if sketches:
                self.sa_agg.merge(sketches)
                sa_stats = self.sa_agg.stats()
                print(f"  [SecureSA] density={sa_stats.get('density',0):.3f} | "
                      f"est_fpr={sa_stats.get('est_fpr',0):.4f}")

            # SecureNA
            na_packs = [u["na_pack"] for u in updates if u["na_pack"]]
            merged_overlaps = self.na_agg.merge(na_packs)
            if merged_overlaps:
                print(f"  [SecureNA] merged {len(merged_overlaps)} overlap embeddings")

            # Privacy accounting
            eps_vals = [u["mean_eps"] for u in updates]
            self.privacy.update(rnd, eps_vals)
            priv = self.privacy.summary()

            # Aggregation
            global_w = aggregate(updates, method=agg_m, cfg=cfg.get("agg_cfg", {}))

            # Update all client models
            for c in clients:
                c.set_weights(global_w)

            # Metrics
            fm   = fairness_metrics(updates)
            wacc = np.average([u["val_acc"] for u in updates],
                               weights=[u["num_train"] for u in updates])
            wacc_test = np.average([u["test_acc"] for u in updates],
                                    weights=[u["num_train"] for u in updates])
            wf1  = np.average([u["val_f1"] for u in updates],
                               weights=[u["num_train"] for u in updates])
            comm = sum(u["comm_bytes"] for u in updates) / 1024 / 1024
            elapsed = time.time() - t0

            log = {
                "round":           rnd,
                "global_acc":      round(float(wacc), 4),
                "global_test_acc": round(float(wacc_test), 4),
                "global_f1":       round(float(wf1), 4),
                "worst_client_f1": round(fm["worst_val_acc"], 4),
                "best_client_f1":  round(fm["best_val_acc"], 4),
                "fairness_gap":    round(fm["gap"], 4),
                "mean_eps":        round(float(np.mean(eps_vals)), 4),
                "cumulative_eps":  round(priv["cumulative_eps"], 4),
                "comms_mb":        round(float(comm), 4),
                "round_time_s":    round(elapsed, 2),
            }
            self.logs.append(log)

            # Save CSV every round (so Streamlit can read live)
            pd.DataFrame(self.logs).to_csv("experiments/log_run.csv", index=False)

            # Checkpoint every 10 rounds
            if rnd % 10 == 0:
                with open(f"experiments/checkpoints/round_{rnd}.pkl", "wb") as f:
                    pickle.dump({"round": rnd, "weights": global_w}, f)

            print(f"  ▶ val_acc={wacc:.4f} | test_acc={wacc_test:.4f} | "
                  f"worst={fm['worst_val_acc']:.4f} | gap={fm['gap']:.4f} | "
                  f"ε={priv['cumulative_eps']:.2f} | {comm:.2f}MB | {elapsed:.1f}s\n")

        # Final save
        with open("experiments/history.json", "w") as f:
            json.dump(self.logs, f, indent=2)

        best = max(self.logs, key=lambda x: x["global_acc"])
        print(f"\n{'═'*60}")
        print(f"  Training Complete!")
        print(f"  Best round:     {best['round']}")
        print(f"  Best val acc:   {best['global_acc']}")
        print(f"  Test acc:       {best['global_test_acc']}")
        print(f"  Worst client:   {min(l['worst_client_f1'] for l in self.logs):.4f}")
        print(f"  Total comms:    {sum(l['comms_mb'] for l in self.logs):.2f} MB")
        print(f"  Final ε:        {self.logs[-1]['cumulative_eps']:.4f}")
        print(f"  Results → experiments/log_run.csv")
        print(f"{'═'*60}\n")

        return self.logs


# ─────────────────────────────────────────────────────────────
#  Ablation runner
# ─────────────────────────────────────────────────────────────

def run_ablation(base_cfg, ablation_rounds=5):
    """Run ablation study comparing system components."""
    import copy

    ablations = {
        "Full System":       base_cfg,
        "- SecureSA":        {**base_cfg, "secure_sa": False},
        "- SecureNA":        {**base_cfg, "secure_na": False},
        "- Adaptive ε":      {**base_cfg, "eps_min": 1.0, "eps_max": 1.0},
        "FedAvg (baseline)": {**base_cfg, "agg_method": "fedavg"},
        "Local Only":        {**base_cfg, "num_rounds": 1, "fraction_fit": 0.2},
    }

    results_abl = []
    for name, cfg in ablations.items():
        print(f"\n{'═'*55}\n  ABLATION: {name}\n{'═'*55}")
        cfg2 = {**cfg, "num_rounds": ablation_rounds}
        orch = FederatedOrchestrator(cfg2)
        logs = orch.run()
        best = max(logs, key=lambda x: x["global_acc"])
        results_abl.append({
            "config":          name,
            "best_val_acc":    best["global_acc"],
            "best_test_acc":   best["global_test_acc"],
            "worst_client":    min(l["worst_client_f1"] for l in logs),
            "fairness_gap":    max(l["fairness_gap"] for l in logs),
            "final_eps":       logs[-1]["cumulative_eps"],
            "total_comms_mb":  sum(l["comms_mb"] for l in logs),
        })

    df_abl = pd.DataFrame(results_abl)
    df_abl.to_csv("experiments/ablation_results.csv", index=False)

    print(f"\n{'─'*80}")
    print(f"  {'Config':<25} {'Val Acc':>9} {'Test':>9} {'Worst':>9} {'Gap':>9} {'ε':>8} {'MB':>8}")
    print(f"{'─'*80}")
    for r in results_abl:
        print(f"  {r['config']:<25} {r['best_val_acc']:>9.4f} {r['best_test_acc']:>9.4f} "
              f"{r['worst_client']:>9.4f} {r['fairness_gap']:>9.4f} "
              f"{r['final_eps']:>8.3f} {r['total_comms_mb']:>8.1f}")
    print(f"{'─'*80}")
    print(f"\n  Ablation results → experiments/ablation_results.csv\n")
    return results_abl


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    default="data")
    p.add_argument("--clients",     type=int,   default=5)
    p.add_argument("--rounds",      type=int,   default=20)
    p.add_argument("--model",       default="gcn", choices=["gcn", "gat"])
    p.add_argument("--hidden",      type=int,   default=64)
    p.add_argument("--lr",          type=float, default=0.01)
    p.add_argument("--local-epochs",type=int,   default=5)
    p.add_argument("--agg",         default="fedavg",
                   choices=["fedavg","qfedavg","krum","trimmed_mean","median"])
    p.add_argument("--eps-min",     type=float, default=0.5)
    p.add_argument("--eps-max",     type=float, default=2.0)
    p.add_argument("--no-secure-sa",action="store_true")
    p.add_argument("--no-secure-na",action="store_true")
    p.add_argument("--synthetic",   action="store_true",
                   help="Auto-run preprocessing with synthetic graph")
    p.add_argument("--ablation",    action="store_true")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("experiments/checkpoints", exist_ok=True)

    # Auto-preprocess if needed
    needs_preprocess = not all(
        os.path.exists(f"data/client_{i}_data.pkl") for i in range(args.clients)
    )
    if needs_preprocess or args.synthetic:
        print("  [Setup] Running preprocessing first...")
        import subprocess
        cmd = [sys.executable, "preprocess.py",
               "--clients", str(args.clients),
               "--seed", str(args.seed)]
        if args.synthetic:
            cmd.append("--synthetic")
        subprocess.run(cmd, check=True)

    cfg = {
        "data_dir":      args.data_dir,
        "num_clients":   args.clients,
        "num_rounds":    args.rounds,
        "fraction_fit":  1.0,
        "model":         args.model,
        "hidden_dim":    args.hidden,
        "lr":            args.lr,
        "local_epochs":  args.local_epochs,
        "dropout":       0.5,
        "heads":         4,
        "agg_method":    args.agg,
        "agg_cfg":       {"q": 0.3, "alpha": 1.0, "tau": 1e-3, "f": 1, "trim": 0.1},
        "eps_min":       args.eps_min,
        "eps_max":       args.eps_max,
        "secure_sa":     not args.no_secure_sa,
        "secure_na":     not args.no_secure_na,
        "bloom_m":       8192,
        "bloom_k":       4,
        "emb_clip":      1.0,
        "clip_norm":     1.0,
        "seed":          args.seed,
    }

    if args.ablation:
        run_ablation(cfg, ablation_rounds=min(args.rounds, 5))
    else:
        orch = FederatedOrchestrator(cfg)
        orch.run()


if __name__ == "__main__":
    main()
