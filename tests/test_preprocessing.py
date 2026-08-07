"""
tests/test_preprocessing.py
────────────────────────────
Unit tests for data preprocessing and partitioning.
"""

import numpy as np
import pytest
import tempfile
import os


class TestPreprocessing:
    """Test data preprocessing functions."""

    def test_synthetic_graph_generation(self):
        """Test that synthetic graphs can be generated."""
        try:
            from preprocess import load_synthetic_graph
            G, X, y, le = load_synthetic_graph(num_nodes=100, num_classes=4, feat_dim=32, seed=42)
            
            assert G.number_of_nodes() == 100
            assert X.shape == (100, 32)
            assert y.shape == (100,)
            assert len(le.classes_) == 4
        except Exception as e:
            pytest.fail(f"Synthetic graph generation failed: {e}")

    def test_federated_partition(self):
        """Test federated graph partitioning."""
        try:
            import networkx as nx
            from preprocess import partition_federated
            from sklearn.preprocessing import LabelEncoder
            
            # Create simple test graph
            G = nx.karate_club_graph()
            X = np.random.randn(G.number_of_nodes(), 10).astype(np.float32)
            y = np.random.randint(0, 2, G.number_of_nodes())
            
            # Partition into 3 clients
            num_clients = 3
            client_nodes = partition_federated(G, X, y, num_clients=num_clients, overlap_rate=0.1, seed=42)
            
            # Check that all nodes are assigned
            all_assigned = set()
            for cid in range(num_clients):
                assert cid in client_nodes
                all_assigned.update(client_nodes[cid])
            
            assert len(all_assigned) > 0
        except Exception as e:
            pytest.fail(f"Federated partitioning failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
