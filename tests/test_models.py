"""
tests/test_models.py
─────────────────────
Unit tests for client models (GCN, GAT, R-GCN).
"""

import numpy as np
import pytest
from client.model import GCN, relu, softmax, dropout, xavier, norm_adj


class TestActivations:
    """Test activation functions."""

    def test_relu(self):
        """Test ReLU activation."""
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        y = relu(x)
        expected = np.array([0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(y, expected)

    def test_softmax(self):
        """Test softmax activation."""
        x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        y = softmax(x)
        # Check that sum of probabilities is 1
        np.testing.assert_array_almost_equal(y.sum(axis=1), [1.0, 1.0])
        # Check that all values are between 0 and 1
        assert np.all(y >= 0) and np.all(y <= 1)

    def test_dropout(self):
        """Test dropout layer during training."""
        rng = np.random.default_rng(42)
        x = np.ones((10, 5))
        # During training with rate=0.5, about half should be zeroed
        y = dropout(x, rate=0.5, training=True, rng=rng)
        # Check that some values are zero and some are non-zero
        assert (y == 0).any() and (y != 0).any()
        # During inference, dropout should be identity
        y_infer = dropout(x, rate=0.5, training=False, rng=rng)
        np.testing.assert_array_equal(y_infer, x)


class TestXavier:
    """Test Xavier/Glorot initialization."""

    def test_xavier_shape(self):
        """Test that Xavier initialization has correct shape."""
        rng = np.random.default_rng(42)
        W = xavier(10, 5, rng)
        assert W.shape == (10, 5)

    def test_xavier_range(self):
        """Test that Xavier initialization is within expected range."""
        rng = np.random.default_rng(42)
        W = xavier(100, 100, rng)
        limit = np.sqrt(6.0 / 200)
        assert np.all(W >= -limit) and np.all(W <= limit)


class TestGCN:
    """Test Graph Convolutional Network."""

    def test_gcn_initialization(self):
        """Test GCN initialization."""
        gcn = GCN(in_dim=10, hidden_dim=16, num_classes=4, dropout=0.5, seed=42)
        assert gcn.W0.shape == (10, 16)
        assert gcn.W1.shape == (16, 4)

    def test_gcn_forward(self):
        """Test GCN forward pass."""
        gcn = GCN(in_dim=10, hidden_dim=16, num_classes=4, dropout=0.5, seed=42)
        
        # Create dummy input
        X = np.random.randn(20, 10).astype(np.float32)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        n = 20
        
        # Forward pass
        probs, h1 = gcn.forward(X, edges, n, training=True)
        
        # Check output shapes
        assert probs.shape == (20, 4)
        assert h1.shape == (20, 16)
        
        # Check that probabilities sum to 1
        np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(20))

    def test_gcn_weights_save_load(self):
        """Test get/set weights."""
        gcn = GCN(in_dim=10, hidden_dim=16, num_classes=4, seed=42)
        
        # Save original weights
        original_weights = gcn.get_weights()
        
        # Modify weights
        new_weights = {
            "W0": np.random.randn(10, 16).astype(np.float32),
            "W1": np.random.randn(16, 4).astype(np.float32),
        }
        gcn.set_weights(new_weights)
        
        # Verify new weights are set
        current_weights = gcn.get_weights()
        np.testing.assert_array_almost_equal(current_weights["W0"], new_weights["W0"])
        np.testing.assert_array_almost_equal(current_weights["W1"], new_weights["W1"])


class TestNormAdjacency:
    """Test normalized adjacency matrix computation."""

    def test_norm_adj_empty_graph(self):
        """Test with empty edge list."""
        A = norm_adj([], 5)
        # Should be identity matrix
        assert A.shape == (5, 5)
        np.testing.assert_array_almost_equal(A.toarray(), np.eye(5, dtype=np.float32))

    def test_norm_adj_simple_graph(self):
        """Test with simple graph."""
        edges = [(0, 1), (1, 2), (2, 0)]
        A = norm_adj(edges, 3)
        assert A.shape == (3, 3)
        # Check symmetry
        np.testing.assert_array_almost_equal(A.toarray(), A.toarray().T)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
