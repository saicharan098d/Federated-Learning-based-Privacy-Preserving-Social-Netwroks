"""
tests/test_aggregation.py
──────────────────────────
Unit tests for server-side aggregation strategies.
"""

import numpy as np
import pytest
from server.aggregation import aggregate


class TestAggregation:
    """Test aggregation strategies."""

    def test_fedavg_aggregation(self):
        """Test FedAvg (standard averaging) aggregation."""
        # Create simple client updates
        client_weights = [
            {"W": np.array([[1.0, 2.0], [3.0, 4.0]])},
            {"W": np.array([[5.0, 6.0], [7.0, 8.0]])},
        ]
        client_sizes = [10, 20]

        # Test FedAvg
        agg_weights = aggregate(client_weights, client_sizes, method="fedavg")
        
        # Expected: weighted average
        expected = {
            "W": (10 * np.array([[1.0, 2.0], [3.0, 4.0]]) + 
                  20 * np.array([[5.0, 6.0], [7.0, 8.0]])) / 30
        }
        
        np.testing.assert_array_almost_equal(agg_weights["W"], expected["W"])

    def test_aggregation_methods_exist(self):
        """Test that all aggregation methods don't crash."""
        client_weights = [
            {"W": np.random.randn(10, 5).astype(np.float32)},
            {"W": np.random.randn(10, 5).astype(np.float32)},
            {"W": np.random.randn(10, 5).astype(np.float32)},
        ]
        client_sizes = [100, 150, 200]
        
        methods = ["fedavg", "qfedavg", "krum", "trimmedmean", "median"]
        
        for method in methods:
            try:
                result = aggregate(client_weights, client_sizes, method=method)
                assert "W" in result
                assert result["W"].shape == (10, 5)
            except ValueError:
                pytest.fail(f"Aggregation method '{method}' failed")

    def test_aggregation_invalid_method(self):
        """Test that invalid method raises error."""
        client_weights = [{"W": np.random.randn(5, 5).astype(np.float32)}]
        client_sizes = [100]
        
        with pytest.raises(ValueError):
            aggregate(client_weights, client_sizes, method="invalid_method")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
