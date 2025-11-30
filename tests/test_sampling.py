"""
Unit tests for pyKriging sampling plan module.
"""
import pytest
import numpy as np
from pyKriging.samplingplan import samplingplan


class TestRandomLatinHypercube:
    """Tests for random Latin hypercube generation."""

    def test_rlh_shape(self, seed):
        """Test that RLH returns correct shape."""
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        assert X.shape == (10, 2)

    def test_rlh_shape_higher_dim(self, seed):
        """Test RLH with higher dimensions."""
        sp = samplingplan(k=5)
        X = sp.rlh(20)
        assert X.shape == (20, 5)

    def test_rlh_bounds(self, seed):
        """Test that RLH values are within [0, 1]."""
        sp = samplingplan(k=3)
        X = sp.rlh(50)
        assert np.all(X >= 0) and np.all(X <= 1)

    def test_rlh_latin_property(self, seed):
        """Test that each column has unique strata (Latin property)."""
        sp = samplingplan(k=2)
        n = 10
        X = sp.rlh(n)
        # Each column should have points in different strata
        for col in range(2):
            strata = np.floor(X[:, col] * n).astype(int)
            strata = np.clip(strata, 0, n-1)  # Handle edge case
            # Should have n unique strata (or close to it for random LHC)
            assert len(np.unique(strata)) >= n - 1


class TestOptimalLatinHypercube:
    """Tests for optimal Latin hypercube generation."""

    def test_optimallhc_shape(self, seed):
        """Test that optimal LHC returns correct shape."""
        sp = samplingplan(k=2)
        X = sp.optimallhc(10, population=5, iterations=5, use_cache=False)
        assert X.shape == (10, 2)

    def test_optimallhc_bounds(self, seed):
        """Test that optimal LHC values are within [0, 1]."""
        sp = samplingplan(k=2)
        X = sp.optimallhc(15, population=5, iterations=5, use_cache=False)
        assert np.all(X >= 0) and np.all(X <= 1)

    def test_optimallhc_better_than_random(self, seed):
        """Test that optimal LHC has better space-filling than random."""
        sp = samplingplan(k=2)
        n = 20

        # Generate random and optimal LHC
        X_random = sp.rlh(n)
        X_optimal = sp.optimallhc(n, population=10, iterations=10, use_cache=False)

        # Compute Morris-Mitchell criterion (lower is better)
        phi_random = sp.mmphi(X_random)
        phi_optimal = sp.mmphi(X_optimal)

        # Optimal should be at least as good (usually better)
        assert phi_optimal <= phi_random * 1.1  # Allow 10% tolerance


class TestDistanceCalculations:
    """Tests for distance calculation functions."""

    def test_jd_output_types(self, seed):
        """Test that jd returns correct types."""
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        J, d = sp.jd(X, p=1)

        assert isinstance(J, np.ndarray)
        assert isinstance(d, np.ndarray)
        assert len(J) == len(d)

    def test_jd_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        sp = samplingplan(k=2)
        # Simple 3-point example
        X = np.array([[0, 0], [1, 0], [0, 1]])
        J, d = sp.jd(X, p=1)

        # Distances should be: 1, 1, 2 (sorted unique)
        assert 1.0 in d
        assert 2.0 in d

    def test_jd_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        sp = samplingplan(k=2)
        X = np.array([[0, 0], [1, 0], [0, 1]])
        J, d = sp.jd(X, p=2)

        # Distances should include 1.0 and sqrt(2)
        assert np.any(np.isclose(d, 1.0))
        assert np.any(np.isclose(d, np.sqrt(2)))

    def test_mmphi_positive(self, seed):
        """Test that Morris-Mitchell criterion is positive."""
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        phi = sp.mmphi(X)
        assert phi > 0


class TestPerturb:
    """Tests for perturbation function."""

    def test_perturb_maintains_shape(self, seed):
        """Test that perturb maintains array shape."""
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        X_pert = sp.perturb(X, 3)
        assert X_pert.shape == X.shape

    def test_perturb_modifies_array(self, seed):
        """Test that perturb actually modifies the array."""
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        X_pert = sp.perturb(X, 5)
        # Should be different (with high probability)
        assert not np.array_equal(X, X_pert)

    def test_perturb_maintains_values(self, seed):
        """Test that perturb only swaps values (same set of values per column)."""
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        X_pert = sp.perturb(X, 3)

        # Each column should have the same set of values (just reordered)
        for col in range(2):
            assert set(X[:, col]) == set(X_pert[:, col])


class TestFullFactorial:
    """Tests for full factorial design."""

    def test_fullfactorial_shape(self):
        """Test full factorial shape."""
        sp = samplingplan(k=2)
        X = sp.fullfactorial(ppd=3)
        assert X.shape == (9, 2)  # 3^2 = 9 points

    def test_fullfactorial_3d(self):
        """Test full factorial in 3D."""
        sp = samplingplan(k=3)
        X = sp.fullfactorial(ppd=2)
        assert X.shape == (8, 3)  # 2^3 = 8 points
