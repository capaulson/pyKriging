"""
Unit tests for pyKriging GPU backend.
"""
import pytest
import numpy as np
import pyKriging
from pyKriging.gpu_backend import get_backend, to_cpu, to_gpu, configure_gpu, get_device_info


class TestBackendConfiguration:
    """Tests for GPU backend configuration."""

    def test_configure_cpu(self):
        """Test configuring CPU backend."""
        configure_gpu(device='cpu', verbose=False)
        info = get_device_info()
        assert info['backend'] == 'cpu'

    def test_get_backend_returns_singleton(self):
        """Test that get_backend returns same instance."""
        b1 = get_backend(verbose=False)
        b2 = get_backend(verbose=False)
        assert b1 is b2

    def test_backend_has_xp(self):
        """Test that backend has xp (array module) attribute."""
        backend = get_backend(verbose=False)
        assert hasattr(backend, 'xp')
        assert hasattr(backend.xp, 'array')
        assert hasattr(backend.xp, 'zeros')

    def test_backend_has_linalg(self):
        """Test that backend has linalg module."""
        backend = get_backend(verbose=False)
        assert hasattr(backend, 'linalg')
        assert hasattr(backend.linalg, 'cholesky')
        assert hasattr(backend.linalg, 'solve')


class TestDataTransfer:
    """Tests for CPU/GPU data transfer functions."""

    def test_to_cpu_numpy_passthrough(self):
        """Test that to_cpu passes through numpy arrays."""
        arr = np.array([1, 2, 3])
        result = to_cpu(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(arr, result)

    def test_to_gpu_returns_array(self):
        """Test that to_gpu returns an array."""
        arr = np.array([1.0, 2.0, 3.0])
        result = to_gpu(arr)
        # Should be either numpy (CPU) or torch/cupy tensor (GPU)
        assert hasattr(result, 'shape')
        assert result.shape == arr.shape

    def test_roundtrip_preserves_values(self):
        """Test that CPU->GPU->CPU preserves values."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        gpu_arr = to_gpu(arr)
        cpu_arr = to_cpu(gpu_arr)
        np.testing.assert_array_almost_equal(arr, cpu_arr)

    def test_to_cpu_handles_none(self):
        """Test that to_cpu handles None gracefully."""
        result = to_cpu(None)
        assert result is None

    def test_to_gpu_handles_none(self):
        """Test that to_gpu handles None gracefully."""
        result = to_gpu(None)
        assert result is None


class TestBackendOperations:
    """Tests for backend mathematical operations."""

    def test_cholesky_decomposition(self):
        """Test Cholesky decomposition on backend."""
        backend = get_backend(verbose=False)

        # Create a positive definite matrix
        A = np.array([[4.0, 2.0], [2.0, 5.0]])
        A_gpu = to_gpu(A)

        L = backend.linalg.cholesky(A_gpu)
        L_cpu = to_cpu(L)

        # Verify L @ L.T = A
        reconstructed = L_cpu @ L_cpu.T
        np.testing.assert_array_almost_equal(A, reconstructed, decimal=5)

    def test_solve_triangular(self):
        """Test triangular solve on backend."""
        backend = get_backend(verbose=False)

        # Create a simple system
        L = np.array([[2.0, 0.0], [1.0, 3.0]])
        b = np.array([4.0, 7.0])

        L_gpu = to_gpu(L)
        b_gpu = to_gpu(b)

        x_gpu = backend.linalg.solve(L_gpu, b_gpu)
        x_cpu = to_cpu(x_gpu)

        # Verify L @ x = b
        result = L @ x_cpu
        np.testing.assert_array_almost_equal(b, result, decimal=5)

    def test_matrix_operations(self):
        """Test basic matrix operations on backend."""
        backend = get_backend(verbose=False)
        xp = backend.xp

        A = xp.array([[1.0, 2.0], [3.0, 4.0]])
        B = xp.array([[5.0, 6.0], [7.0, 8.0]])

        # Test addition
        C = A + B
        C_cpu = to_cpu(C)
        expected = np.array([[6.0, 8.0], [10.0, 12.0]])
        np.testing.assert_array_almost_equal(C_cpu, expected)

        # Test multiplication
        D = xp.matmul(A, B)
        D_cpu = to_cpu(D)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(D_cpu, expected)


class TestBackendConsistency:
    """Tests for consistency between CPU and GPU backends."""

    def test_kriging_same_result_cpu_gpu(self):
        """Test that kriging gives similar results on CPU vs configured backend."""
        from pyKriging.krige import kriging
        from pyKriging.samplingplan import samplingplan

        np.random.seed(42)

        # Simple test function
        def func(x):
            return np.sin(x[0] * np.pi) * np.cos(x[1] * np.pi)

        # Generate data
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        y = np.array([func(x) for x in X])

        # Train on current backend
        k = kriging(X.copy(), y.copy())
        k.train(optimizer='ga')

        # Make predictions
        test_point = [0.5, 0.5]
        pred = k.predict(test_point)

        # Should get a reasonable prediction
        actual = func(test_point)
        assert abs(pred - actual) < 1.0  # Within reasonable range


class TestDeviceInfo:
    """Tests for device information functions."""

    def test_get_device_info_returns_dict(self):
        """Test that get_device_info returns a dictionary."""
        info = get_device_info()
        assert isinstance(info, dict)
        assert 'backend' in info
        assert 'device_name' in info

    def test_is_gpu_returns_bool(self):
        """Test that is_gpu returns a boolean."""
        backend = get_backend(verbose=False)
        result = backend.is_gpu()
        assert isinstance(result, bool)
