"""
GPU Backend Abstraction Layer for pyKriging

This module provides a unified interface for GPU-accelerated linear algebra operations
with automatic fallback to CPU computation. It supports:
- NVIDIA GPUs via CuPy (CUDA backend)
- Apple Silicon via PyTorch MPS (Metal backend)
- CPU via NumPy (fallback when no GPU is available)

The module automatically detects available hardware and selects the best backend.
Users can also manually configure the backend via the configure_gpu() function.

Architecture:
    - GPUBackend: Main class that manages backend selection and array operations
    - get_backend(): Singleton function to get the global backend instance
    - configure_gpu(): User-facing API to configure GPU settings
    - to_cpu()/to_gpu(): Utility functions for data transfer between devices

Usage Example:
    >>> from pyKriging.gpu_backend import get_backend
    >>> backend = get_backend()
    >>> xp = backend.xp  # NumPy-compatible array module
    >>> linalg = backend.linalg  # Linear algebra operations
    >>>
    >>> # Use xp like numpy
    >>> x = xp.array([1, 2, 3])
    >>> y = xp.dot(x, x)
"""

import numpy as np
import warnings
import os
from typing import Optional, Union, Any, Tuple

# Global backend instance (singleton pattern)
_global_backend = None


class GPUBackend:
    """
    Unified GPU/CPU backend with automatic device selection.

    This class provides a NumPy-compatible API with GPU acceleration when available.
    It handles three backend types:
    - 'cuda': NVIDIA GPUs using CuPy
    - 'metal': Apple Silicon using PyTorch with MPS backend
    - 'cpu': Standard NumPy (fallback)

    Attributes:
        backend_type (str): The active backend type ('cuda', 'metal', or 'cpu')
        xp: NumPy-compatible array module (cupy, torch->numpy wrapper, or numpy)
        linalg: Linear algebra module with GPU support
        device: Device object for GPU operations (None for CPU)
        available_backends (list): List of all available backend types
    """

    def __init__(self, device: str = 'auto', verbose: bool = True):
        """
        Initialize the GPU backend with automatic or manual device selection.

        Args:
            device: Backend device to use. Options:
                - 'auto': Automatically select best available (CUDA > Metal > CPU)
                - 'cuda': Force CUDA/CuPy backend (requires NVIDIA GPU)
                - 'metal': Force Metal/MPS backend (requires Apple Silicon)
                - 'cpu': Force CPU/NumPy backend
            verbose: If True, print backend selection information
        """
        self.verbose = verbose
        self.backend_type = None
        self.xp = None  # Array module (numpy-compatible)
        self.linalg = None  # Linear algebra module
        self.device = None  # Device object for GPU operations
        self.available_backends = []

        # CuPy-specific attributes (for CUDA backend)
        self._cupy = None
        self._cupyx = None

        # PyTorch-specific attributes (for Metal backend)
        self._torch = None
        self._torch_device = None

        # Detect available backends
        self._detect_available_backends()

        # Initialize the requested backend
        if device == 'auto':
            self._auto_select_backend()
        else:
            self._select_backend(device)

        if self.verbose:
            self._print_backend_info()

    def _detect_available_backends(self):
        """
        Detect which GPU backends are available on the system.
        Populates the available_backends list.
        """
        # CPU is always available
        self.available_backends.append('cpu')

        # Check for CUDA (CuPy)
        try:
            import cupy as cp
            if cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0:
                self.available_backends.append('cuda')
                self._cupy = cp
                try:
                    import cupyx
                    self._cupyx = cupyx
                except ImportError:
                    pass
        except (ImportError, Exception):
            pass

        # Check for Metal (PyTorch with MPS)
        try:
            import torch
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.available_backends.append('metal')
                self._torch = torch
        except (ImportError, Exception):
            pass

    def _auto_select_backend(self):
        """
        Automatically select the best available backend.
        Priority: CUDA > Metal > CPU
        """
        if 'cuda' in self.available_backends:
            self._select_backend('cuda')
        elif 'metal' in self.available_backends:
            self._select_backend('metal')
        else:
            self._select_backend('cpu')

    def _select_backend(self, device: str):
        """
        Select and initialize a specific backend.

        Args:
            device: Backend to initialize ('cuda', 'metal', or 'cpu')

        Raises:
            RuntimeError: If requested backend is not available
        """
        if device not in self.available_backends:
            available_str = ', '.join(self.available_backends)
            raise RuntimeError(
                f"Backend '{device}' is not available on this system. "
                f"Available backends: {available_str}"
            )

        self.backend_type = device

        if device == 'cuda':
            self._init_cuda_backend()
        elif device == 'metal':
            self._init_metal_backend()
        elif device == 'cpu':
            self._init_cpu_backend()
        else:
            raise ValueError(f"Unknown backend type: {device}")

    def _init_cuda_backend(self):
        """Initialize CUDA backend using CuPy."""
        import cupy as cp

        self.xp = cp
        self.device = cp.cuda.Device()

        # Create a custom linalg module that wraps CuPy's linalg
        class CuPyLinalgWrapper:
            """Wrapper for CuPy linear algebra to ensure consistent API."""

            @staticmethod
            def cholesky(a):
                """Cholesky decomposition (lower triangular)."""
                return cp.linalg.cholesky(a)

            @staticmethod
            def solve(a, b):
                """Solve linear system ax = b."""
                return cp.linalg.solve(a, b)

            @staticmethod
            def norm(x, ord=None):
                """Compute vector or matrix norm."""
                return cp.linalg.norm(x, ord=ord)

            @staticmethod
            def inv(a):
                """Compute matrix inverse."""
                return cp.linalg.inv(a)

        self.linalg = CuPyLinalgWrapper()

    def _init_metal_backend(self):
        """Initialize Metal backend using PyTorch with MPS device."""
        import torch

        self._torch_device = torch.device('mps')

        # Create a NumPy-like wrapper around PyTorch tensors
        class PyTorchArrayWrapper:
            """
            NumPy-compatible wrapper around PyTorch tensors on MPS device.
            This provides a NumPy-like API while leveraging Metal acceleration.
            """

            def __init__(self, torch_module, device):
                self._torch = torch_module
                self._device = device
                # NumPy compatibility: np.newaxis is just None
                self.newaxis = None

            def array(self, obj, dtype=None):
                """
                Create a tensor from array-like object.

                MPS doesn't support float64, so we convert to float32 automatically.
                """
                if isinstance(obj, torch.Tensor):
                    tensor = obj
                else:
                    # Convert to numpy first to check dtype
                    obj_np = np.asarray(obj)
                    if obj_np.dtype == np.float64 and dtype is None:
                        # Auto-convert float64 to float32 for MPS
                        obj_np = obj_np.astype(np.float32)
                    tensor = torch.tensor(obj_np, dtype=self._map_dtype(dtype))
                return tensor.to(self._device)

            def asarray(self, obj, dtype=None):
                """Convert to tensor (similar to np.asarray)."""
                return self.array(obj, dtype=dtype)

            def zeros(self, shape, dtype=float):
                """Create tensor filled with zeros."""
                torch_dtype = self._map_dtype(dtype)
                return torch.zeros(shape, dtype=torch_dtype, device=self._device)

            def ones(self, shape, dtype=float):
                """Create tensor filled with ones."""
                torch_dtype = self._map_dtype(dtype)
                return torch.ones(shape, dtype=torch_dtype, device=self._device)

            def eye(self, n, dtype=float):
                """Create identity matrix."""
                torch_dtype = self._map_dtype(dtype)
                return torch.eye(n, dtype=torch_dtype, device=self._device)

            def empty(self, shape, dtype=float):
                """Create uninitialized tensor."""
                torch_dtype = self._map_dtype(dtype)
                return torch.empty(shape, dtype=torch_dtype, device=self._device)

            def arange(self, *args, **kwargs):
                """Create tensor with evenly spaced values."""
                return torch.arange(*args, **kwargs, device=self._device)

            def linspace(self, start, stop, num):
                """Create tensor with linearly spaced values."""
                return torch.linspace(start, stop, num, device=self._device)

            def exp(self, x):
                """Element-wise exponential."""
                return torch.exp(x)

            def log(self, x):
                """Element-wise natural logarithm."""
                return torch.log(x)

            def abs(self, x):
                """Element-wise absolute value."""
                return torch.abs(x)

            def power(self, x, exponent):
                """Element-wise power."""
                return torch.pow(x, exponent)

            def sqrt(self, x):
                """Element-wise square root."""
                return torch.sqrt(x)

            def sum(self, x, axis=None, keepdims=False):
                """Sum of tensor elements."""
                if axis is None:
                    return torch.sum(x)
                # PyTorch uses 'keepdim' not 'keepdims'
                return torch.sum(x, dim=axis, keepdim=keepdims)

            def dot(self, a, b):
                """Dot product of two tensors."""
                # Handle different dimensions appropriately
                if a.ndim == 1 and b.ndim == 1:
                    return torch.dot(a, b)
                elif a.ndim == 2 and b.ndim == 1:
                    return torch.mv(a, b)
                elif a.ndim == 1 and b.ndim == 2:
                    return torch.matmul(a.unsqueeze(0), b).squeeze(0)
                else:
                    return torch.matmul(a, b)

            def matmul(self, a, b):
                """Matrix multiplication."""
                return torch.matmul(a, b)

            def multiply(self, a, b):
                """Element-wise multiplication."""
                return torch.mul(a, b)

            def diag(self, v):
                """Extract diagonal or construct diagonal matrix."""
                return torch.diag(v)

            def triu(self, m, k=0):
                """Upper triangular part of matrix."""
                return torch.triu(m, diagonal=k)

            def spacing(self, x):
                """
                Distance to nearest floating point number.
                PyTorch doesn't have this, so we approximate with machine epsilon.
                """
                if isinstance(x, (int, float)):
                    return torch.finfo(torch.float64).eps
                return torch.full_like(x, torch.finfo(x.dtype).eps)

            def atleast_2d(self, x):
                """Ensure tensor has at least 2 dimensions."""
                if isinstance(x, torch.Tensor):
                    tensor = x
                else:
                    tensor = torch.tensor(x, device=self._device)

                if tensor.ndim == 0:
                    return tensor.view(1, 1)
                elif tensor.ndim == 1:
                    return tensor.unsqueeze(0)
                return tensor

            def _map_dtype(self, dtype):
                """
                Map NumPy dtypes to PyTorch dtypes.

                Note: MPS (Metal Performance Shaders) doesn't support float64,
                so we default to float32 for Metal backend.
                """
                if dtype is None or dtype == float:
                    return torch.float32  # MPS doesn't support float64
                elif dtype == np.float32 or dtype == 'float32':
                    return torch.float32
                elif dtype == np.float64 or dtype == 'float64':
                    return torch.float32  # Convert float64 -> float32 for MPS
                elif dtype == np.int32 or dtype == 'int32':
                    return torch.int32
                elif dtype == np.int64 or dtype == 'int64':
                    return torch.int64
                else:
                    return torch.float32  # Default to float32 for MPS compatibility

        # Create a linear algebra module for PyTorch
        class PyTorchLinalgWrapper:
            """Linear algebra operations using PyTorch with MPS acceleration."""

            def __init__(self, torch_module, device):
                self._torch = torch_module
                self._device = device

            def cholesky(self, a):
                """Cholesky decomposition (lower triangular).

                Falls back to CPU for MPS since linalg_cholesky isn't supported.
                """
                try:
                    return torch.linalg.cholesky(a)
                except NotImplementedError:
                    # MPS doesn't support cholesky, fall back to CPU
                    result_cpu = torch.linalg.cholesky(a.cpu())
                    return result_cpu.to(self._device)

            def solve(self, a, b):
                """Solve linear system ax = b.

                Falls back to CPU for MPS if linalg_solve isn't supported.
                """
                try:
                    return torch.linalg.solve(a, b)
                except NotImplementedError:
                    # MPS may not support solve, fall back to CPU
                    result_cpu = torch.linalg.solve(a.cpu(), b.cpu())
                    return result_cpu.to(self._device)

            def norm(self, x, ord=None):
                """Compute vector or matrix norm."""
                if ord is None:
                    return torch.linalg.norm(x)
                return torch.linalg.norm(x, ord=ord)

            def inv(self, a):
                """Compute matrix inverse.

                Falls back to CPU for MPS if linalg_inv isn't supported.
                """
                try:
                    return torch.linalg.inv(a)
                except NotImplementedError:
                    # MPS may not support inv, fall back to CPU
                    result_cpu = torch.linalg.inv(a.cpu())
                    return result_cpu.to(self._device)

        self.xp = PyTorchArrayWrapper(torch, self._torch_device)
        self.linalg = PyTorchLinalgWrapper(torch, self._torch_device)
        self.device = self._torch_device

    def _init_cpu_backend(self):
        """Initialize CPU backend using standard NumPy."""
        self.xp = np
        self.linalg = np.linalg
        self.device = None

    def _print_backend_info(self):
        """Print information about the selected backend."""
        print("="*70)
        print("pyKriging GPU Backend Initialization")
        print("="*70)
        print(f"Backend: {self.backend_type.upper()}")

        if self.backend_type == 'cuda':
            device_name = self._cupy.cuda.Device().name.decode()
            device_id = self._cupy.cuda.Device().id
            print(f"Device: {device_name} (ID: {device_id})")

            # Get memory info
            mempool = self._cupy.get_default_memory_pool()
            total_bytes = self._cupy.cuda.Device().mem_info[1]
            total_gb = total_bytes / (1024**3)
            print(f"Total GPU Memory: {total_gb:.2f} GB")

        elif self.backend_type == 'metal':
            print(f"Device: Apple Silicon (Metal Performance Shaders)")
            print(f"PyTorch Version: {self._torch.__version__}")
            print(f"Note: MPS uses float32 (not float64) for all operations")

        elif self.backend_type == 'cpu':
            print(f"Device: CPU (NumPy)")
            print(f"NumPy Version: {np.__version__}")
            if len(self.available_backends) == 1:
                print("Note: No GPU detected. Using CPU for all computations.")
                print("      For GPU acceleration, install:")
                print("      - NVIDIA GPU: pip install cupy")
                print("      - Apple Silicon: pip install torch")

        print(f"Available backends: {', '.join(self.available_backends)}")
        print("="*70 + "\n")

    def to_cpu(self, arr):
        """
        Transfer array from GPU to CPU (returns NumPy array).

        Args:
            arr: Array on GPU or CPU

        Returns:
            NumPy array on CPU
        """
        if arr is None:
            return None

        if self.backend_type == 'cuda':
            if hasattr(arr, 'get'):
                return arr.get()  # CuPy array -> NumPy
            return arr

        elif self.backend_type == 'metal':
            if hasattr(arr, 'cpu'):
                return arr.cpu().numpy()  # PyTorch tensor -> NumPy
            return arr

        else:  # CPU backend
            return np.asarray(arr)

    def to_gpu(self, arr):
        """
        Transfer array from CPU to GPU.

        Args:
            arr: NumPy array or array-like object

        Returns:
            Array on GPU (CuPy array or PyTorch tensor) or NumPy array if CPU backend
        """
        if arr is None:
            return None

        if self.backend_type == 'cuda':
            if not hasattr(arr, 'device'):  # Not already a CuPy array
                return self._cupy.asarray(arr)
            return arr

        elif self.backend_type == 'metal':
            if not hasattr(arr, 'cpu'):  # Not already a PyTorch tensor
                # MPS doesn't support float64, convert to float32
                arr_np = np.asarray(arr)
                if arr_np.dtype == np.float64:
                    arr_np = arr_np.astype(np.float32)
                return self._torch.tensor(arr_np, device=self._torch_device)
            return arr

        else:  # CPU backend
            return np.asarray(arr)

    def is_gpu(self):
        """Check if currently using GPU backend."""
        return self.backend_type in ['cuda', 'metal']

    def get_device_info(self):
        """
        Get detailed information about the current device.

        Returns:
            dict: Dictionary with device information including:
                - backend: Backend type ('cuda', 'metal', or 'cpu')
                - device_name: Name of the device
                - memory_total: Total memory in GB (for GPU backends)
                - memory_used: Used memory in GB (for CUDA backend)
        """
        info = {
            'backend': self.backend_type,
            'device_name': None,
            'memory_total': None,
            'memory_used': None
        }

        if self.backend_type == 'cuda':
            info['device_name'] = self._cupy.cuda.Device().name.decode()
            mem_info = self._cupy.cuda.Device().mem_info
            info['memory_used'] = mem_info[0] / (1024**3)  # Convert to GB
            info['memory_total'] = mem_info[1] / (1024**3)

        elif self.backend_type == 'metal':
            info['device_name'] = 'Apple Silicon (Metal)'
            # PyTorch MPS doesn't expose memory info easily

        elif self.backend_type == 'cpu':
            info['device_name'] = 'CPU'

        return info


# Singleton backend instance management
def get_backend(device: str = 'auto', verbose: bool = None, force_reinit: bool = False) -> GPUBackend:
    """
    Get the global GPU backend instance (singleton pattern).

    This function returns the same backend instance across all calls unless force_reinit=True.
    The backend is initialized on first call and reused subsequently.

    Args:
        device: Backend device to use ('auto', 'cuda', 'metal', or 'cpu')
        verbose: If True, print backend information on initialization
                 If None, uses True for first initialization, False afterwards
        force_reinit: If True, force reinitialization of the backend

    Returns:
        GPUBackend instance

    Example:
        >>> backend = get_backend()
        >>> xp = backend.xp  # NumPy-compatible array module
        >>> linalg = backend.linalg  # Linear algebra module
    """
    global _global_backend

    if _global_backend is None or force_reinit:
        if verbose is None:
            verbose = True  # Print info on first initialization
        _global_backend = GPUBackend(device=device, verbose=verbose)

    return _global_backend


def configure_gpu(device: str = 'auto', verbose: bool = True):
    """
    Configure the GPU backend for pyKriging.

    This function reinitializes the global backend with new settings.
    Call this at the start of your script to configure GPU behavior.

    Args:
        device: Backend device to use:
            - 'auto': Automatically select best available (default)
            - 'cuda': Use NVIDIA GPU with CuPy
            - 'metal': Use Apple Silicon with PyTorch MPS
            - 'cpu': Force CPU computation with NumPy
        verbose: If True, print backend information

    Example:
        >>> import pyKriging
        >>> pyKriging.configure_gpu(device='cuda')  # Force CUDA
        >>> pyKriging.configure_gpu(device='cpu')   # Force CPU

    Raises:
        RuntimeError: If requested backend is not available
    """
    global _global_backend
    _global_backend = GPUBackend(device=device, verbose=verbose)
    return _global_backend


def get_device_info() -> dict:
    """
    Get information about the current GPU/CPU device.

    Returns:
        dict: Device information including backend type, device name, and memory

    Example:
        >>> info = get_device_info()
        >>> print(f"Using {info['backend']} on {info['device_name']}")
    """
    backend = get_backend(verbose=False)
    return backend.get_device_info()


def to_cpu(arr):
    """
    Transfer array from GPU to CPU.

    Args:
        arr: Array on GPU or CPU

    Returns:
        NumPy array on CPU

    Example:
        >>> gpu_array = xp.array([1, 2, 3])
        >>> cpu_array = to_cpu(gpu_array)  # NumPy array
    """
    backend = get_backend(verbose=False)
    return backend.to_cpu(arr)


def to_gpu(arr):
    """
    Transfer array from CPU to GPU.

    Args:
        arr: NumPy array or array-like

    Returns:
        Array on GPU (or CPU if no GPU available)

    Example:
        >>> cpu_array = np.array([1, 2, 3])
        >>> gpu_array = to_gpu(cpu_array)
    """
    backend = get_backend(verbose=False)
    return backend.to_gpu(arr)


# Convenience function for checking if GPU is available
def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available and active.

    Returns:
        bool: True if using GPU backend (CUDA or Metal), False otherwise

    Example:
        >>> if is_gpu_available():
        >>>     print("GPU acceleration enabled!")
    """
    backend = get_backend(verbose=False)
    return backend.is_gpu()
