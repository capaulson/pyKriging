"""
pyKriging - Kriging (Gaussian Process) Metamodeling with GPU Acceleration

This package provides efficient Kriging implementations with automatic GPU acceleration
when CUDA (NVIDIA) or Metal (Apple Silicon) backends are available.

GPU Configuration:
    By default, the best available backend is automatically selected.
    You can manually configure the backend using:

    >>> import pyKriging
    >>> pyKriging.configure_gpu(device='cuda')  # Force CUDA
    >>> pyKriging.configure_gpu(device='cpu')   # Force CPU

    Check current backend:
    >>> pyKriging.get_device_info()

For more information see: https://github.com/capaulson/pyKriging
"""

__author__ = 'chrispaulson'
__version__ = '0.1.0'

# Import main modules
from .krige import *
from .samplingplan import *
from .testfunctions import *
from .utilities import *

# Import GPU backend configuration functions
from .gpu_backend import (
    configure_gpu,
    get_device_info,
    is_gpu_available,
    get_backend,
    to_cpu,
    to_gpu
)

# Expose GPU functions in public API
__all__ = [
    # Core Kriging classes
    'kriging',
    'regression_kriging',
    'coKriging',
    # Sampling plans
    'samplingplan',
    # Test functions
    'testfunctions',
    # GPU configuration
    'configure_gpu',
    'get_device_info',
    'is_gpu_available',
    'get_backend',
    'to_cpu',
    'to_gpu',
]