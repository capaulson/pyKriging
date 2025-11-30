"""
Pytest configuration and fixtures for pyKriging tests.
"""
import pytest
import numpy as np


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def simple_2d_function():
    """Simple 2D test function: sin(x) * cos(y)."""
    def func(x):
        return np.sin(x[0] * np.pi) * np.cos(x[1] * np.pi)
    return func


@pytest.fixture
def complex_2d_function():
    """More complex 2D test function for accuracy testing."""
    def func(x):
        return np.sin(x[0] * 3 * np.pi) * np.cos(x[1] * 3 * np.pi)
    return func


@pytest.fixture
def branin_function():
    """Branin test function - common optimization benchmark."""
    def func(x):
        # Branin function (scaled to [0,1]^2)
        x1 = x[0] * 15 - 5  # Scale to [-5, 10]
        x2 = x[1] * 15      # Scale to [0, 15]
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        return a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s
    return func
