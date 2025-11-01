#!/usr/bin/env python
"""
Diagnostic test to identify why GPU isn't being used with 100-point model.
"""

import numpy as np
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan

print("="*80)
print("DIAGNOSING 100-POINT MODEL GPU USAGE")
print("="*80)

# Configure GPU
print("\n1. Configuring GPU backend...")
pyKriging.configure_gpu(device='auto', verbose=True)
info = pyKriging.get_device_info()
print(f"   Backend: {info['backend']}")
print(f"   Device: {info['device_name']}")

# Create 100-point dataset
print("\n2. Creating 100-point dataset...")
np.random.seed(42)
sp = samplingplan(2)
X = sp.optimallhc(100)  # 100 points in 2D

def testfun(x):
    return np.sin(x[0] * np.pi) + np.cos(x[1] * np.pi)

y = np.array([testfun(x) for x in X])
print(f"   Dataset: {X.shape[0]} points in {X.shape[1]}D")

# Create model
print("\n3. Creating Kriging model...")
k = kriging(X, y, name='test_100pt')

# Check backend
print(f"\n4. Checking model backend:")
print(f"   Model backend type: {k._backend.backend_type}")
print(f"   Model xp module: {k.xp.__name__ if hasattr(k.xp, '__name__') else type(k.xp).__name__}")

# Check where data actually is
print(f"\n5. Checking data location BEFORE training:")
print(f"   X type: {type(k.X)}")
print(f"   y type: {type(k.y)}")
print(f"   theta type: {type(k.theta)}")
print(f"   pl type: {type(k.pl)}")

# Check if data has device attribute (GPU) or not (CPU)
if hasattr(k.X, 'device'):
    print(f"   X device: {k.X.device}")
if hasattr(k.y, 'device'):
    print(f"   y device: {k.y.device}")

# Try to identify numpy vs cupy vs torch
def identify_array_location(arr, name):
    arr_type = type(arr).__module__ + '.' + type(arr).__name__
    if 'numpy' in arr_type:
        return f"{name}: CPU (NumPy array)"
    elif 'cupy' in arr_type:
        return f"{name}: GPU (CuPy array)"
    elif 'torch' in arr_type:
        device = arr.device if hasattr(arr, 'device') else 'unknown'
        return f"{name}: {device} (PyTorch tensor)"
    else:
        return f"{name}: Unknown ({arr_type})"

print(f"\n6. Data location details:")
print(f"   {identify_array_location(k.X, 'X')}")
print(f"   {identify_array_location(k.y, 'y')}")
print(f"   {identify_array_location(k.theta, 'theta')}")
print(f"   {identify_array_location(k.pl, 'pl')}")

# Check distance matrix
if hasattr(k, 'distance'):
    print(f"   {identify_array_location(k.distance, 'distance')}")

# Check correlation matrix
if hasattr(k, 'Psi'):
    print(f"   {identify_array_location(k.Psi, 'Psi')}")

print("\n7. Starting training (this will be slow if CPU-bound)...")
print("   Watch your Activity Monitor for GPU usage!")
print("   Training...")

import time
start = time.time()
k.train(optimizer='ga')  # Use GA optimizer
elapsed = time.time() - start

print(f"\n8. Training completed in {elapsed:.2f} seconds")

print(f"\n9. Checking data location AFTER training:")
print(f"   {identify_array_location(k.X, 'X')}")
print(f"   {identify_array_location(k.y, 'y')}")
print(f"   {identify_array_location(k.theta, 'theta')}")
print(f"   {identify_array_location(k.Psi, 'Psi')}")
print(f"   {identify_array_location(k.U, 'U')}")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
