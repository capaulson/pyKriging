"""
GPU Data Transfer Verification Script

This script checks if data is actually on the GPU or still on CPU.
"""

import numpy as np
import sys

print("="*80)
print("GPU DATA TRANSFER VERIFICATION")
print("="*80)

# Setup
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan

# Force Metal backend
pyKriging.configure_gpu(device='metal', verbose=True)

# Create small dataset
print("\n[TEST] Creating kriging model...")
np.random.seed(42)
sp = samplingplan(2)
X = sp.optimallhc(20)
y = np.random.rand(20)

k = kriging(X, y, name='transfer_test')

print("\n[CHECK 1] Where is training data?")
print(f"k.X type: {type(k.X)}")
print(f"k.y type: {type(k.y)}")
print(f"k.theta type: {type(k.theta)}")
print(f"k.pl type: {type(k.pl)}")

# Check if it's a PyTorch tensor
import torch
if isinstance(k.X, torch.Tensor):
    print(f"✓ k.X is on GPU: {k.X.device}")
else:
    print(f"✗ k.X is NOT a PyTorch tensor - still on CPU!")
    print(f"  Type: {type(k.X)}")

if isinstance(k.y, torch.Tensor):
    print(f"✓ k.y is on GPU: {k.y.device}")
else:
    print(f"✗ k.y is NOT a PyTorch tensor - still on CPU!")

print("\n[CHECK 2] Where is correlation matrix?")
if hasattr(k, 'Psi'):
    print(f"k.Psi type: {type(k.Psi)}")
    if isinstance(k.Psi, torch.Tensor):
        print(f"✓ k.Psi is on GPU: {k.Psi.device}")
    else:
        print(f"✗ k.Psi is NOT on GPU!")

print("\n[CHECK 3] Where is distance matrix?")
if hasattr(k, 'distance'):
    print(f"k.distance type: {type(k.distance)}")
    if isinstance(k.distance, torch.Tensor):
        print(f"✓ k.distance is on GPU: {k.distance.device}")
    else:
        print(f"✗ k.distance is NOT on GPU!")

print("\n[CHECK 4] What backend is xp using?")
from pyKriging.matrixops import xp
print(f"xp module: {xp}")
print(f"xp type: {type(xp)}")

# Create a test array
test_arr = xp.array([1, 2, 3])
print(f"xp.array([1,2,3]) type: {type(test_arr)}")
if isinstance(test_arr, torch.Tensor):
    print(f"✓ xp creates PyTorch tensors on: {test_arr.device}")
else:
    print(f"✗ xp creates {type(test_arr)} - NOT GPU!")

print("\n[CHECK 5] Test vectorized operations")
try:
    # Test the vectorized distance computation
    X_i = k.X[:, xp.newaxis, :]
    print(f"✓ X_i created, type: {type(X_i)}")

    X_j = k.X[xp.newaxis, :, :]
    print(f"✓ X_j created, type: {type(X_j)}")

    distance_test = xp.abs(X_i - X_j)
    print(f"✓ Distance computed, type: {type(distance_test)}")

    if isinstance(distance_test, torch.Tensor):
        print(f"✓ Distance computation on GPU: {distance_test.device}")
    else:
        print(f"✗ Distance computation NOT on GPU!")

except Exception as e:
    print(f"✗ Error in vectorized operations: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Diagnose the issue
issues = []
if not isinstance(k.X, torch.Tensor):
    issues.append("Training data (X, y) not on GPU - still NumPy arrays")
if hasattr(k, 'distance') and not isinstance(k.distance, torch.Tensor):
    issues.append("Distance matrix not on GPU")
if not isinstance(test_arr, torch.Tensor):
    issues.append("xp.array() creates CPU arrays, not GPU tensors")

if issues:
    print("❌ PROBLEMS FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\n💡 ROOT CAUSE:")
    print("  Data is not actually being transferred to GPU!")
    print("  The to_gpu() calls might not be working correctly.")
else:
    print("✅ All data is on GPU!")
    print("\n💡 If GPU utilization is still low:")
    print("  - Check that operations are actually using GPU tensors")
    print("  - Verify no forced CPU synchronization in loops")
    print("  - Monitor during actual training, not just initialization")
