#!/usr/bin/env python
"""
Direct PyTorch Metal Test

This tests if PyTorch can actually use Metal/MPS device.
Run this FIRST before the main benchmark.
"""

import sys

print("=" * 80)
print("DIRECT PYTORCH METAL TEST")
print("=" * 80)

# Test 1: Import PyTorch
print("\n1. Testing PyTorch import...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} imported successfully")
except ImportError:
    print("❌ PyTorch not installed!")
    print("   Install with: pip install torch")
    sys.exit(1)

# Test 2: Check MPS availability
print("\n2. Checking MPS (Metal) availability...")
if hasattr(torch.backends, 'mps'):
    print(f"   MPS backend available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print("   ✓ Metal/MPS is available!")
    else:
        print("   ❌ MPS backend exists but not available")
        print("      Are you on Apple Silicon (M1/M2/M3)?")
        sys.exit(1)
else:
    print("   ❌ MPS backend not found in this PyTorch version")
    print(f"      Your PyTorch version: {torch.__version__}")
    print("      You need PyTorch 1.12+ with MPS support")
    sys.exit(1)

# Test 3: Create tensor on MPS
print("\n3. Creating tensor on MPS device...")
try:
    device = torch.device('mps')
    x = torch.randn(1000, 1000, device=device)
    print(f"   ✓ Created tensor on {x.device}")
except Exception as e:
    print(f"   ❌ Failed to create MPS tensor: {e}")
    sys.exit(1)

# Test 4: Matrix multiplication on MPS
print("\n4. Testing matrix multiplication on MPS...")
try:
    import time
    n = 2000
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    # Warm up
    c = torch.matmul(a, b)
    torch.mps.synchronize()

    # Benchmark
    start = time.time()
    c = torch.matmul(a, b)
    torch.mps.synchronize()  # Wait for GPU to finish
    elapsed = time.time() - start

    print(f"   ✓ {n}x{n} matrix multiplication: {elapsed:.4f} seconds")
    print(f"     (Expected: <0.1s on Metal, >1s on CPU)")

    if elapsed < 0.5:
        print("   ✓ Looks like Metal is working!")
    else:
        print("   ⚠ Slower than expected, might be running on CPU")
except Exception as e:
    print(f"   ❌ Matrix multiplication failed: {e}")
    sys.exit(1)

# Test 5: Cholesky decomposition (key operation for Kriging)
print("\n5. Testing Cholesky decomposition on MPS...")
try:
    n = 500
    # Create positive definite matrix
    A = torch.randn(n, n, device=device)
    A = torch.matmul(A, A.T) + torch.eye(n, device=device) * 0.1

    start = time.time()
    L = torch.linalg.cholesky(A)
    torch.mps.synchronize()
    elapsed = time.time() - start

    print(f"   ✓ {n}x{n} Cholesky: {elapsed:.4f} seconds")
    print(f"     (This is the key operation in Kriging training)")

    if elapsed < 0.2:
        print("   ✓ Cholesky is GPU-accelerated!")
    else:
        print("   ⚠ Slower than expected")
except Exception as e:
    print(f"   ❌ Cholesky failed: {e}")
    sys.exit(1)

# Test 6: Check if operations actually use GPU
print("\n6. GPU utilization test...")
print("   Running compute-intensive loop...")
print("   ** CHECK ACTIVITY MONITOR NOW **")
print("   Window → GPU History - you should see GPU usage spike\n")

input("Press ENTER to start GPU stress test...")

try:
    for i in range(50):
        n = 1000
        A = torch.randn(n, n, device=device)
        B = torch.randn(n, n, device=device)
        C = torch.matmul(A, B)
        L = torch.linalg.cholesky(C @ C.T + torch.eye(n, device=device) * 0.1)
        print(f"   Iteration {i+1}/50", end='\r')
    torch.mps.synchronize()
    print("\n   ✓ Stress test completed")
    print("\n   Did you see GPU usage spike in Activity Monitor?")
    response = input("   (y/n): ")

    if response.lower() == 'y':
        print("\n   ✓ PyTorch Metal is working correctly!")
    else:
        print("\n   ⚠ PyTorch Metal might not be working as expected")
except Exception as e:
    print(f"\n   ❌ Stress test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("✓ PyTorch is installed and functional")
print("✓ MPS (Metal) backend is available")
print("✓ Tensors can be created on MPS device")
print("✓ Matrix operations run on MPS")
print("✓ Cholesky decomposition works on MPS")
print("\nIf you saw GPU usage during the stress test,")
print("then PyTorch Metal is working correctly.")
print("\nNow run: python benchmark_metal.py")
print("=" * 80)
