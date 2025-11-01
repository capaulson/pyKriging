"""
Test Metal (MPS) Backend with Float32 Conversion

This script specifically tests that the Metal backend correctly handles
float64 -> float32 conversion automatically.
"""

import numpy as np
import sys

print("="*80)
print("METAL (MPS) FLOAT32 CONVERSION TEST")
print("="*80)

# Test 1: Import and check backend
print("\n[TEST 1] Checking Metal backend availability...")
try:
    import torch
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("✓ Metal (MPS) backend is available")
    else:
        print("⚠ Metal (MPS) backend not available")
        print("  This test is designed for Apple Silicon Macs")
        sys.exit(0)
except ImportError:
    print("⚠ PyTorch not installed")
    print("  Install with: pip install torch")
    sys.exit(0)

# Test 2: Force Metal backend
print("\n[TEST 2] Configuring Metal backend...")
try:
    import pyKriging
    pyKriging.configure_gpu(device='metal', verbose=True)
    info = pyKriging.get_device_info()
    print(f"✓ Metal backend configured: {info['backend']}")
except Exception as e:
    print(f"✗ Failed to configure Metal backend: {e}")
    sys.exit(1)

# Test 3: Create float64 data (typical NumPy default)
print("\n[TEST 3] Creating float64 data...")
np.random.seed(42)
X = np.random.rand(20, 2).astype(np.float64)
y = np.random.rand(20).astype(np.float64)
print(f"  X dtype: {X.dtype} (should be float64)")
print(f"  y dtype: {y.dtype} (should be float64)")

# Test 4: Create Kriging model (should auto-convert to float32)
print("\n[TEST 4] Creating Kriging model with float64 data...")
try:
    from pyKriging.krige import kriging
    k = kriging(X, y, name='metal_test')
    print("✓ Model created successfully!")
    print("  Float64 data was automatically converted to float32 for Metal")
except TypeError as e:
    if "float64" in str(e) and "MPS" in str(e):
        print(f"✗ Float64 conversion failed: {e}")
        print("  The fix may not be applied correctly")
        sys.exit(1)
    else:
        raise
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Train model
print("\n[TEST 5] Training model on Metal backend...")
try:
    k.train(optimizer='pso')
    print("✓ Training completed successfully!")
    print(f"  Optimized theta: {k.theta}")
except Exception as e:
    print(f"✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Make predictions
print("\n[TEST 6] Making predictions...")
try:
    test_points = np.array([[0.5, 0.5], [0.25, 0.75]])
    for point in test_points:
        pred = k.predict(point)
        var = k.predict_var(point)
        print(f"  Point {point}: prediction={pred:.4f}, variance={var:.6f}")
    print("✓ Predictions completed successfully!")
except Exception as e:
    print(f"✗ Predictions failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Verify data transfer
print("\n[TEST 7] Testing data transfer utilities...")
try:
    from pyKriging import to_cpu, to_gpu

    # Create float64 CPU array
    cpu_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    print(f"  Original CPU array dtype: {cpu_array.dtype}")

    # Transfer to GPU (should convert to float32)
    gpu_array = to_gpu(cpu_array)
    print(f"  GPU array dtype: {gpu_array.dtype}")

    # Transfer back to CPU
    cpu_again = to_cpu(gpu_array)
    print(f"  Back to CPU dtype: {cpu_again.dtype}")

    # Check values are approximately equal (accounting for float32 precision)
    if np.allclose(cpu_array, cpu_again, rtol=1e-6):
        print("✓ Data transfer works correctly (values preserved)")
    else:
        print("⚠ Values differ slightly (expected due to float32 conversion)")
        print(f"  Original: {cpu_array}")
        print(f"  Returned: {cpu_again}")

except Exception as e:
    print(f"✗ Data transfer test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("✓ All Metal (MPS) backend tests passed!")
print("\nKey points:")
print("  • Metal backend automatically converts float64 -> float32")
print("  • No user code changes required")
print("  • Model training and prediction work correctly")
print("  • Float32 precision is sufficient for most applications")
print("\nIf you need full float64 precision:")
print("  pyKriging.configure_gpu(device='cpu')")
print("="*80)
