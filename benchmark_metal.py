#!/usr/bin/env python
"""
Metal GPU Benchmark and Diagnostic Script

This script will help diagnose why GPU isn't being utilized.
Run this on your Mac and share the output.

INSTRUCTIONS:
1. Open Activity Monitor → Window → GPU History BEFORE running
2. Run this script: python benchmark_metal.py
3. Watch GPU usage in Activity Monitor during "TRAINING PHASE"
4. Share the complete output

Expected GPU usage: 50-80% during training phase
If seeing <10% GPU usage, there's still a problem.
"""

import numpy as np
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan
import time
import sys

def check_array_type(arr, name):
    """Helper function to identify array type and device location"""
    mod = type(arr).__module__
    cls = type(arr).__name__

    if 'torch' in mod:
        device = arr.device if hasattr(arr, 'device') else 'unknown'
        return f"{name}: {mod}.{cls} on {device}"
    elif 'numpy' in mod:
        return f"{name}: {mod}.{cls} (CPU)"
    elif 'cupy' in mod:
        return f"{name}: {mod}.{cls} (GPU)"
    else:
        return f"{name}: {mod}.{cls}"

def testfun(x):
    """Test function for kriging"""
    return np.sin(x[0] * 3 * np.pi) * np.cos(x[1] * 3 * np.pi)

if __name__ == '__main__':
    print("=" * 80)
    print("METAL GPU BENCHMARK AND DIAGNOSTIC")
    print("=" * 80)

    # ============================================================================
    # PHASE 1: Backend Configuration Check
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: Backend Configuration")
    print("=" * 80)

    print("\n1.1 Configuring GPU backend to 'metal'...")
    pyKriging.configure_gpu(device='metal', verbose=True)

    print("\n1.2 Getting device info...")
    info = pyKriging.get_device_info()
    print(f"   Backend type: {info['backend']}")
    print(f"   Device name: {info['device_name']}")

    if info['backend'] != 'metal':
        print("\n❌ WARNING: Backend is not 'metal'!")
        print("   This explains why GPU isn't being used.")
        print("   Is PyTorch installed? Try: pip install torch")
        sys.exit(1)
    else:
        print("\n✓ Backend correctly set to Metal")

    # ============================================================================
    # PHASE 2: Create 100-Point Dataset
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: Creating 100-Point Dataset")
    print("=" * 80)

    np.random.seed(42)
    sp = samplingplan(2)
    X = sp.optimallhc(1000)  # 100 points in 2D

    y = np.array([testfun(x) for x in X])
    print(f"\n✓ Created dataset: {X.shape[0]} points in {X.shape[1]}D")

    # ============================================================================
    # PHASE 3: Create Kriging Model
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: Creating Kriging Model")
    print("=" * 80)

    print("\n3.1 Creating kriging instance...")
    k = kriging(X, y, name='metal_benchmark')

    print("\n3.2 Checking model internals...")
    print(f"   Model backend type: {k._backend.backend_type}")

    # Check what xp actually is
    xp_module = k.xp.__class__.__module__
    xp_class = k.xp.__class__.__name__
    print(f"   Model xp: {xp_module}.{xp_class}")

    print("\n3.3 Data location check:")
    print(f"   {check_array_type(k.X, 'X')}")
    print(f"   {check_array_type(k.y, 'y')}")
    print(f"   {check_array_type(k.theta, 'theta')}")
    print(f"   {check_array_type(k.pl, 'pl')}")
    print(f"   {check_array_type(k.Psi, 'Psi')}")

    # Critical check: Are arrays actually on MPS device?
    if hasattr(k.X, 'device'):
        if 'mps' in str(k.X.device):
            print("\n✓ Data is on MPS (Metal) device!")
        else:
            print(f"\n❌ WARNING: Data is on {k.X.device}, not MPS!")
    else:
        print("\n❌ WARNING: Data has no 'device' attribute - likely CPU NumPy array!")

    # ============================================================================
    # PHASE 4: Training (Watch Activity Monitor GPU Usage!)
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: TRAINING PHASE")
    print("=" * 80)
    print("\n*** WATCH YOUR ACTIVITY MONITOR GPU USAGE NOW ***")
    print("    Window → GPU History")
    print("    You should see GPU usage spike to 50-80%")
    print()
    input("Press ENTER when ready to start training...")

    print("\n4.1 Training model...")
    print("    This will call neglikelihood() thousands of times")
    print("    Each call does Cholesky decomposition (O(n³))")
    print("    This MUST run on GPU for good performance")
    print()

    start_time = time.time()
    k.train(optimizer='ga')
    training_time = time.time() - start_time

    print(f"\n✓ Training completed in {training_time:.2f} seconds")

    # ============================================================================
    # PHASE 5: Post-Training Analysis
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 5: Post-Training Analysis")
    print("=" * 80)

    print("\n5.1 Data location after training:")
    print(f"   {check_array_type(k.X, 'X')}")
    print(f"   {check_array_type(k.Psi, 'Psi')}")
    print(f"   {check_array_type(k.U, 'U')}")

    print("\n5.2 Model parameters:")
    print(f"   theta: {k.theta}")
    print(f"   p: {k.pl}")

    # ============================================================================
    # PHASE 6: Prediction Benchmark
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 6: Prediction Benchmark")
    print("=" * 80)

    print("\n6.1 Making 1000 predictions...")
    test_points = sp.rlh(1000)

    start_time = time.time()
    predictions = [k.predict(pt) for pt in test_points]
    prediction_time = time.time() - start_time

    print(f"✓ 1000 predictions in {prediction_time:.2f} seconds")
    print(f"  Average: {prediction_time/1000*1000:.2f} ms per prediction")

    # ============================================================================
    # FINAL DIAGNOSIS
    # ============================================================================
    print("\n" + "=" * 80)
    print("FINAL DIAGNOSIS")
    print("=" * 80)

    print("\nPlease answer these questions:")
    print("1. During Phase 4 (training), what was the GPU usage in Activity Monitor?")
    print("   a) < 10% → Still CPU-bound, problem NOT fixed")
    print("   b) 10-40% → Partially using GPU, needs investigation")
    print("   c) 50-80% → SUCCESS! GPU properly utilized")
    print()
    print("2. Was training time reasonable?")
    print(f"   Training time: {training_time:.2f} seconds for 100 points")
    print("   Expected on Metal: 10-30 seconds")
    print("   Expected on CPU only: 60-120 seconds")
    print()

    if hasattr(k.X, 'device') and 'mps' in str(k.X.device):
        print("✓ Data is on Metal device")
    else:
        print("❌ Data is NOT on Metal device - this is the problem!")

    print("\n" + "=" * 80)
    print("Please share this entire output + your Activity Monitor observations")
    print("=" * 80)
