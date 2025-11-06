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
import os
import matplotlib.pyplot as plt
from matplotlib import cm

# Enable MPS fallback for operations not yet implemented on Metal
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

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
    X = sp.optimallhc(100)  # 100 points in 2D

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
    # PHASE 7: Visualize Trained Model
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 7: Model Visualization")
    print("=" * 80)

    print("\n7.1 Creating prediction grid...")
    # Create a dense grid for visualization
    grid_resolution = 50
    x1_grid = np.linspace(0, 1, grid_resolution)
    x2_grid = np.linspace(0, 1, grid_resolution)
    X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)

    # Generate predictions on the grid
    grid_points = np.column_stack([X1_mesh.ravel(), X2_mesh.ravel()])
    print(f"   Predicting on {len(grid_points)} grid points...")

    grid_predictions = np.array([k.predict(pt) for pt in grid_points])
    Z_pred = grid_predictions.reshape(X1_mesh.shape)

    # Also compute true values for comparison
    Z_true = np.array([testfun(pt) for pt in grid_points]).reshape(X1_mesh.shape)

    print("   ✓ Predictions complete")

    print("\n7.2 Generating plots...")
    fig = plt.figure(figsize=(16, 5))

    # Plot 1: True function
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X1_mesh, X2_mesh, Z_true, cmap=cm.viridis,
                              alpha=0.8, linewidth=0, antialiased=True)
    ax1.scatter(X[:, 0], X[:, 1], y, c='red', marker='o', s=50, label='Training points')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('f(x)')
    ax1.set_title('True Function', fontweight='bold')
    ax1.legend()
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    # Plot 2: Kriging predictions
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X1_mesh, X2_mesh, Z_pred, cmap=cm.viridis,
                              alpha=0.8, linewidth=0, antialiased=True)
    ax2.scatter(X[:, 0], X[:, 1], y, c='red', marker='o', s=50, label='Training points')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('f(x)')
    ax2.set_title('Kriging Prediction (Metal GPU)', fontweight='bold')
    ax2.legend()
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    # Plot 3: Error
    ax3 = fig.add_subplot(133, projection='3d')
    error = np.abs(Z_true - Z_pred)
    surf3 = ax3.plot_surface(X1_mesh, X2_mesh, error, cmap=cm.hot,
                              alpha=0.8, linewidth=0, antialiased=True)
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_zlabel('|Error|')
    ax3.set_title('Absolute Error', fontweight='bold')
    fig.colorbar(surf3, ax=ax3, shrink=0.5)

    plt.tight_layout()

    # Save plot
    plot_filename = 'metal_benchmark_model.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"   ✓ Plot saved to: {plot_filename}")

    # Calculate error statistics
    mae = np.mean(np.abs(Z_true - Z_pred))
    rmse = np.sqrt(np.mean((Z_true - Z_pred)**2))
    max_error = np.max(np.abs(Z_true - Z_pred))

    print(f"\n7.3 Prediction accuracy:")
    print(f"   Mean Absolute Error (MAE): {mae:.6f}")
    print(f"   Root Mean Square Error (RMSE): {rmse:.6f}")
    print(f"   Maximum Error: {max_error:.6f}")

    plt.show()

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
