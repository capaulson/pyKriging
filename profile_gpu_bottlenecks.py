#!/usr/bin/env python
"""
Profile GPU bottlenecks to understand why it's 6x SLOWER for n=100.

This will help identify:
1. CPU-GPU transfer overhead
2. Synchronization points
3. Small tensor overhead
4. Operation-level bottlenecks
"""

import numpy as np
import time
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan


def profile_training(n_samples, device='metal'):
    """Profile a single training run."""
    print(f"\n{'='*80}")
    print(f"PROFILING {n_samples} SAMPLES ON {device.upper()}")
    print(f"{'='*80}")

    # Configure backend
    pyKriging.configure_gpu(device=device, verbose=False)

    # Create dataset
    np.random.seed(42)
    sp = samplingplan(2)
    X = sp.optimallhc(n_samples, n_jobs=1)  # Serial LHC
    y = np.array([np.sin(x[0] * 3 * np.pi) * np.cos(x[1] * 3 * np.pi) for x in X])

    # Create model
    print(f"\n1. Creating model...")
    t0 = time.time()
    k = kriging(X, y, name='profile_test')
    t_create = time.time() - t0
    print(f"   Time: {t_create:.3f}s")

    # Check data location
    backend_type = k._backend.backend_type
    if hasattr(k.X, 'device'):
        data_loc = f"{k.X.device}"
    else:
        data_loc = "CPU (NumPy)"
    print(f"   Backend: {backend_type}")
    print(f"   Data location: {data_loc}")

    # Profile updateModel (called thousands of times during training)
    print(f"\n2. Profiling updateModel() [called ~30,000 times during training]...")
    k.updateData()

    # Time a single updateModel call
    t0 = time.time()
    k.updateModel()
    t_update_model = time.time() - t0
    print(f"   Single call: {t_update_model*1000:.3f}ms")
    print(f"   For 30,000 calls: {t_update_model*30000:.1f}s")

    # Profile neglikelihood (called after each updateModel)
    print(f"\n3. Profiling neglikelihood() [called ~30,000 times during training]...")
    t0 = time.time()
    k.neglikelihood()
    t_neglike = time.time() - t0
    likelihood_value = k.NegLnLike
    print(f"   Single call: {t_neglike*1000:.3f}ms")
    print(f"   For 30,000 calls: {t_neglike*30000:.1f}s")
    print(f"   Likelihood value type: {type(likelihood_value)}")

    # Profile update() (sets new hyperparameters)
    print(f"\n4. Profiling update() [called ~30,000 times during training]...")
    new_params = np.array([1.0, 1.0, 2.0, 2.0])  # theta + pl
    t0 = time.time()
    k.update(new_params)
    t_update = time.time() - t0
    print(f"   Single call: {t_update*1000:.3f}ms")
    print(f"   For 30,000 calls: {t_update*30000:.1f}s")

    # Total estimated time for training
    t_per_iteration = t_update + t_neglike
    estimated_training_time = t_per_iteration * 30000
    print(f"\n5. ESTIMATED TRAINING TIME:")
    print(f"   Time per iteration: {t_per_iteration*1000:.3f}ms")
    print(f"   Total for 30,000 iterations: {estimated_training_time:.1f}s")

    # Profile prediction
    print(f"\n6. Profiling prediction...")
    test_point = [0.5, 0.5]

    # First prediction (may include warmup)
    t0 = time.time()
    pred1 = k.predict(test_point)
    t_pred_first = time.time() - t0
    print(f"   First prediction: {t_pred_first*1000:.3f}ms")

    # Subsequent predictions (warmed up)
    times = []
    for _ in range(10):
        t0 = time.time()
        pred = k.predict(test_point)
        times.append(time.time() - t0)
    t_pred_avg = np.mean(times)
    print(f"   Average prediction: {t_pred_avg*1000:.3f}ms")
    print(f"   For 1000 predictions: {t_pred_avg*1000:.1f}s")

    # Breakdown by operation
    print(f"\n7. OPERATION BREAKDOWN:")
    print(f"   updateModel: {t_update_model*1000:.3f}ms ({t_update_model/t_per_iteration*100:.1f}%)")
    print(f"   neglikelihood: {t_neglike*1000:.3f}ms ({t_neglike/t_per_iteration*100:.1f}%)")
    print(f"   update: {t_update*1000:.3f}ms ({t_update/t_per_iteration*100:.1f}%)")

    # Identify bottleneck
    if t_update > t_neglike and t_update > t_update_model:
        bottleneck = "update() - likely CPU-GPU transfer overhead"
    elif t_neglike > t_update_model:
        bottleneck = "neglikelihood() - likely scalar extraction overhead"
    else:
        bottleneck = "updateModel() - likely Cholesky decomposition"

    print(f"\n8. PRIMARY BOTTLENECK: {bottleneck}")

    return {
        'n': n_samples,
        'device': device,
        'backend': backend_type,
        'data_loc': data_loc,
        't_update_model': t_update_model,
        't_neglike': t_neglike,
        't_update': t_update,
        't_per_iteration': t_per_iteration,
        'estimated_training': estimated_training_time,
        't_prediction': t_pred_avg,
        'bottleneck': bottleneck
    }


def main():
    """Main profiling function."""
    print("="*80)
    print("GPU BOTTLENECK PROFILING")
    print("="*80)
    print("\nThis will identify why GPU is 6x SLOWER for n=100")
    print("="*80)

    # Profile different sizes
    sizes = [50, 100, 250]

    results = []
    for n in sizes:
        # Profile Metal (GPU)
        result_gpu = profile_training(n, device='metal')
        results.append(result_gpu)

        # Profile CPU for comparison
        result_cpu = profile_training(n, device='cpu')
        results.append(result_cpu)

        # Compare
        print(f"\n{'='*80}")
        print(f"COMPARISON FOR n={n}")
        print(f"{'='*80}")
        speedup = result_cpu['t_per_iteration'] / result_gpu['t_per_iteration']
        print(f"Time per iteration:")
        print(f"  CPU:    {result_cpu['t_per_iteration']*1000:.3f}ms")
        print(f"  Metal:  {result_gpu['t_per_iteration']*1000:.3f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        if speedup < 1.0:
            print(f"\n⚠️  GPU is {1/speedup:.2f}x SLOWER!")
            print(f"   Bottleneck: {result_gpu['bottleneck']}")
        else:
            print(f"\n✓ GPU is {speedup:.2f}x faster")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("\nKey findings:")
    print("1. Where is most time spent?")
    print("2. What operations are slow on GPU?")
    print("3. Are there excessive CPU-GPU transfers?")
    print("4. Is scalar extraction a bottleneck?")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
