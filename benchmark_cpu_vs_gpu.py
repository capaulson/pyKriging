#!/usr/bin/env python
"""
CPU vs GPU Benchmark Script

Compares kriging model training and prediction performance between
CPU and GPU (Metal) backends across different sample sizes.

Sample sizes tested: 10, 50, 100, 250, 500, 750, 1000
"""

import numpy as np
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan
import time
import matplotlib.pyplot as plt
import os

# Enable MPS fallback for operations not yet implemented on Metal
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def testfun(x):
    """Test function: 2D sinusoidal function"""
    return np.sin(x[0] * 3 * np.pi) * np.cos(x[1] * 3 * np.pi)


def benchmark_training(X, y, backend='cpu', name='benchmark'):
    """Benchmark training time for a given backend"""
    # Configure backend
    if backend == 'cpu':
        pyKriging.configure_gpu(device='cpu', verbose=False)
    else:  # metal
        pyKriging.configure_gpu(device='metal', verbose=False)

    # Create and train model
    k = kriging(X, y, name=name)

    start_time = time.time()
    k.train(optimizer='ga')
    training_time = time.time() - start_time

    return training_time, k


def benchmark_predictions(model, test_points):
    """Benchmark prediction time"""
    start_time = time.time()
    predictions = [model.predict(pt) for pt in test_points]
    prediction_time = time.time() - start_time

    return prediction_time, predictions


if __name__ == '__main__':
    print("=" * 80)
    print("CPU vs GPU (Metal) BENCHMARK")
    print("=" * 80)

    # Sample sizes to test
    sample_sizes = [10, 50, 100, 250, 500, 750, 1000]
    n_test_points = 100  # Fixed number of test points for prediction

    # Results storage
    results = {
        'sample_sizes': sample_sizes,
        'cpu_training_times': [],
        'gpu_training_times': [],
        'cpu_prediction_times': [],
        'gpu_prediction_times': [],
        'speedups_training': [],
        'speedups_prediction': []
    }

    # Setup sampling plan
    np.random.seed(42)
    sp = samplingplan(2)

    print(f"\nTesting sample sizes: {sample_sizes}")
    print(f"Number of test points for prediction: {n_test_points}")
    print(f"Dimensions: 2D")
    print()

    for n_samples in sample_sizes:
        print("=" * 80)
        print(f"BENCHMARK: {n_samples} training samples")
        print("=" * 80)

        # Generate training data
        X = sp.optimallhc(n_samples)
        y = np.array([testfun(x) for x in X])

        # Generate test points
        test_points = sp.rlh(n_test_points)

        # CPU Benchmark
        print(f"\n1. CPU Backend ({n_samples} samples)")
        print("   Training...")
        cpu_train_time, cpu_model = benchmark_training(X, y, backend='cpu',
                                                        name=f'cpu_{n_samples}')
        print(f"   ✓ Training time: {cpu_train_time:.2f} seconds")

        print("   Predicting...")
        cpu_pred_time, _ = benchmark_predictions(cpu_model, test_points)
        print(f"   ✓ Prediction time: {cpu_pred_time:.3f} seconds "
              f"({cpu_pred_time/n_test_points*1000:.2f} ms/prediction)")

        # GPU Benchmark
        print(f"\n2. GPU (Metal) Backend ({n_samples} samples)")
        print("   Training...")
        gpu_train_time, gpu_model = benchmark_training(X, y, backend='metal',
                                                        name=f'gpu_{n_samples}')
        print(f"   ✓ Training time: {gpu_train_time:.2f} seconds")

        print("   Predicting...")
        gpu_pred_time, _ = benchmark_predictions(gpu_model, test_points)
        print(f"   ✓ Prediction time: {gpu_pred_time:.3f} seconds "
              f"({gpu_pred_time/n_test_points*1000:.2f} ms/prediction)")

        # Calculate speedups
        train_speedup = cpu_train_time / gpu_train_time
        pred_speedup = cpu_pred_time / gpu_pred_time

        print(f"\n3. Speedup Analysis")
        print(f"   Training speedup: {train_speedup:.2f}x")
        print(f"   Prediction speedup: {pred_speedup:.2f}x")

        # Store results
        results['cpu_training_times'].append(cpu_train_time)
        results['gpu_training_times'].append(gpu_train_time)
        results['cpu_prediction_times'].append(cpu_pred_time)
        results['gpu_prediction_times'].append(gpu_pred_time)
        results['speedups_training'].append(train_speedup)
        results['speedups_prediction'].append(pred_speedup)

        print()

    # ============================================================================
    # Plot Results
    # ============================================================================
    print("=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CPU vs GPU (Metal) Benchmark Results', fontsize=16, fontweight='bold')

    # Plot 1: Training Time Comparison
    ax1 = axes[0, 0]
    ax1.plot(sample_sizes, results['cpu_training_times'], 'o-',
             label='CPU', linewidth=2, markersize=8)
    ax1.plot(sample_sizes, results['gpu_training_times'], 's-',
             label='GPU (Metal)', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Training Samples', fontsize=12)
    ax1.set_ylabel('Training Time (seconds)', fontsize=12)
    ax1.set_title('Training Time vs Sample Size', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Plot 2: Prediction Time Comparison
    ax2 = axes[0, 1]
    ax2.plot(sample_sizes, results['cpu_prediction_times'], 'o-',
             label='CPU', linewidth=2, markersize=8)
    ax2.plot(sample_sizes, results['gpu_prediction_times'], 's-',
             label='GPU (Metal)', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Training Samples', fontsize=12)
    ax2.set_ylabel(f'Prediction Time for {n_test_points} points (seconds)', fontsize=12)
    ax2.set_title('Prediction Time vs Sample Size', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    # Plot 3: Training Speedup
    ax3 = axes[1, 0]
    ax3.plot(sample_sizes, results['speedups_training'], 'o-',
             linewidth=2, markersize=8, color='green')
    ax3.axhline(y=1.0, color='red', linestyle='--', label='No speedup', alpha=0.5)
    ax3.set_xlabel('Number of Training Samples', fontsize=12)
    ax3.set_ylabel('Speedup (CPU time / GPU time)', fontsize=12)
    ax3.set_title('Training Speedup (GPU vs CPU)', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')

    # Plot 4: Prediction Speedup
    ax4 = axes[1, 1]
    ax4.plot(sample_sizes, results['speedups_prediction'], 's-',
             linewidth=2, markersize=8, color='purple')
    ax4.axhline(y=1.0, color='red', linestyle='--', label='No speedup', alpha=0.5)
    ax4.set_xlabel('Number of Training Samples', fontsize=12)
    ax4.set_ylabel('Speedup (CPU time / GPU time)', fontsize=12)
    ax4.set_title('Prediction Speedup (GPU vs CPU)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')

    plt.tight_layout()

    # Save plot
    plot_filename = 'cpu_vs_gpu_benchmark.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_filename}")

    plt.show()

    # ============================================================================
    # Summary Statistics
    # ============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    avg_train_speedup = np.mean(results['speedups_training'])
    avg_pred_speedup = np.mean(results['speedups_prediction'])
    max_train_speedup = np.max(results['speedups_training'])
    max_pred_speedup = np.max(results['speedups_prediction'])

    print(f"\nAverage training speedup: {avg_train_speedup:.2f}x")
    print(f"Maximum training speedup: {max_train_speedup:.2f}x (at {sample_sizes[np.argmax(results['speedups_training'])]} samples)")
    print(f"\nAverage prediction speedup: {avg_pred_speedup:.2f}x")
    print(f"Maximum prediction speedup: {max_pred_speedup:.2f}x (at {sample_sizes[np.argmax(results['speedups_prediction'])]} samples)")

    print("\n" + "=" * 80)
    print("DETAILED RESULTS TABLE")
    print("=" * 80)
    print(f"\n{'Samples':<10} {'CPU Train':<12} {'GPU Train':<12} {'Speedup':<10} "
          f"{'CPU Pred':<12} {'GPU Pred':<12} {'Speedup':<10}")
    print("-" * 78)

    for i, n in enumerate(sample_sizes):
        print(f"{n:<10} "
              f"{results['cpu_training_times'][i]:<12.2f} "
              f"{results['gpu_training_times'][i]:<12.2f} "
              f"{results['speedups_training'][i]:<10.2f} "
              f"{results['cpu_prediction_times'][i]:<12.3f} "
              f"{results['gpu_prediction_times'][i]:<12.3f} "
              f"{results['speedups_prediction'][i]:<10.2f}")

    print("\n" + "=" * 80)
