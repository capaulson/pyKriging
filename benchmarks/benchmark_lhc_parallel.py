#!/usr/bin/env python
"""
Benchmark script comparing serial vs parallel optimal LHC generation.

Tests the speedup achieved by parallelizing q-value optimization.
"""

import time
import numpy as np
from multiprocessing import cpu_count

# Import both versions
from pyKriging.samplingplan import samplingplan
from pyKriging.samplingplan_parallel import samplingplan_parallel

print("=" * 80)
print("OPTIMAL LATIN HYPERCUBE: PARALLEL vs SERIAL BENCHMARK")
print("=" * 80)

# Test configurations
test_configs = [
    {"name": "Small (20 points)", "n": 20, "k": 2},
    {"name": "Medium (50 points)", "n": 50, "k": 2},
    {"name": "Large (100 points)", "n": 100, "k": 2},
]

print(f"\nSystem info:")
print(f"  CPU cores available: {cpu_count()}")
print(f"  Parallel workers: {min(7, cpu_count())} (for 7 q values)")

for config in test_configs:
    print("\n" + "=" * 80)
    print(f"TEST: {config['name']}")
    print("=" * 80)

    n = config['n']
    k = config['k']

    # Original serial version
    print(f"\n1. Original serial version:")
    sp_serial = samplingplan(k=k)
    np.random.seed(42)
    start = time.time()
    X_serial = sp_serial.optimallhc(n, population=30, iterations=30)
    time_serial = time.time() - start
    print(f"   Time: {time_serial:.2f} seconds")

    # New parallel version (serial mode for comparison)
    print(f"\n2. New version (serial mode, n_jobs=1):")
    sp_parallel = samplingplan_parallel(k=k)
    np.random.seed(42)
    start = time.time()
    X_parallel_serial = sp_parallel.optimallhc(n, population=30, iterations=30, n_jobs=1)
    time_parallel_serial = time.time() - start
    print(f"   Time: {time_parallel_serial:.2f} seconds")

    # Check if vectorized jd() provides speedup
    if time_parallel_serial < time_serial:
        jd_speedup = time_serial / time_parallel_serial
        print(f"   ✓ Vectorized jd() speedup: {jd_speedup:.2f}x")
    else:
        print(f"   (No speedup from vectorization alone)")

    # New parallel version (full parallelization)
    print(f"\n3. New version (parallel mode, n_jobs=-1):")
    sp_parallel = samplingplan_parallel(k=k)
    np.random.seed(42)
    start = time.time()
    X_parallel = sp_parallel.optimallhc(n, population=30, iterations=30, n_jobs=-1)
    time_parallel = time.time() - start
    print(f"   Time: {time_parallel:.2f} seconds")

    # Calculate speedups
    speedup_vs_serial = time_serial / time_parallel
    speedup_vs_parallel_serial = time_parallel_serial / time_parallel

    print(f"\n   RESULTS:")
    print(f"   ✓ Speedup vs original: {speedup_vs_serial:.2f}x")
    print(f"   ✓ Speedup from parallelization: {speedup_vs_parallel_serial:.2f}x")
    print(f"   ✓ Time saved: {time_serial - time_parallel:.2f} seconds")

    # Theoretical max speedup
    n_workers = min(7, cpu_count())
    theoretical_max = min(n_workers, 7)
    efficiency = (speedup_vs_parallel_serial / theoretical_max) * 100
    print(f"   ✓ Parallel efficiency: {efficiency:.1f}% (vs theoretical max {theoretical_max}x)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nOptimizations implemented:")
print("  1. Parallelized q-value optimization (7 independent tasks)")
print("  2. Vectorized distance calculations in jd() method")
print("\nUsage in your code:")
print("  from pyKriging.samplingplan_parallel import samplingplan_parallel")
print("  sp = samplingplan_parallel(k=2)")
print("  X = sp.optimallhc(100, n_jobs=-1)  # Use all CPU cores")
print("=" * 80)
