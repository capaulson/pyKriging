"""
Test 2D Kriging Example (Simplified from examples/2D_simple_train.py)

This tests the GPU-accelerated Kriging on a 2D problem without plotting.
"""

from __future__ import print_function
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan
import numpy as np

print("="*80)
print("2D KRIGING EXAMPLE TEST")
print("="*80)

# Show GPU backend info
print("\nGPU Backend Information:")
info = pyKriging.get_device_info()
print(f"  Backend: {info['backend'].upper()}")
print(f"  Device: {info['device_name']}")

# Create sampling plan
print("\nStep 1: Creating sampling plan...")
sp = samplingplan(2)
X = sp.optimallhc(15)
print(f"  Created {X.shape[0]} points in {X.shape[1]}D")

# Define test function (Branin)
print("\nStep 2: Evaluating Branin test function...")
testfun = pyKriging.testfunctions().branin
y = testfun(X)
print(f"  Evaluated {len(y)} points")

# Create Kriging model
print("\nStep 3: Setting up Kriging model...")
k = kriging(X, y, testfunction=testfun, name='2d_test')
print("  Model initialized")

# Train model
print("\nStep 4: Training model (this may take a minute)...")
k.train(optimizer='ga')
print("  Training complete!")
print(f"  Optimized theta: {k.theta}")
print(f"  Optimized p: {k.pl}")
print(f"  Negative log-likelihood: {k.NegLnLike}")

# Add infill points
print("\nStep 5: Adding infill points...")
for i in range(3):
    newpoints = k.infill(2)
    print(f"  Iteration {i+1}: Adding {len(newpoints)} points")
    for point in newpoints:
        k.addPoint(point, testfun(point)[0])
    k.train()

print(f"  Final model has {k.n} points")

# Make predictions
print("\nStep 6: Testing predictions...")
test_points = np.array([
    [0.5, 0.5],
    [0.25, 0.75],
    [0.75, 0.25]
])

for point in test_points:
    pred = k.predict(point)
    actual = testfun(np.array([point]))[0]
    error = abs(pred - actual)
    print(f"  Point {point}: predicted={pred:.4f}, actual={actual:.4f}, error={error:.4f}")

print("\n" + "="*80)
print("2D KRIGING EXAMPLE COMPLETED SUCCESSFULLY!")
print("="*80)
