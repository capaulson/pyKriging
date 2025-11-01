#!/usr/bin/env python
"""
Quick test to verify the training fix works.
"""

import numpy as np
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan

print("Testing training fix...")

# Configure backend (will use CPU in this environment, Metal on your Mac)
pyKriging.configure_gpu(device='auto', verbose=False)

# Create small dataset
np.random.seed(42)
sp = samplingplan(2)
X = sp.optimallhc(20)
y = np.array([np.sin(x[0] * np.pi) + np.cos(x[1] * np.pi) for x in X])

print(f"Created dataset: {X.shape[0]} points in {X.shape[1]}D")

# Create and train model
print("Creating and training model...")
k = kriging(X, y, name='test_fix')
k.train(optimizer='ga')

print(f"✓ Training completed")
print(f"  theta: {k.theta}")
print(f"  p: {k.pl}")

# Check if mu and U were computed
print(f"\nChecking model state after training:")
print(f"  mu: {k.mu}")
print(f"  SigmaSqr: {k.SigmaSqr}")
print(f"  U is None: {k.U is None}")

if k.mu is None:
    print("\n❌ FAILED: mu is still None after training!")
    exit(1)

if k.U is None:
    print("\n❌ FAILED: U is still None after training!")
    exit(1)

# Test prediction
print(f"\nTesting prediction...")
test_point = [0.5, 0.5]
try:
    pred = k.predict(test_point)
    print(f"✓ Prediction at {test_point}: {pred}")
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    exit(1)

# Test multiple predictions
print(f"\nTesting multiple predictions...")
test_points = sp.rlh(10)
try:
    predictions = [k.predict(pt) for pt in test_points]
    print(f"✓ Made {len(predictions)} predictions successfully")
except Exception as e:
    print(f"❌ Multiple predictions failed: {e}")
    exit(1)

print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
print("The training fix is working correctly!")
