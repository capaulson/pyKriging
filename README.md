# pyKriging

A Python Kriging toolkit with automatic GPU acceleration.

[![CI](https://github.com/capaulson/pyKriging/actions/workflows/ci.yml/badge.svg)](https://github.com/capaulson/pyKriging/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyKriging.svg)](https://pypi.org/project/pyKriging/)
[![Downloads](https://pepy.tech/badge/pykriging)](https://pepy.tech/project/pykriging)

## Installation

```bash
pip install pyKriging
```

**With GPU support:**
```bash
pip install pyKriging[gpu-nvidia]  # NVIDIA/CUDA
pip install pyKriging[gpu-metal]   # Apple Silicon
```

## Quick Start

```python
import numpy as np
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan

# Generate sample points
sp = samplingplan(k=2)
X = sp.optimallhc(20)

# Evaluate your function
y = np.array([my_function(x) for x in X])

# Train model
model = kriging(X, y)
model.train()

# Predict
prediction = model.predict([0.5, 0.5])
uncertainty = model.predicterr([0.5, 0.5])
```

## Core API

### Kriging Model

```python
model = kriging(X, y)
model.train()                    # Train the model
model.predict(x)                 # Predict at point x
model.predicterr(x)              # Prediction uncertainty
model.infill_ei(x)               # Expected improvement
model.addPoint(x, y)             # Add training point
```

### Sampling Plans

```python
sp = samplingplan(k=2)           # k = number of dimensions
X = sp.optimallhc(n)             # Optimal Latin hypercube (n points)
X = sp.rlh(n)                    # Random Latin hypercube
X = sp.fullfactorial(ppd)        # Full factorial (ppd points per dim)
```

### GPU Configuration

GPU acceleration is automatic when available. To check or configure:

```python
import pyKriging

pyKriging.get_device_info()              # Check current backend
pyKriging.configure_gpu(device='auto')   # Auto-detect (default)
pyKriging.configure_gpu(device='cuda')   # Force NVIDIA
pyKriging.configure_gpu(device='metal')  # Force Apple Silicon
pyKriging.configure_gpu(device='cpu')    # Force CPU
```

## Requirements

- Python 3.9+
- NumPy, SciPy, Matplotlib

## Citation

```bibtex
@software{paulson2015pykriging,
  author = {Paulson, Chris},
  title = {pyKriging: A Python Kriging Toolkit},
  year = {2015},
  doi = {10.5281/zenodo.21389},
  url = {https://github.com/capaulson/pyKriging}
}
```

## License

MIT
