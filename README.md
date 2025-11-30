# pyKriging

A Kriging (Gaussian Process) metamodeling toolkit for Python with automatic GPU acceleration.

[![CI](https://github.com/capaulson/pyKriging/actions/workflows/ci.yml/badge.svg)](https://github.com/capaulson/pyKriging/actions/workflows/ci.yml)
[![Downloads](https://pepy.tech/badge/pykriging)](https://pepy.tech/project/pykriging)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.21389.svg)](http://dx.doi.org/10.5281/zenodo.21389)

## Features

- **Kriging metamodels** with automatic hyperparameter optimization
- **GPU acceleration** for 10-50x faster training on large datasets
  - NVIDIA GPUs via CUDA/CuPy
  - Apple Silicon via Metal/PyTorch MPS
  - Automatic fallback to CPU when GPU unavailable
- **Optimal Latin Hypercube** sampling with parallel optimization
- **Expected Improvement** for sequential experimental design
- **Uncertainty quantification** for predictions

## Installation

### Basic Installation (CPU only)
```bash
pip install pyKriging
```

### With GPU Support

**NVIDIA GPU (CUDA):**
```bash
pip install pyKriging[gpu-nvidia]
```

**Apple Silicon (Metal):**
```bash
pip install pyKriging[gpu-metal]
```

**From source:**
```bash
git clone https://github.com/capaulson/pyKriging.git
cd pyKriging
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan

# Define test function
def my_function(x):
    return np.sin(x[0] * np.pi) * np.cos(x[1] * np.pi)

# Generate optimal Latin hypercube sample
sp = samplingplan(k=2)
X = sp.optimallhc(20)  # 20 points in 2D

# Evaluate function at sample points
y = np.array([my_function(x) for x in X])

# Create and train kriging model
model = kriging(X, y, name='my_model')
model.train()

# Make predictions
prediction = model.predict([0.5, 0.5])
uncertainty = model.predicterr([0.5, 0.5])

print(f"Prediction: {prediction:.4f}")
print(f"Uncertainty: {uncertainty:.4f}")
```

## GPU Acceleration

pyKriging automatically detects and uses available GPU hardware:

```python
import pyKriging

# Check current backend
info = pyKriging.get_device_info()
print(f"Backend: {info['backend']}")
print(f"Device: {info['device_name']}")

# Explicitly configure backend
pyKriging.configure_gpu(device='metal')  # For Apple Silicon
pyKriging.configure_gpu(device='cuda')   # For NVIDIA
pyKriging.configure_gpu(device='cpu')    # Force CPU
pyKriging.configure_gpu(device='auto')   # Auto-detect (default)
```

### Performance

GPU acceleration provides significant speedups for training, especially with larger datasets:

| Training Points | CPU Time | GPU Time (Metal) | Speedup |
|----------------|----------|------------------|---------|
| 100            | 15s      | 8s               | 1.9x    |
| 250            | 45s      | 12s              | 3.8x    |
| 500            | 180s     | 25s              | 7.2x    |
| 1000           | 720s     | 60s              | 12x     |

*Times are approximate and vary by hardware.*

## Examples

See the `examples/` directory for more detailed examples:

- `2D_simple_train.py` - Basic 2D kriging model
- `2D_simple_train_expected_improvement.py` - Sequential sampling with EI
- `3d_Simple_Train.py` - 3D kriging model
- `2D_model_convergence.py` - Model convergence analysis

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests (faster)
pytest tests/ -v --ignore=tests/test_e2e.py

# Run with coverage
pytest tests/ -v --cov=pyKriging
```

## Running Benchmarks

```bash
# CPU vs GPU comparison
python benchmarks/benchmark_cpu_vs_gpu.py

# Sampling plan parallel optimization
python benchmarks/benchmark_lhc_parallel.py
```

## API Reference

### `kriging(X, y, name='', testfunction=None)`

Create a Kriging model.

- `X`: Training input points (n x k array)
- `y`: Training output values (n-length array)
- `name`: Optional model name
- `testfunction`: Optional ground truth function for validation

**Methods:**
- `train(optimizer='ga')`: Train the model
- `predict(x)`: Predict at point x
- `predicterr(x)`: Get prediction uncertainty at x
- `infill_ei(x)`: Expected improvement at x
- `addPoint(x, y)`: Add a new training point

### `samplingplan(k)`

Create sampling plans in k dimensions.

- `rlh(n)`: Random Latin hypercube with n points
- `optimallhc(n)`: Optimal Latin hypercube with n points
- `fullfactorial(ppd)`: Full factorial with ppd points per dimension

## Citation

If you use pyKriging in your research, please cite:

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

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
