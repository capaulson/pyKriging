# pyKriging

A Python Kriging (Gaussian-process) toolkit for surrogate modelling: fit a model to a few expensive evaluations,
predict with error estimates, and choose where to sample next.

[![CI](https://github.com/capaulson/pyKriging/actions/workflows/ci.yml/badge.svg)](https://github.com/capaulson/pyKriging/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyKriging.svg)](https://pypi.org/project/pyKriging/)
[![Downloads](https://pepy.tech/badge/pykriging)](https://pepy.tech/project/pykriging)

## Installation

```bash
pip install pyKriging
```

With NumPy/SciPy built on OpenBLAS (Linux, Windows, macOS before 14), also install `threadpoolctl`
(`pip install "pyKriging[threads]"`): pyKriging then keeps OpenBLAS's threads from competing with its own during
prediction, which makes predictions 2-6x faster. NumPy and SciPy on macOS 14 and later use Accelerate and gain
nothing from it.

## Quick Start

```python
import numpy as np
from pyKriging import kriging, samplingplan, testfunctions

X = samplingplan(k=2).optimallhc(20)    # 20-point optimal Latin hypercube in [0, 1]^2
y = testfunctions().branin(X)           # your expensive function: one value per row of X

model = kriging(X, y, seed=0)           # seed makes training and infill reproducible
model.train()

model.predict([0.5, 0.5])               # one point -> float
model.predict(np.random.rand(1000, 2))  # many points -> array
model.predicterr([0.5, 0.5])            # standard error of the prediction
```

## Core API

### Kriging Model

```python
model = kriging(X, y, seed=None)        # X: (n, k) samples, y: (n,) responses, in your own units
model.train(optimizer='pso')            # maximum-likelihood theta and p ('pso' or 'ga')
model.predict(x)                        # x: one point (k,) or points (m, k)
model.predicterr(x)                     # standard error, in the units of y
model.infill_ei(x)                      # expected improvement
model.infill(2, method='ei')            # 2 new points to sample: method='error' or 'ei'
model.addPoint(x_new, y_new)            # add an observation; call train() again to re-tune
```

### Regression Kriging

For noisy data: the same API, with a regression constant `model.Lambda` tuned along with theta and p.

```python
from pyKriging import regression_kriging
model = regression_kriging(X, y, seed=0)
model.train()
```

### Co-Kriging

Combines many cheap, approximate samples with a few expensive ones (Forrester, Sóbester & Keane, *Multi-fidelity
optimization via surrogate modelling*, Proc. R. Soc. A 463:3251, 2007,
[doi:10.1098/rspa.2007.1900](https://doi.org/10.1098/rspa.2007.1900)).
`examples/coKriging.py` reproduces the paper's one-variable example.

```python
from pyKriging.coKriging import coKriging
ck = coKriging(Xc, yc, Xe, ye, seed=0)  # cheap (Xc, yc) and expensive (Xe, ye) data
ck.train()
ck.predict(x)                           # prediction of the expensive function
ck.predict_var(x)                       # its standard error, with y scaled to [0, 1]
ck.rho                                  # estimated scaling between the cheap and expensive data
```

### Sampling Plans

```python
sp = samplingplan(k=2)                  # k = number of dimensions; random_state=... for repeatable plans
X = sp.optimallhc(n)                    # Morris-Mitchell optimal Latin hypercube (n points)
X = sp.rlh(n)                           # random Latin hypercube
X = sp.fullfactorial(ppd)               # full factorial (ppd points per dim)
Xe, Xrest = sp.subset(X, ne)            # space-filling subset, e.g. the expensive points of a co-kriging plan
```

### Cross-Validation, Sensitivity and Saving

```python
from pyKriging import loadModel, saveModel
from pyKriging.CrossValidation import Cross_Validation
from pyKriging.GlobalSensitivity import SobolSensitivity

Cross_Validation(model).calculate_SCVR()    # leave-one-out predictions and standardised residuals
sobol = SobolSensitivity(X, y)              # fits its own kriging surrogate (or pass model=...)
sobol.sensitivity_Sobol()                   # first-order Sobol' indices; total-order indices in sobol.ST
saveModel(model, 'model.pkl')               # save with dill; loadModel('model.pkl') restores it
```

### GPU Configuration

pyKriging computes on the CPU by default: all model state is float64 NumPy, and LAPACK is fastest at the sizes
kriging is usually used for. GPU use is opt-in: with PyTorch and an NVIDIA GPU (`pip install "pyKriging[gpu]"`),
`configure_gpu('cuda')` runs the likelihood search of models trained on 300 or more samples on CUDA, in float64.

```python
import pyKriging

pyKriging.configure_gpu('cuda')         # use CUDA (RuntimeError without PyTorch CUDA support)
pyKriging.configure_gpu('auto')         # or: CUDA if PyTorch finds a device, else the CPU
pyKriging.configure_gpu('cpu')          # back to the default
pyKriging.get_device_info()             # current backend and device
```

Apple Metal is not supported: it has no float64, and the float32 path was both slower than the CPU code and less
accurate. `configure_gpu('metal')` warns and uses the CPU. CuPy is no longer used.

## What's New in 0.2

- A rewritten numerical core: training, prediction and optimal Latin hypercubes are much faster, and predictions
  accept arrays of points (`benchmarks/bench.py` measures it on your machine).
- New: `seed` for reproducible models, `predicterr` and `infill_ei` in the units of y, a complete co-kriging
  model, `samplingplan.subset`, closed-form leave-one-out cross-validation and true Sobol' indices.
- inspyred is no longer needed; `optimallhc` no longer caches plans on disk or starts worker processes.
- GPU use is opt-in and CUDA-only (`configure_gpu('cuda')`): the `gpu-metal` and `gpu-all` extras are gone, and
  `gpu-nvidia` remains as an alias of `gpu`.

## Requirements

- Python 3.9+
- NumPy, SciPy, Matplotlib, dill
- Optional: PyTorch with CUDA, for the GPU likelihood

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
