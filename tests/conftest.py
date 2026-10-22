"""
Shared configuration of the pyKriging test suite, which is hermetic and deterministic:

* matplotlib draws with the Agg backend and every figure is closed after each test;
* every test starts on the CPU backend, and the module-wide backend is restored afterwards, whatever the test
  configured with configure_gpu();
* numpy's and Python's global random generators are seeded before each test (models are seeded with seed=);
* a test that writes into the pyKriging package directory fails.

Helpers shared by the test modules live in _reference.py, an independent float64 implementation of the
mathematics that does not import pyKriging; this directory is not a package, so the test modules import it by name.
"""
import os
import sys

os.environ['MPLBACKEND'] = 'Agg'  # before anything imports matplotlib
if 'matplotlib' in sys.modules:
    sys.modules['matplotlib'].use('Agg')

import random  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:  # pytest's default (prepend) import mode adds it; --import-mode=importlib does not
    sys.path.insert(0, HERE)
try:
    import pyKriging  # noqa: E402
except ImportError:  # a checkout that is not installed: test the package next to this directory
    sys.path.insert(0, str(Path(HERE).parent))
    import pyKriging  # noqa: E402
from pyKriging import gpu_backend  # noqa: E402

SEED = 1234
PACKAGE = Path(pyKriging.__file__).resolve().parent


def _package_files():
    """Size and modification time of every file in the package directory (bytecode caches excepted)."""
    files = {}
    for path in PACKAGE.rglob('*'):
        if '__pycache__' not in path.parts and path.is_file():
            stat = path.stat()
            files[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return files


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    """Each test starts on the CPU backend; monkeypatch puts the module-wide backend back afterwards, even when the
    test called configure_gpu()."""
    monkeypatch.setattr(gpu_backend, '_backend', gpu_backend.GPUBackend('cpu'))


@pytest.fixture(autouse=True)
def _seeded_global_rngs():
    np.random.seed(SEED)
    random.seed(SEED)


@pytest.fixture(autouse=True)
def _package_untouched():
    before = _package_files()
    yield
    after = _package_files()
    changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    assert not changed, f'the test wrote into the package directory: {changed}'


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    if 'matplotlib.pyplot' in sys.modules:
        sys.modules['matplotlib.pyplot'].close('all')
