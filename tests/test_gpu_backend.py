"""
GPU backend: the compatibility shims and the float64 PyTorch likelihood kernel. The kernel runs on torch's CPU
device (the code is the same as on CUDA), and on CUDA too where available, and must reproduce matrixops' LAPACK
likelihood.
"""
import functools
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pyKriging
from pyKriging import gpu_backend as gb
from pyKriging import kriging
from pyKriging.matrixops import EPS, PENALTY


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    """Every test starts on the CPU backend and leaves the module-wide backend as it found it."""
    monkeypatch.setattr(gb, '_backend', gb.GPUBackend('cpu'))


@pytest.fixture(params=['cpu', 'cuda'])
def kernel(request):
    """The CUDA likelihood kernel, run on torch's CPU device and, where available, on CUDA."""
    torch = pytest.importorskip('torch')
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    return functools.partial(gb._torch_batch_nll, device=request.param)


def _model(n, k, seed=0):
    X = np.random.default_rng(seed).random((n, k))
    return kriging(X, np.sin(3 * X[:, 0]) + np.cos(2 * X[:, -1]) + np.sum(X, axis=1) ** 2 / 10)


def _candidates(B, k, seed=1):
    """Well-conditioned hyper-parameters: theta >= 10^0.5, about 30% of p exactly 2 (the bound optimisers clip to)."""
    rng = np.random.default_rng(seed)
    return 10 ** rng.uniform(0.5, 2.0, (B, k)), np.where(rng.random((B, k)) < 0.3, 2.0, rng.uniform(1.0, 2.0, (B, k)))


# ------------------------------------------------------------------ compatibility shims
def test_get_backend_is_a_module_singleton():
    first = gb.get_backend()
    assert gb.get_backend(verbose=False) is first and pyKriging.get_backend() is first
    assert gb.get_backend(device='metal') is first  # the device only applies when the backend is created
    assert gb.get_backend(device='cpu', force_reinit=True) is not first
    assert pyKriging.configure_gpu('cpu', verbose=False) is gb.get_backend()


def test_configure_cpu(capsys):
    backend = pyKriging.configure_gpu('cpu')
    assert (backend.backend_type, backend.device, backend.is_gpu()) == ('cpu', None, False)
    assert pyKriging.is_gpu_available() == ('cuda' in backend.available_backends)  # whether CUDA could be used
    assert backend.xp is np and backend.linalg is np.linalg and 'cpu' in backend.available_backends
    assert capsys.readouterr().out == ''  # verbose=True is accepted, but nothing is printed


@pytest.mark.parametrize('device', ['metal', 'mps'])
def test_metal_warns_and_falls_back_to_cpu(device):
    with pytest.warns(UserWarning, match='float64'):
        backend = pyKriging.configure_gpu(device, verbose=False)
    assert backend.backend_type == 'cpu' and not backend.is_gpu()


def test_unknown_device_raises_value_error():
    with pytest.raises(ValueError, match='tpu'):
        pyKriging.configure_gpu('tpu', verbose=False)


@pytest.mark.parametrize('torch_installed', [False, True])
def test_cuda_without_cuda_raises_runtime_error(monkeypatch, torch_installed):
    if not torch_installed:
        monkeypatch.setitem(sys.modules, 'torch', None)  # `import torch` now raises ImportError
    elif pytest.importorskip('torch').cuda.is_available():
        pytest.skip('CUDA is available')
    with pytest.raises(RuntimeError, match='cuda'):
        pyKriging.configure_gpu('cuda', verbose=False)
    assert gb.get_backend().backend_type == 'cpu'  # the backend in effect is unchanged


def test_auto_is_resolved_at_first_use(monkeypatch):
    probes = []
    monkeypatch.setattr(gb, '_cuda_available', lambda: probes.append(1) or False)
    backend = pyKriging.configure_gpu('auto', verbose=False)
    assert gb.batch_nll(_model(20, 2), np.ones((2, 2)), np.full((2, 2), 2.0), np.full(2, EPS)) is None
    assert probes == []  # neither creating the backend nor a small problem looks for CUDA (or imports torch)
    assert backend.backend_type == 'cpu' and backend.device is None
    assert probes == [1]  # resolved once, at first use


@pytest.mark.parametrize('platform, cuda', [('darwin', False), ('linux', True), ('win32', True)])
def test_cuda_probe_rules_out_macos_without_torch(monkeypatch, platform, cuda):
    """macOS has no CUDA (PyTorch >= 2.0 has no CUDA build for it), so 'auto' need not import torch there."""
    monkeypatch.setattr(sys, 'platform', platform)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    assert gb._cuda_available() is cuda


def test_a_failing_gpu_is_fatal_only_when_cuda_was_requested(monkeypatch):
    """'auto' promised acceleration only where it works: if the GPU fails (out of memory, a GPU or PyTorch release
    the kernel cannot use, ...) it warns and uses the CPU from then on; configure_gpu('cuda') raises the error."""
    def fail(*args):
        calls.append(args)
        raise RuntimeError('CUDA out of memory')

    calls, m = [], _model(gb.GPU_MIN_N, 2)
    theta, p, nugget = np.array([[20.0, 50.0], [30.0, 40.0]]), np.full((2, 2), 1.5), np.full(2, EPS)
    monkeypatch.setattr(gb, '_cuda_available', lambda: True)
    monkeypatch.setattr(gb, '_torch_batch_nll', fail)
    pyKriging.configure_gpu('cuda')
    with pytest.raises(RuntimeError, match='out of memory'):
        gb.batch_nll(m, theta, p, nugget)
    backend = pyKriging.configure_gpu('auto')
    with pytest.warns(RuntimeWarning, match='out of memory'):
        assert gb.batch_nll(m, theta, p, nugget) is None  # matrixops then computes the population with LAPACK
    assert (backend.backend_type, len(calls)) == ('cpu', 2) and pyKriging.is_gpu_available()  # device still there
    assert gb.batch_nll(m, theta, p, nugget) is None and len(calls) == 2  # the GPU is not tried again
    assert np.all(m._batch_nll(theta, p, nugget) < PENALTY)


def test_to_cpu_and_to_gpu_return_numpy():
    a = np.array([[1.0, 2.0], [3.0, 4.5]])
    backend = gb.get_backend()
    for move in (pyKriging.to_gpu, pyKriging.to_cpu, backend.to_gpu, backend.to_cpu):
        assert isinstance(move(a), np.ndarray) and move(None) is None
    np.testing.assert_array_equal(pyKriging.to_cpu(pyKriging.to_gpu(a)), a)

    class Points(list):  # has a .get() that is not CuPy's
        def get(self):
            raise AssertionError('to_cpu called .get() on an object that is not a CuPy array')

    for move in (gb.to_cpu, gb.to_gpu, backend.to_cpu):
        np.testing.assert_array_equal(move(Points([1.0, 2.0])), [1.0, 2.0])


def test_to_cpu_converts_torch_tensors():
    torch = pytest.importorskip('torch')
    t = torch.tensor([[1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    for move in (pyKriging.to_cpu, pyKriging.to_gpu, gb.to_cpu, gb.get_backend().to_cpu):
        assert isinstance(move(t), np.ndarray)
        np.testing.assert_array_equal(move(t * 2), [[2.0, 4.0]])


def test_device_info_and_is_gpu():
    info = pyKriging.get_device_info()
    assert {'backend', 'device_name', 'memory_total', 'memory_used'} <= set(info)
    assert info['backend'] == 'cpu' and info['device_name']
    assert isinstance(gb.get_backend().is_gpu(), bool) and isinstance(pyKriging.is_gpu_available(), bool)


def test_import_and_small_problems_do_not_import_torch():
    """torch costs ~0.5 s to import: only a problem large enough for the GPU may load it."""
    code = ('import sys, numpy as np, pyKriging; from pyKriging import gpu_backend as gb; pyKriging.get_backend(); '
            'm = pyKriging.kriging(np.random.rand(20, 2), np.random.rand(20)); '
            'assert gb.batch_nll(m, np.ones((2, 2)), np.full((2, 2), 2.0), np.full(2, 1e-10)) is None; '
            'print("torch" in sys.modules)')
    path = [str(Path(pyKriging.__file__).resolve().parents[1]), os.environ.get('PYTHONPATH', '')]
    run = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, path))))
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == 'False'


# ------------------------------------------------------------------ likelihood kernel (torch CPU, float64)
def test_kernel_matches_lapack_likelihood(kernel):
    m = _model(60, 3)
    theta, p = _candidates(40, 3)
    nugget = np.where(np.arange(40) % 2, EPS, np.logspace(-10, -2, 40))  # kriging's eps and regression Lambdas
    ref = m._batch_nll(theta, p, nugget)
    assert np.all(ref < PENALTY)
    np.testing.assert_allclose(kernel(m, theta, p, nugget), ref, rtol=1e-9)


def test_kernel_penalises_exactly_the_non_positive_definite_candidates(kernel):
    """theta ~ 1e-5 with p = 2 makes Psi numerically singular, so it is not PD under any rounding."""
    m = _model(60, 3)
    theta, p = _candidates(30, 3)
    theta[::3], p[::3] = 10 ** np.random.default_rng(2).uniform(-5, -4, (10, 3)), 2.0
    ref = m._batch_nll(theta, p, np.full(30, EPS))
    got = kernel(m, theta, p, np.full(30, EPS))
    assert np.sum(ref == PENALTY) == 10
    np.testing.assert_array_equal(got == PENALTY, ref == PENALTY)
    np.testing.assert_allclose(got[ref < PENALTY], ref[ref < PENALTY], rtol=1e-9)


def test_kernel_takes_the_verdict_from_info(kernel, monkeypatch):
    """A failed CUDA factorisation leaves unspecified, possibly finite, values: only info may decide."""
    torch = pytest.importorskip('torch')
    m = _model(30, 2)
    theta, p = _candidates(6, 2)
    factorise, failed = torch.linalg.cholesky_ex, torch.tensor([0, 3, 0, 0, 7, 0], dtype=torch.int32)
    monkeypatch.setattr(torch.linalg, 'cholesky_ex', lambda A, **kw: (factorise(A, **kw)[0], failed.to(A.device)))
    assert (kernel(m, theta, p, np.full(6, EPS)) == PENALTY).tolist() == [False, True, False, False, True, False]


def test_kernel_with_responses_per_candidate(kernel):
    """y (B, n) replaces the model's y per candidate (co-kriging's differences y_e - rho y_c(X_e)); a constant y
    gives sigma^2 = 0, a non-finite likelihood and so PENALTY."""
    m = _model(60, 3)
    theta, p = _candidates(12, 3)
    nugget = np.full(12, EPS)
    Y = m.y + np.random.default_rng(3).normal(0.0, 0.2, (12, m.n))
    Y[5] = 0.0
    with np.errstate(divide='ignore'):
        ref = m._batch_nll(theta, p, nugget, Y)
    got = kernel(m, theta, p, nugget, Y)
    assert ref[5] == got[5] == PENALTY
    np.testing.assert_allclose(got, ref, rtol=1e-9)
    assert not np.allclose(got, kernel(m, theta, p, nugget))


def test_kernel_chunks_the_population(kernel, monkeypatch):
    torch = pytest.importorskip('torch')
    m = _model(30, 2)
    theta, p = _candidates(25, 2)
    nugget = np.full(25, EPS)
    whole = kernel(m, theta, p, nugget)
    sizes, factorise = [], torch.linalg.cholesky_ex
    monkeypatch.setattr(torch.linalg, 'cholesky_ex', lambda A, **kw: sizes.append(len(A)) or factorise(A, **kw))
    chunked = kernel(m, theta, p, nugget, chunk_bytes=4 * 4 * 8 * 30 ** 2)  # four candidates: ~4 (30, 30) tensors each
    assert sizes == [4] * 6 + [1]
    np.testing.assert_allclose(chunked, whole, rtol=1e-12)


def test_batch_nll_runs_only_populations_of_large_problems_on_cuda(monkeypatch):
    """batch_nll returns None (use LAPACK) unless the backend is CUDA, n >= GPU_MIN_N and there are several
    candidates; then it gives the CPU's likelihoods. 'CUDA' is a backend whose torch device is the CPU."""
    pytest.importorskip('torch')
    big, small = _model(gb.GPU_MIN_N, 2), _model(gb.GPU_MIN_N - 1, 2)
    theta, p, nugget = np.array([[20.0, 50.0], [30.0, 40.0]]), np.full((2, 2), 1.5), np.full(2, EPS)
    Y = big.y + np.random.default_rng(4).normal(0.0, 0.1, (2, big.n))
    ref, ref_y = big._batch_nll(theta, p, nugget), big._batch_nll(theta, p, nugget, Y)
    assert gb.batch_nll(big, theta, p, nugget) is None  # CPU backend
    monkeypatch.setattr(gb, '_backend', SimpleNamespace(backend_type='cuda', device='cpu'))
    assert gb.batch_nll(small, theta, p, nugget) is None
    assert gb.batch_nll(big, theta[:1], p[:1], nugget[:1]) is None  # the SLSQP polish stays on LAPACK
    np.testing.assert_allclose(gb.batch_nll(big, theta, p, nugget), ref, rtol=1e-9)
    np.testing.assert_allclose(gb.batch_nll(big, theta, p, nugget, Y), ref_y, rtol=1e-9)


def test_batch_nll_confirms_the_best_candidate_with_lapack(monkeypatch):
    """The model refits the best candidate with LAPACK, so a best value that LAPACK cannot reproduce (Psi PD on the
    GPU only, which rounding allows near singularity) must become PENALTY, until LAPACK confirms one."""
    m = _model(gb.GPU_MIN_N, 2)
    theta = np.array([[1e-5, 1e-5], [20.0, 50.0], [30.0, 40.0], [2e-5, 1e-5]])  # rows 0 and 3: Psi singular
    p, nugget = np.array([[2.0, 2.0], [1.5, 1.5], [1.5, 1.5], [2.0, 2.0]]), np.full(4, EPS)
    assert (m._batch_nll(theta, p, nugget) == PENALTY).tolist() == [True, False, False, True]
    monkeypatch.setattr(gb, '_backend', SimpleNamespace(backend_type='cuda', device='cpu'))
    monkeypatch.setattr(gb, '_torch_batch_nll', lambda *args: np.array([-2e3, -900.0, -890.0, -3e3]))
    np.testing.assert_array_equal(gb.batch_nll(m, theta, p, nugget), [PENALTY, -900.0, -890.0, PENALTY])
    Y = np.vstack([m.y] * 4)
    Y[1] = 0.0  # confirmed with its own responses, candidate 1 has no finite likelihood (sigma^2 = 0)
    with np.errstate(divide='ignore'):
        np.testing.assert_array_equal(gb.batch_nll(m, theta, p, nugget, Y), [PENALTY, PENALTY, -890.0, PENALTY])
