"""
Optional CUDA acceleration of the hyper-parameter search (float64 PyTorch); model state is always float64 NumPy.

The one accelerated computation is the concentrated likelihood of a whole optimiser population (Forrester et al.
2007, eqs 2.3 and 3.4-3.6): for each candidate, with Psi = L L^T, a = L^-1 1 and b = L^-1 y,

    mu = a.b / a.a,    sigma^2 = |b - mu a|^2 / n,    NegLnLike = n/2 ln(sigma^2) + sum(ln diag L),

by batched Cholesky factorisations and triangular solves. It is opt-in: the default backend is the CPU, and the
kernel runs after configure_gpu('cuda') (or configure_gpu('auto') on a machine with CUDA, which moves to the CPU with
a warning if the GPU fails) for problems with n >= GPU_MIN_N. Apple Metal is not supported: it has no float64, and the nuggets float32 needs raise a model's test
error about 100x.
"""
import sys
import warnings

import numpy as np

GPU_MIN_N = 300  # smaller problems are as fast in LAPACK (an estimate, still to be measured on CUDA hardware)
_backend = None


def _cuda_available():
    """Whether PyTorch is installed with a usable CUDA device (the first call imports torch, ~0.5 s and ~150 MB;
    macOS, which has no CUDA, is ruled out without it)."""
    if sys.platform == 'darwin':
        return False
    try:
        import torch
    except (ImportError, OSError):
        return False
    return torch.cuda.is_available()


def to_cpu(arr):
    """NumPy array of arr (torch tensors and CuPy arrays are copied to the host); None passes through."""
    torch, cupy = sys.modules.get('torch'), sys.modules.get('cupy')
    if torch is not None and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    if cupy is not None and isinstance(arr, cupy.ndarray):
        return arr.get()
    return None if arr is None else np.asarray(arr)


to_gpu = to_cpu  # model state is NumPy on every backend (kept for compatibility)


class GPUBackend:
    """Where train() evaluates its population likelihood: backend_type 'cuda' or 'cpu'. 'auto' is resolved at first
    use, so creating a backend never imports torch, and becomes 'cpu' if the GPU fails. verbose is kept for
    compatibility (nothing is printed)."""
    xp, linalg = np, np.linalg
    to_cpu, to_gpu = staticmethod(to_cpu), staticmethod(to_gpu)

    def __init__(self, device='auto', verbose=True):
        if device in ('metal', 'mps'):
            warnings.warn('Apple Metal has no float64, which Kriging needs: float32 cannot factorise the near-singular '
                          'correlation matrices of good fits without nuggets that spoil them. Using the CPU.',
                          UserWarning, stacklevel=3)
            device = 'cpu'
        if device not in ('auto', 'cpu', 'cuda'):
            raise ValueError(f"unknown device {device!r}: use 'auto', 'cpu' or 'cuda'")
        if device == 'cuda' and not _cuda_available():
            raise RuntimeError("Backend 'cuda' is not available on this system: it needs PyTorch with CUDA support.")
        self._device, self._auto = device, device == 'auto'

    @property
    def backend_type(self):
        if self._device == 'auto':
            self._device = 'cuda' if _cuda_available() else 'cpu'
        return self._device

    @property
    def device(self):
        """torch device of the likelihood kernel (None on the CPU backend)."""
        return 'cuda' if self.backend_type == 'cuda' else None

    @property
    def available_backends(self):
        return ['cpu', 'cuda'] if _cuda_available() else ['cpu']

    def is_gpu(self):
        return self.backend_type == 'cuda'

    def get_device_info(self):
        """Backend, device name and, on CUDA, the device's total and used memory in GiB."""
        info = {'backend': self.backend_type, 'device_name': 'CPU', 'memory_total': None, 'memory_used': None}
        if self.is_gpu():
            import torch
            free, total = torch.cuda.mem_get_info()
            info.update(device_name=torch.cuda.get_device_name(), memory_total=total / 2 ** 30,
                        memory_used=(total - free) / 2 ** 30)
        return info


def get_backend(device='cpu', verbose=None, force_reinit=False):
    """The module-wide backend, created as GPUBackend(device) on first use (or again when force_reinit); the GPU is
    opt-in through configure_gpu()."""
    global _backend
    if _backend is None or force_reinit:
        _backend = GPUBackend(device)
    return _backend


def configure_gpu(device='auto', verbose=True):
    """Make GPUBackend(device) the module-wide backend and return it: 'cuda' (RuntimeError without PyTorch CUDA
    support), 'cpu', 'auto' (CUDA if available, decided by the first problem large enough to use it; the CPU from
    the first GPU failure on) or 'metal'/'mps' (a warning, then the CPU); ValueError for other names."""
    global _backend
    _backend = GPUBackend(device)
    return _backend


def get_device_info():
    """{'backend', 'device_name', 'memory_total', 'memory_used'} of the module-wide backend."""
    return get_backend().get_device_info()


def is_gpu_available():
    """Whether a CUDA device can be used (configure_gpu('cuda') opts in; the default backend is the CPU)."""
    return 'cuda' in get_backend().available_backends


def batch_nll(model, theta, p, nugget, y=None):
    """model._batch_nll(theta, p, nugget, y) on the GPU, or None where that does not apply: backend not CUDA, model.n <
    GPU_MIN_N (checked first, so small problems never import torch) or one candidate (the SLSQP polish). If the GPU
    fails (out of memory, a GPU or PyTorch release the kernel cannot use), a backend chosen by 'auto' warns and moves
    to the CPU for good, while configure_gpu('cuda') raises the error. The model refits the best candidate with
    LAPACK, where rounding can find a near-singular Psi not PD although the GPU factorised it, so candidates are
    confirmed alone by model._batch_nll (LAPACK), best first, and become PENALTY until one passes."""
    backend = get_backend()
    if model.n < GPU_MIN_N or len(theta) < 2 or backend.backend_type != 'cuda':
        return None
    from .matrixops import PENALTY  # imported here because matrixops imports this module
    try:
        out = _torch_batch_nll(model, theta, p, nugget, y, backend.device)
    except Exception as err:  # 'auto' promised acceleration only where it works: the CPU computes the same values
        if not backend._auto:
            raise
        backend._device = 'cpu'
        warnings.warn(f'the CUDA likelihood failed, so the CPU is used from now on ({type(err).__name__}: {err})',
                      RuntimeWarning)
        return None
    for i in np.argsort(out):
        if out[i] >= PENALTY:
            break
        one = slice(i, i + 1)
        if model._batch_nll(theta[one], p[one], nugget[one], None if y is None else y[one])[0] < PENALTY:
            break
        out[i] = PENALTY
    return out


def _torch_batch_nll(model, theta, p, nugget, y=None, device='cuda', chunk_bytes=2 ** 30):
    """NegLnLike of candidates theta, p (B, k) with diagonal nuggets (B,) and optional responses y (B, n), in float64
    torch on `device`; PENALTY where Psi is not positive definite (info != 0) or the value is not finite. Psi is built
    per dimension as in matrixops._correlations, in chunks of b candidates whose (b, n, n) tensors fit in chunk_bytes;
    one host synchronisation per chunk."""
    import torch
    from .matrixops import PENALTY
    f64, n = dict(dtype=torch.float64, device=device), model.n
    d, upper = torch.as_tensor(model._d, **f64), torch.as_tensor(model._upper, device=device)
    theta, p, nugget = (torch.as_tensor(np.ascontiguousarray(a, dtype=float), **f64) for a in (theta, p, nugget))
    ys = torch.as_tensor(np.ascontiguousarray(model.y if y is None else y, dtype=float), **f64).expand(len(theta), n)
    step, out = max(1, chunk_bytes // (4 * 8 * n * n)), np.empty(len(theta))  # ~4 (b, n, n) tensors alive at once
    for s in range(0, len(theta), step):
        th, pp, yc = theta[s:s + step], p[s:s + step], ys[s:s + step]
        acc = torch.zeros(len(th), d.shape[1], **f64)
        for j, dj in enumerate(d):  # |d|^p, exactly d*d (model._d2) where p == 2
            acc += torch.where(pp[:, j, None] == 2.0, dj * dj, torch.pow(dj, pp[:, j, None])).mul_(th[:, j, None])
        Psi = torch.zeros(len(th), n * n, **f64).index_copy_(1, upper, acc.neg_().exp_()).view(-1, n, n)
        Psi.diagonal(dim1=1, dim2=2).copy_(1.0 + nugget[s:s + step, None])
        U, info = torch.linalg.cholesky_ex(Psi, upper=True)  # Psi = U^T U from its upper triangle; never raises
        ab = torch.linalg.solve_triangular(U.mT, torch.stack([torch.ones_like(yc), yc], dim=2), upper=False)
        a, b = ab[..., 0], ab[..., 1]
        res = b - ((a * b).sum(1) / (a * a).sum(1))[:, None] * a
        nll = 0.5 * n * torch.log((res * res).sum(1) / n) + torch.log(U.diagonal(dim1=1, dim2=2)).sum(1)
        out[s:s + step] = torch.where((info == 0) & torch.isfinite(nll), nll, PENALTY).cpu().numpy()
    return out
