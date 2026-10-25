"""
Linear-algebra core shared by the Kriging models (float64 NumPy / LAPACK).

With the Cholesky factorisation Psi = U^T U of the correlation matrix and a = U^-T 1, b = U^-T y,
the maximum-likelihood estimates of Forrester et al. (2007, eqs 2.3 and 3.4-3.6) are

    mu = a.b / a.a,    n sigma^2 = |b - mu a|^2,    ln|Psi| = 2 sum(ln diag U),
    NegLnLike = n/2 ln(sigma^2) + 1/2 ln|Psi|,

and the predictor weights w = Psi^-1 (y - 1 mu) = U^-1 (b - mu a) are cached once per fit, so a
prediction costs O(nk) and an error estimate a single triangular solve.
"""
import contextlib
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.linalg import LinAlgError, lapack
from scipy.special import erfcx, ndtr

from . import gpu_backend

EPS = np.finfo(float).eps
PENALTY = 10000.0  # objective value for hyper-parameters whose correlation matrix is not positive definite
_THREADS = min(8, os.cpu_count() or 1)
_pool = _pool_pid = None
_blas = None  # threadpoolctl controller, False if threadpoolctl is not installed


def _one_blas_thread(size):
    """Context limiting BLAS to one thread for work of `size` elements large enough for the threaded kernels, if the
    optional threadpoolctl is installed. Batch prediction interleaves small BLAS calls with the threaded correlation
    kernel, and OpenBLAS's own spinning threads then compete with it (2-6x slower); Accelerate is unaffected.
    Small batches skip the limit, which costs a few microseconds, and training keeps BLAS threads: they help large
    Cholesky factorisations."""
    global _blas
    if size <= 2 ** 17 or _THREADS < 2:
        return contextlib.nullcontext()
    if _blas is None:
        try:
            from threadpoolctl import ThreadpoolController  # deferred: scans the loaded libraries
            _blas = ThreadpoolController()
        except ImportError:
            _blas = False
    return _blas.limit(limits=1, user_api='blas') if _blas else contextlib.nullcontext()


def _executor():
    """Thread pool for the correlation kernels; recreated in a forked child, which inherits no worker threads."""
    global _pool, _pool_pid
    if _pool_pid != os.getpid():
        _pool, _pool_pid = ThreadPoolExecutor(_THREADS), os.getpid()
    return _pool


def _correlations(theta, p, d, d2):
    """exp(-sum_j theta_j |d_j|^p_j) for B hyper-parameter rows (B, k) and pair distances d (k, m) -> (B, m).
    d2 = d**2 serves the common p = 2 exactly (pow(d, 2) == d*d bitwise)."""
    acc = np.zeros((len(theta), d.shape[1]))
    for j in range(len(d)):
        at2 = p[:, j] == 2.0
        term = np.empty_like(acc)
        term[at2] = d2[j]
        term[~at2] = np.power(d[j], p[~at2, j, None])
        acc += theta[:, j, None] * term
    return np.exp(-acc, out=acc)


def correlation(A, B, theta, p):
    """Correlation matrix exp(-sum_j theta_j |a_j - b_j|^p_j) between point sets A (m, k) and B (n, k) (eq. 2.2),
    from contiguous (k, rows, n) blocks of about 1 MB, which large problems spread over threads. p = 2 is
    computed as d*d: exact, and several times cheaper than pow."""
    m, (n, k) = len(A), B.shape
    At, Bt = np.ascontiguousarray(A.T, dtype=float), np.ascontiguousarray(B.T, dtype=float)
    theta, p = np.asarray(theta, dtype=float)[:, None, None], np.asarray(p, dtype=float)[:, None, None]
    square = bool(np.all(p == 2.0))
    out, step = np.empty((m, n)), max(1, 2 ** 17 // (n * k))

    def rows(a):
        D = np.subtract(At[:, a:a + step, None], Bt[:, None, :])
        np.abs(D, out=D)
        if square:
            np.multiply(D, D, out=D)
        else:
            np.power(D, p, out=D)
        block = np.sum(np.multiply(D, theta, out=D), axis=0, out=out[a:a + step])
        np.exp(np.negative(block, out=block), out=block)

    list((_executor().map if m > step and _THREADS > 1 else map)(rows, range(0, m, step)))
    return out


def log_expected_improvement(f, s, y_min):
    """ln EI without underflow or cancellation (Forrester et al. 2007, sec. 5 and appendix B, whose eq. (B2) lacks
    a leading minus sign). EI = s h(u), u = (y_min - f)/s, h(u) = u Phi(u) + phi(u); for u < -1,
    h(u) = phi(u) (1 - |u| sqrt(pi/2) erfcx(|u|/sqrt(2))), which tends to phi(u)/u^2. -inf where s = 0."""
    f, s = np.broadcast_arrays(np.asarray(f, dtype=float), np.asarray(s, dtype=float))
    out = np.full(f.shape, -np.inf)
    ok = s > 0
    u = (y_min - f[ok]) / s[ok]
    log_h = np.empty_like(u)
    mid, far = (u <= -1) & (u > -1e4), u <= -1e4  # far: asymptotic phi(u)/u^2, relative error < 3/u^2
    near = ~(mid | far)
    log_h[near] = np.log(u[near] * ndtr(u[near]) + np.exp(-0.5 * u[near] ** 2) / np.sqrt(2 * np.pi))
    t = np.log(-u[mid] * erfcx(-u[mid] / np.sqrt(2))) + 0.5 * np.log(np.pi / 2)  # ln(1 - h/phi) < 0
    log1mexp = np.where(t > -np.log(2), np.log(-np.expm1(t)), np.log1p(-np.exp(t)))
    log_h[mid] = -0.5 * u[mid] ** 2 - 0.5 * np.log(2 * np.pi) + log1mexp
    log_h[far] = -0.5 * u[far] ** 2 - 0.5 * np.log(2 * np.pi) - 2 * np.log(-u[far])
    out[ok] = np.log(s[ok]) + log_h
    return out


def expected_improvement(f, s, y_min):
    """Expected improvement (y_min - f) Phi(u) + s phi(u), u = (y_min - f)/s; 0 where s = 0 (section 5)."""
    return np.exp(log_expected_improvement(f, s, y_min))


class matrixops:
    _jitter = 0.0  # extra diagonal used only while imputing points during multi-point infill

    def __init__(self):
        self.LnDetPsi = self.mu = self.U = self.SigmaSqr = self.NegLnLike = None
        self.updateData()

    def _nugget(self):
        """Diagonal added to Psi for numerical stability (regression kriging: the regression constant Lambda)."""
        return EPS

    def _err_offset(self):
        """Constant added inside the error estimate s^2 = sigma^2 (offset + 1 - psi^T Psi^-1 psi)."""
        return 0.0

    def updateData(self):
        """Pairwise |x_i - x_j| (i < j) of the normalised training data, per dimension (k, n(n-1)/2)."""
        self.n, self.k = self.X.shape
        iu = np.triu_indices(self.n, 1)
        self._upper = iu[0] * self.n + iu[1]  # flat C-order indices of the strict upper triangle
        self._lower = iu[1] * self.n + iu[0]  # and of its mirror image, the strict lower triangle
        self._d = np.abs(self.X[iu[0]] - self.X[iu[1]]).T.copy()
        self._d2 = self._d * self._d

    def _pair_correlations(self, theta, p):
        """Correlations of all training pairs for B hyper-parameter rows; large problems split the pairs over threads
        (the arithmetic is element-wise, so the values do not depend on the split)."""
        theta, p = np.atleast_2d(theta), np.atleast_2d(p)
        m = self._d.shape[1]
        if theta.size * m < 200_000 or _THREADS < 2:
            return _correlations(theta, p, self._d, self._d2)
        out, cuts = np.empty((len(theta), m)), np.linspace(0, m, _THREADS + 1).astype(int)

        def block(a, b):
            out[:, a:b] = _correlations(theta, p, self._d[:, a:b], self._d2[:, a:b])

        list(_executor().map(block, cuts[:-1], cuts[1:]))
        return out

    def _batch_nll(self, theta, p, nugget, y=None):
        """NegLnLike for B hyper-parameter sets (theta, p: (B, k); nugget: (B,)), PENALTY where Psi is not PD.
        y: optional per-candidate responses (B, n) to use instead of self.y."""
        accelerated = gpu_backend.batch_nll(self, theta, p, nugget, y)  # None unless CUDA is configured and n is large
        if accelerated is not None:
            return accelerated
        n, out = self.n, np.full(len(theta), PENALTY)
        rhs = np.asfortranarray(np.column_stack([np.ones(n), self.y]))
        work = np.empty((n, n))
        step = max(1, 4_000_000 // max(1, self._d.size))  # bound the (B, m) correlation block to ~32 MB
        for s in range(0, len(theta), step):
            for i, r in enumerate(self._pair_correlations(theta[s:s + step], p[s:s + step]), s):
                U = self._cholesky(r, nugget[i], work)
                if U is None:
                    continue
                if y is not None:
                    rhs[:, 1] = y[i]
                ab = lapack.dtrtrs(U, rhs, lower=0, trans=1)[0]
                a, b = ab[:, 0], ab[:, 1]
                res = b - (a @ b) / (a @ a) * a
                nll = 0.5 * n * np.log(res @ res / n) + np.sum(np.log(U.diagonal()))
                if np.isfinite(nll):
                    out[i] = nll
        return out

    def _cholesky(self, r, nugget, work):
        """Upper Cholesky factor U (Psi = U^T U) of Psi (pair correlations r, 1 + nugget on the diagonal) computed
        in `work`, or None if Psi is not numerically positive definite. Psi is written to the lower triangle of the
        C-ordered work array, i.e. the upper triangle of its Fortran-ordered transpose, so LAPACK factorises in
        place. (The upper form also avoids a crash of dpotrf(lower=1) in the OpenBLAS of scipy 1.7 on arm64.)
        Every fit uses this routine: near-singular matrices then pass or fail identically in every code path (though
        not across LAPACK libraries)."""
        flat = work.reshape(-1)
        flat[self._lower] = r
        flat[::self.n + 1] = 1.0 + nugget
        U, info = lapack.dpotrf(work.T, lower=0, clean=0, overwrite_a=1)
        return None if info else U

    def updatePsi(self):
        """Build the correlation matrix Psi and its upper Cholesky factor U (raises LinAlgError if not PD)."""
        r = self._pair_correlations(self.theta, self.pl)[0]
        nugget = self._nugget() + self._jitter
        U = self._cholesky(r, nugget, np.empty((self.n, self.n)))
        if U is None:
            raise LinAlgError('correlation matrix is not positive definite')
        U[np.tril_indices(self.n, -1)] = 0.0  # clean in place: U stays Fortran-ordered for LAPACK
        self.U = U
        self.Psi = np.eye(self.n) * (1.0 + nugget)
        self.Psi.reshape(-1)[self._upper] = r
        self.Psi = np.triu(self.Psi) + np.triu(self.Psi, 1).T

    def neglikelihood(self):
        """MLEs mu and sigma^2, ln|Psi| and the concentrated NegLnLike; caches the predictor weights."""
        ab = lapack.dtrtrs(self.U, np.column_stack([np.ones(self.n), self.y]), lower=0, trans=1)[0]
        a, b = ab[:, 0], ab[:, 1]
        self.mu = (a @ b) / (a @ a)
        res = b - self.mu * a
        self.SigmaSqr = (res @ res) / self.n
        self.LnDetPsi = 2.0 * np.sum(np.log(np.abs(np.diag(self.U))))
        with np.errstate(divide='ignore'):  # constant data: sigma^2 = 0
            self.NegLnLike = 0.5 * self.n * np.log(self.SigmaSqr) + 0.5 * self.LnDetPsi
        self._weights = lapack.dtrtrs(self.U, res, lower=0)[0]

    @staticmethod
    def _as_points(x, k):
        """(m, k) view of x and whether x was a single point (scalar results are returned for single points).
        A point has shape (k,); m points have shape (m, k) (or (m,) when k = 1)."""
        x = np.asarray(x, dtype=float)
        if x.ndim > 2 or (x.ndim == 2 and x.shape[1] != k) or (x.ndim < 2 and k > 1 and x.size != k):
            raise ValueError(f'expected a point of length {k} or an (m, {k}) array of points, got shape {x.shape}')
        return x.reshape(-1, k), x.ndim <= 1 and x.size == k

    def _psi(self, x):
        """Correlations between normalised points x (m, k) and the training points -> (m, n)."""
        return correlation(x, self.X, self.theta, self.pl)

    def _variance(self, psi):
        """sigma^2 (offset + 1 - psi^T Psi^-1 psi) for correlation rows psi (m, n)."""
        v = lapack.dtrtrs(self.U, psi.T, lower=0, trans=1)[0]
        return self.SigmaSqr * (1.0 + self._err_offset() - np.einsum('ij,ij->j', v, v))

    def _improvement_variance(self, psi):
        """Variance used by expected improvement and error-based infill (regression kriging: re-interpolation)."""
        return self._variance(psi)

    def _predict(self, x, mean=True, error=True, improvement=False):
        """Predictor mean and/or standard error (normalised units; the infill error if improvement) at points x
        (m, k), from one correlation block per chunk of points (chunked to bound memory)."""
        if self.mu is None:
            raise RuntimeError('the model is not fitted yet: call train() or update(values) first')
        step = max(1, 2_000_000 // (self.n * self.k))
        f, s = np.empty(len(x)), np.empty(len(x))
        with _one_blas_thread(len(x) * self.n * self.k):
            for i in range(0, len(x), step):
                psi = self._psi(x[i:i + step])
                if mean:
                    f[i:i + step] = self.mu + psi @ self._weights
                if error:
                    s[i:i + step] = np.sqrt(np.abs(self._improvement_variance(psi) if improvement else
                                                   self._variance(psi)))
        return f, s

    def predict_normalized(self, x):
        """Kriging prediction mu + psi^T Psi^-1 (y - 1 mu) at normalised point(s) x."""
        x, single = self._as_points(x, self.k)
        f = self._predict(x, error=False)[0]
        return float(f[0]) if single else f

    def predicterr_normalized(self, x):
        """Standard error sqrt(sigma^2 (1 - psi^T Psi^-1 psi)) of the prediction at normalised point(s) x."""
        x, single = self._as_points(x, self.k)
        s = self._predict(x, mean=False)[1]
        return float(s[0]) if single else s

    def _incumbent(self):
        """Best observed (normalised) response, the reference for expected improvement."""
        return np.min(self.y)

    def _improvement_error(self, x):
        """Standard error used by expected improvement and error-based infill (re-interpolation for regression)."""
        return self._predict(x, mean=False, improvement=True)[1]

    def _log_expected_improvement(self, x):
        """ln of the expected improvement at normalised points x (m, k)."""
        return log_expected_improvement(*self._predict(x, improvement=True), self._incumbent())
