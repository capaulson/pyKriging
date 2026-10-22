"""
Independent float64 reference implementations for the pyKriging test suite.

Nothing here imports pyKriging: the library is checked against a second, deliberately plain implementation of the
same mathematics (explicit matrices and solves, where the library uses packed pair distances, cached weights,
batched likelihoods and closed forms).

[F07] Forrester, Sobester & Keane (2007), Multi-fidelity optimization via surrogate modelling, Proc. R. Soc. A
      463:3251. Equation numbers refer to it.
[J98] Jones, Schonlau & Welch (1998), Efficient global optimization of expensive black-box functions, J. Global
      Optim. 13:455 (expected improvement).
[F06] Forrester, Keane & Bressloff (2006), Design and analysis of 'noisy' computer experiments, AIAA J. 44:2331
      (regression kriging Psi + Lambda I, and re-interpolation).

Factorisations use the upper Cholesky form, Psi = U^T U, as the lower form of the OpenBLAS bundled with
numpy/scipy releases of 2021 can crash on arm64.

Library conventions reproduced here on purpose: inputs and responses are min-max normalised to [0, 1]; Psi has
1 + eps (kriging) or 1 + Lambda (regression kriging) on its diagonal; the error estimate is
s^2 = sigma^2 (offset + 1 - psi^T Psi^-1 psi) without the mean-estimation term.
"""
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.special import ndtr

EPS = float(np.finfo(float).eps)


# ------------------------------------------------------------------ data
def lhs(n, k, seed):
    """Jittered Latin hypercube in [0, 1)^k drawn with numpy only."""
    rng = np.random.default_rng(seed)
    return (np.argsort(rng.random((n, k)), axis=0) + rng.random((n, k))) / n


def normalise(X, y):
    """Min-max normalised X, y and the ranges (xlo, xhi), (ylo, yhi)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    xlo, xhi, ylo, yhi = X.min(0), X.max(0), y.min(), y.max()
    return (X - xlo) / (xhi - xlo), (y - ylo) / (yhi - ylo), (xlo, xhi), (ylo, yhi)


def corr(A, B, theta, p):
    """exp(-sum_j theta_j |a_j - b_j|^p_j) between point sets A (m, k) and B (n, k) [F07 eq. 2.2]."""
    A, B = np.atleast_2d(np.asarray(A, dtype=float)), np.atleast_2d(np.asarray(B, dtype=float))
    theta, p = np.asarray(theta, dtype=float), np.asarray(p, dtype=float)
    return np.exp(-np.sum(theta * np.abs(A[:, None, :] - B[None, :, :]) ** p, axis=2))


# ------------------------------------------------------------------ expected improvement
def ei(f, s, ymin):
    """EI = (ymin - f) Phi(u) + s phi(u), u = (ymin - f) / s [J98 eq. 15]; 0 where s = 0. Accurate for u > -5."""
    f, s = np.broadcast_arrays(np.asarray(f, dtype=float), np.asarray(s, dtype=float))
    out = np.zeros(f.shape)
    ok = s > 0
    u = (ymin - f[ok]) / s[ok]
    out[ok] = (ymin - f[ok]) * ndtr(u) + s[ok] * np.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)
    return out


def log_h(u, terms=3000):
    """ln(u Phi(u) + phi(u)), the log of the expected improvement of a unit-variance prediction, accurate to
    ~1e-14 for every u: directly for u >= -1, else from Laplace's continued fraction for the Mills ratio
    R(t) = 1/(t + 1/(t + 2/(t + 3/...))) with t = -u, which gives h/phi = 1 - t R = 1/(E D) with
    D = t + 1/E and E = t + 2/(t + 3/(t + ...)): positive terms only, so no cancellation."""
    u = np.atleast_1d(np.asarray(u, dtype=float))
    out = np.empty_like(u)
    near = u >= -1
    v = u[near]
    out[near] = np.log(v * ndtr(v) + np.exp(-0.5 * v * v) / math.sqrt(2 * math.pi))
    t = -u[~near]
    tail = t.copy()  # backward evaluation of E's continued fraction
    for j in range(terms, 2, -1):
        tail = t + j / tail
    E = t + 2.0 / tail
    D = t + 1.0 / E
    out[~near] = -0.5 * t * t - 0.5 * math.log(2 * math.pi) - np.log(E) - np.log(D)
    return out


# ------------------------------------------------------------------ ordinary and regression kriging
class RefKriging:
    """Kriging with fixed hyper-parameters on normalised data [F07 section 2, eqs 3.4-3.6 for one level]:

        mu = 1' Psi^-1 y / 1' Psi^-1 1,   sigma^2 = (y - 1 mu)' Psi^-1 (y - 1 mu) / n,
        NegLnLike = n/2 ln sigma^2 + 1/2 ln|Psi|,   f(x) = mu + psi' Psi^-1 (y - 1 mu),
        s^2(x) = sigma^2 (offset + 1 - psi' Psi^-1 psi).

    lam > 0 puts 1 + lam on the diagonal (regression kriging [F06]) instead of 1 + nugget."""

    def __init__(self, Xn, yn, theta, p, nugget=EPS, lam=0.0):
        self.y = np.asarray(yn, dtype=float).ravel()
        self.X = np.asarray(Xn, dtype=float).reshape(len(self.y), -1)
        self.theta, self.p, self.lam = np.asarray(theta, dtype=float), np.asarray(p, dtype=float), lam
        self.n = len(self.y)
        self.Psi0 = corr(self.X, self.X, self.theta, self.p)
        self.Psi = self.Psi0.copy()
        np.fill_diagonal(self.Psi, 1.0 + (lam if lam else nugget))
        self.U = cholesky(self.Psi, lower=False)  # Psi = U^T U; LinAlgError if not positive definite
        self.L = self.U.T
        one = np.ones(self.n)
        a = solve_triangular(self.U, one, trans='T')  # U^-T 1
        b = solve_triangular(self.U, self.y, trans='T')
        self.mu = float(a @ b / (a @ a))
        r = b - self.mu * a
        self.sigma2 = float(r @ r / self.n)
        self.nll = float(0.5 * self.n * np.log(self.sigma2) + np.sum(np.log(np.diag(self.L))))
        self.w = solve_triangular(self.U, r)  # Psi^-1 (y - 1 mu)

    def _points(self, x):
        return np.asarray(x, dtype=float).reshape(-1, self.X.shape[1])

    def psi(self, x):
        return corr(self._points(x), self.X, self.theta, self.p)

    def predict(self, x):
        return self.mu + self.psi(x) @ self.w

    def mse(self, x, offset=0.0):
        v = solve_triangular(self.U, self.psi(x).T, trans='T')
        return self.sigma2 * (1.0 + offset - np.sum(v * v, axis=0))

    def std(self, x, offset=0.0):
        return np.sqrt(np.abs(self.mse(x, offset)))

    def ei(self, x, ymin=None):
        """EI in normalised units with s = sqrt|s^2| and ymin = min(y)."""
        return ei(self.predict(x), self.std(x), float(np.min(self.y)) if ymin is None else ymin)

    def regressed(self):
        """Regression predictions at the samples, mu + Psi0 Psi^-1 (y - 1 mu) [F06]."""
        return self.mu + self.Psi0 @ self.w

    def reinterpolation_std(self, x, delta=1e-12):
        """Error of a kriging interpolation of the regressed data with the same theta, p and mu [F06; F07 sec. 5b]:
        sigma_ri^2 = (y_r - 1 mu)' Psi0^-1 (y_r - 1 mu) / n and s_ri^2 = sigma_ri^2 (1 - psi' Psi0^-1 psi);
        delta I keeps Psi0 numerically positive definite."""
        cf = cho_factor(self.Psi0 + delta * np.eye(self.n))
        r = self.regressed() - self.mu
        sigma2_ri = r @ cho_solve(cf, r) / self.n
        ps = self.psi(x)
        return np.sqrt(np.abs(sigma2_ri * (1.0 - np.sum(ps.T * cho_solve(cf, ps.T), axis=0))))


def concentrated_nll(Xn, yn, theta, p, nugget=EPS, lam=0.0):
    """Reference NegLnLike; +inf where Psi is not numerically positive definite or sigma^2 = 0."""
    try:
        value = RefKriging(Xn, yn, theta, p, nugget=nugget, lam=lam).nll
    except np.linalg.LinAlgError:
        return float('inf')
    return value if np.isfinite(value) else float('inf')


def nll_optimum(Xn, yn, theta_bounds, p_bounds, lam_bounds=None, n_starts=24, seed=0):
    """Best concentrated NegLnLike over the box by a dense random search in (log10 theta, p[, log10 Lambda]), plus
    isotropic starts (all theta_j equal, p = its upper bound), and L-BFGS-B polishing of the best starts. Returns
    (nll, theta, p[, Lambda]). The isotropic starts matter in higher dimensions, where random starts miss the long
    small-theta ridge of smooth functions."""
    k = np.asarray(Xn).reshape(len(np.ravel(yn)), -1).shape[1]
    lo = [np.log10(theta_bounds[0])] * k + [p_bounds[0]] * k
    hi = [np.log10(theta_bounds[1])] * k + [p_bounds[1]] * k
    if lam_bounds is not None:
        lo.append(np.log10(max(lam_bounds[0], 1e-12)))
        hi.append(np.log10(lam_bounds[1]))
    lo, hi = np.array(lo), np.array(hi)

    def f(v):
        lam = 10 ** v[-1] if lam_bounds is not None else 0.0
        return concentrated_nll(Xn, yn, 10 ** v[:k], v[k:2 * k], lam=lam)

    rng = np.random.default_rng(seed)
    starts = lo + (hi - lo) * rng.random((max(400, 60 * len(lo)), len(lo)))
    isotropic = np.tile(hi, (29, 1))
    isotropic[:, :k] = np.linspace(lo[0], hi[0], 29)[:, None]
    if lam_bounds is not None:
        isotropic[:, -1] = (lo[-1] + hi[-1]) / 2
    starts = np.vstack([starts, isotropic])
    values = np.array([f(v) for v in starts])
    best = (np.inf, None)
    for v in starts[np.argsort(values)[:n_starts if len(lo) <= 7 else n_starts // 3]]:  # polishing costs ~ dim^2
        r = minimize(lambda z: min(f(z), 1e300), v, method='L-BFGS-B', bounds=list(zip(lo, hi)))
        if r.fun < best[0]:
            best = (float(r.fun), r.x)
    v = best[1]
    out = (best[0], 10 ** v[:k], v[k:2 * k])
    return out + (10 ** v[-1],) if lam_bounds is not None else out


# ------------------------------------------------------------------ co-kriging [F07 section 3]
def gls(Psi, F, y):
    """Generalised least squares beta = (F' Psi^-1 F)^-1 F' Psi^-1 y and the residual quadratic form."""
    cf = cho_factor(Psi)
    PiF = cho_solve(cf, F)
    beta = np.linalg.solve(F.T @ PiF, PiF.T @ y)
    r = y - F @ beta
    return beta, float(r @ cho_solve(cf, r))


def difference_nll(Xe, ye, yce, theta, p):
    """Concentrated -ln L of the difference model, eq. (3.10), with mu_d and rho at their MLEs: for fixed Psi_d they
    are the GLS fit of y_e on [1, y_c(X_e)] (then sigma_d^2 = residual quadratic form / n_e). +inf if not PD."""
    Psi = corr(Xe, Xe, theta, p) + EPS * np.eye(len(ye))
    try:
        U = cholesky(Psi)
        (mud, rho), q = gls(Psi, np.column_stack([np.ones(len(ye)), yce]), ye)
    except np.linalg.LinAlgError:
        return float('inf'), np.nan, np.nan
    return 0.5 * len(ye) * np.log(q / len(ye)) + np.sum(np.log(np.diag(U))), rho, mud


def cokriging_predict(Xc, yc, Xe, ye, thc, pc, thd, pd, rho, s2c, s2d, x, mean='koh'):
    """Co-kriging predictor, mean squared error and the mean of the expensive process at points x from the joint
    covariance matrix C (eq. 3.2) at the given hyper-parameters, by a direct Cholesky solve (all in the same
    normalised units). mean='paper': one GLS mean
    mu = 1' C^-1 y / 1' C^-1 1 (eq. 3.11); 'koh': Kennedy & O'Hagan's per-level means, E[y_c] = mu_c and
    E[y_e] = rho mu_c + mu_d, by GLS. MSE = rho^2 s2c + s2d - c' C^-1 c (eq. 3.12). Correlation matrices carry
    eps on the diagonal, as the library's."""
    Xc, Xe, x = (np.asarray(a, dtype=float).reshape(len(a), -1) for a in (Xc, Xe, x))
    nc, ne = len(Xc), len(Xe)
    Pc = corr(Xc, Xc, thc, pc) + EPS * np.eye(nc)
    Pce = corr(Xc, Xe, thc, pc)
    Pee = corr(Xe, Xe, thc, pc) + EPS * np.eye(ne)
    Pd = corr(Xe, Xe, thd, pd) + EPS * np.eye(ne)
    C = np.block([[s2c * Pc, rho * s2c * Pce], [rho * s2c * Pce.T, rho ** 2 * s2c * Pee + s2d * Pd]])
    y = np.r_[yc, ye]
    if mean == 'paper':
        F, Fx = np.ones((nc + ne, 1)), np.ones((len(x), 1))
    else:
        F = np.r_[np.c_[np.ones(nc), np.zeros(nc)], np.c_[np.full(ne, rho), np.ones(ne)]]
        Fx = np.c_[np.full(len(x), rho), np.ones(len(x))]
    beta, _ = gls(C, F, y)
    cf = cho_factor(C)
    c = np.vstack([rho * s2c * corr(Xc, x, thc, pc),
                   rho ** 2 * s2c * corr(Xe, x, thc, pc) + s2d * corr(Xe, x, thd, pd)])
    f = Fx @ beta + c.T @ cho_solve(cf, y - F @ beta)
    mse = rho ** 2 * s2c + s2d - np.sum(c * cho_solve(cf, c), axis=0)
    return f, mse, float(beta[0] if mean == 'paper' else rho * beta[0] + beta[1])


def _conc(Psi, y):
    n = len(y)
    try:
        cf = cho_factor(Psi)
    except np.linalg.LinAlgError:
        return np.inf, np.nan, np.nan
    one = np.ones(n)
    mu = one @ cho_solve(cf, y) / (one @ cho_solve(cf, one))
    r = y - mu
    s2 = r @ cho_solve(cf, r) / n
    if not s2 > 0:
        return np.inf, np.nan, np.nan
    return 0.5 * n * np.log(s2) + np.sum(np.log(np.diag(cf[0]))), mu, s2


class RefCoKriging:
    """Two-level co-kriging of [F07] section 3 fitted from scratch (p = 2, X_e a subset of X_c, one mean, eq. 3.11):
    log10 theta_c by grid search + polish of (3.6), then log10 theta_d and rho by grid search + polish of (3.10)
    over log10 theta in [-3, 2] and rho in [-5, 5]. The paper's GA reported rho = 1.87 on its 1-D example; the
    exact bounded MLE is ~2.0 (truth 1/A = 2)."""

    def __init__(self, Xc, yc, Xe, ye, lt_bounds=(-3.0, 2.0), rho_bounds=(-5.0, 5.0)):
        self.Xc, self.Xe = np.asarray(Xc, float).reshape(len(Xc), -1), np.asarray(Xe, float).reshape(len(Xe), -1)
        self.yc, self.ye = np.asarray(yc, float), np.asarray(ye, float)
        nc, ne = len(self.yc), len(self.ye)
        idx = [int(np.argmin(np.sum(np.abs(self.Xc - xe), axis=1))) for xe in self.Xe]
        assert np.allclose(self.Xc[idx], self.Xe), 'the reference needs X_e to be a subset of X_c'
        yce = self.yc[idx]
        Ic, Ie = np.eye(nc) * EPS, np.eye(ne) * EPS

        def fc(lt):
            return _conc(corr(self.Xc, self.Xc, [10 ** lt], [2.0]) + Ic, self.yc)[0]

        grid = np.linspace(*lt_bounds, 201)
        lc = grid[int(np.argmin([fc(t) for t in grid]))]
        lc = minimize(lambda v: fc(v[0]), [lc], method='L-BFGS-B', bounds=[lt_bounds]).x[0]
        _, self.mu_c, self.s2_c = _conc(corr(self.Xc, self.Xc, [10 ** lc], [2.0]) + Ic, self.yc)

        def fd(v):
            return _conc(corr(self.Xe, self.Xe, [10 ** v[0]], [2.0]) + Ie, self.ye - v[1] * yce)[0]

        start = min((fd((a, b)), a, b) for a in np.linspace(*lt_bounds, 61) for b in np.linspace(*rho_bounds, 201))[1:]
        r = minimize(fd, start, method='L-BFGS-B', bounds=[lt_bounds, rho_bounds])
        ld, self.rho = r.x
        _, self.mu_d, self.s2_d = _conc(corr(self.Xe, self.Xe, [10 ** ld], [2.0]) + Ie, self.ye - self.rho * yce)
        self.theta_c, self.theta_d = 10 ** lc, 10 ** ld

    def predict(self, x):
        return cokriging_predict(self.Xc, self.yc, self.Xe, self.ye, [self.theta_c], [2.0], [self.theta_d], [2.0],
                                 self.rho, self.s2_c, self.s2_d, np.atleast_1d(x), mean='paper')[0]

    def mse(self, x):
        return cokriging_predict(self.Xc, self.yc, self.Xe, self.ye, [self.theta_c], [2.0], [self.theta_d], [2.0],
                                 self.rho, self.s2_c, self.s2_d, np.atleast_1d(x), mean='paper')[1]


def forrester_fe(x):
    """The expensive function of [F07] section 3a, (6x - 2)^2 sin(12x - 4)."""
    x = np.asarray(x, dtype=float)
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def forrester_fc(x, A=0.5, B=10.0, C=-5.0):
    """The cheap function A f_e + B (x - 0.5) - C ([F07] section 3a: A = 0.5, B = 10, C = -5)."""
    x = np.asarray(x, dtype=float)
    return A * forrester_fe(x) + B * (x - 0.5) - C


PAPER_XC = np.linspace(0, 1, 11)
PAPER_XE = np.array([0.0, 0.4, 0.6, 1.0])

# A Morris-Mitchell optimal Latin hypercube of 40 points in 5-D (samplingplan(5, random_state=3).optimallhc(40) of
# pyKriging 0.2.0), stored as the bin index of every coordinate, x = (rank + 0.5) / 40, so that tests using it do
# not depend on the library's plan generator.
MAXIMIN_LHS_40x5 = (np.array([
    [18, 35, 29, 33, 22, 13, 34, 2, 7, 24, 21, 27, 38, 3, 4, 23, 0, 39, 26, 32, 19, 17, 16, 25, 31, 30, 15, 11, 10, 20,
     5, 1, 28, 37, 12, 14, 36, 9, 6, 8],
    [38, 33, 22, 37, 1, 24, 5, 13, 21, 31, 11, 16, 17, 23, 15, 27, 19, 26, 39, 30, 12, 9, 2, 0, 28, 6, 32, 7, 36, 18,
     34, 4, 20, 14, 10, 29, 8, 35, 25, 3],
    [11, 7, 10, 30, 6, 24, 35, 19, 21, 39, 33, 0, 18, 16, 38, 2, 8, 29, 25, 14, 31, 4, 34, 20, 36, 9, 12, 27, 32, 17,
     22, 23, 28, 3, 13, 37, 26, 5, 1, 15],
    [34, 19, 1, 39, 32, 33, 11, 38, 0, 26, 37, 27, 10, 5, 23, 12, 17, 22, 6, 36, 2, 9, 24, 16, 4, 35, 3, 14, 13, 25,
     20, 8, 21, 15, 18, 7, 31, 29, 28, 30],
    [8, 18, 9, 13, 35, 23, 10, 32, 36, 20, 11, 24, 28, 1, 17, 37, 21, 4, 16, 33, 22, 14, 31, 19, 30, 6, 26, 3, 34, 0,
     12, 27, 38, 2, 39, 7, 25, 29, 5, 15]]).T + 0.5) / 40


# ------------------------------------------------------------------ Morris-Mitchell criteria [F07 section 4]
def distances(X, p=1):
    return pdist(np.asarray(X, float), 'cityblock' if p == 1 else 'euclidean')


def mm_key(X, p=1):
    """Morris & Mitchell (1995) definition 1.2B as a sort key: maximise d_1, then minimise J_1, then maximise d_2,
    ...; a smaller key is more space-filling."""
    values, counts = np.unique(np.round(distances(X, p), 9), return_counts=True)
    return tuple(v for pair in zip(-values, counts) for v in pair)


def mmphi(X, q=2, p=1):
    """Phi_q = (sum over point pairs of d^-q)^(1/q)."""
    return float(np.sum(distances(X, p) ** (-float(q))) ** (1.0 / q))
