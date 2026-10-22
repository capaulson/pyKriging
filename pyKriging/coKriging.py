"""
Two-level co-kriging (Forrester, Sobester & Keane 2007, section 3, after Kennedy & O'Hagan 2000).

The expensive function is modelled as Z_e(x) = rho Z_c(x) + Z_d(x) (eq. 3.1): a Gaussian process of the
cheap function scaled by rho plus a Gaussian process of the difference. Hyper-parameters are estimated in
two steps: theta_c and p_c from the cheap data alone (eq. 3.6), then theta_d, p_d and rho from the
differences d = y_e - rho y_c(X_e) (eqs. 3.7-3.10); for given theta_d, p_d the MLEs of rho and mu_d are the
generalised least-squares fit of y_e on [1, y_c(X_e)], so only theta_d and p_d are searched. Predictions use
the joint covariance matrix C (eqs. 3.2, 3.11, 3.12) and interpolate the expensive data. By default the means
are Kennedy & O'Hagan's per-level means (mu_c for cheap data, rho mu_c + mu_d for expensive data); mean='paper'
uses the single mu = 1^T C^-1 y / 1^T C^-1 1 of eq. (3.11). With per-level means and X_e a subset of X_c, the
predictor and its error are exactly rho y_c(x) + y_d(x) and rho^2 s_c^2(x) + s_d^2(x) (Le Gratiet & Garnier 2014),
which are computed level by level: as stable as ordinary kriging, with no joint matrix to factorise.

    ck = coKriging(Xc, yc, Xe, ye)   # cheap data (nc, k), expensive data (ne, k), ideally Xe a subset of Xc
    ck.train()
    ck.predict(x), ck.predict_var(x), ck.expimp(x)
"""
__author__ = 'cpaulson'
import numpy as np
from scipy.linalg import LinAlgError, lapack

from .krige import _search, kriging
from .matrixops import EPS, PENALTY, correlation, expected_improvement, matrixops


class _Level(kriging):
    """Ordinary kriging of one fidelity level, on data already in co-kriging units (no further scaling)."""

    def normalizeData(self):
        self._xlo, self._xspan, self._fixed = np.zeros(self.k), np.ones(self.k), np.zeros(self.k, dtype=bool)
        self.normRange, self.ynormRange = [[0.0, 1.0]] * self.k, [0.0, 1.0]


class coKriging():
    def __init__(self, Xc, yc, Xe, ye, seed=None, mean='koh'):
        """
        Xc, yc: cheap samples (nc, k) and responses; Xe, ye: expensive samples (ne, k) and responses.
        1-D X arrays are one-variable problems. Both levels share one scaling of x and one of y, so rho
        keeps the units of the data. seed: seed for the hyper-parameter searches (None = random).
        mean: 'koh' (per-level means) or 'paper' (one mean, eq. 3.11).
        """
        if mean not in ('koh', 'paper'):
            raise ValueError("mean must be 'koh' or 'paper'")
        self.mean = mean
        Xc, Xe = np.array(Xc, dtype=float), np.array(Xe, dtype=float)
        Xc, Xe = Xc.reshape(len(Xc), -1), Xe.reshape(len(Xe), -1)
        yc, ye = np.array(yc, dtype=float).ravel(), np.array(ye, dtype=float).ravel()
        if len(yc) != len(Xc) or len(ye) != len(Xe) or Xc.shape[1] != Xe.shape[1]:
            raise ValueError(f'inconsistent shapes: Xc {Xc.shape}, yc {yc.shape}, Xe {Xe.shape}, ye {ye.shape}')
        X, y = np.vstack([Xc, Xe]), np.concatenate([yc, ye])
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise ValueError('Xc, yc, Xe and ye must be finite (NaN or inf)')
        self._xlo, hi = X.min(axis=0), X.max(axis=0)
        self._xspan = np.where(hi > self._xlo, hi - self._xlo, 1.0)
        self._ylo, self._yspan = y.min(), (np.ptp(y) or 1.0)
        self.Xc, self.Xe, self.yc, self.ye = self.normX(Xc), self.normX(Xe), self.normy(yc), self.normy(ye)
        (self.nc, self.k), self.ne = self.Xc.shape, len(self.ye)
        self.rng = np.random.default_rng(seed)
        # a smooth (e.g. linear) difference d drives the MLE towards theta_d -> 0 along a flat likelihood ridge
        # where Psi_d becomes numerically singular; theta >= 1e-4 keeps C well conditioned without biasing rho
        self.thetamin, self.thetamax, self.pmin, self.pmax = 1e-4, 100.0, 1.0, 2.0
        self.rho = None

    def normX(self, X):
        """Real-world point(s) -> model units."""
        return (np.asarray(X, dtype=float) - self._xlo) / self._xspan

    def normy(self, y):
        return (np.asarray(y, dtype=float) - self._ylo) / self._yspan

    def inversenormy(self, y):
        return y * self._yspan + self._ylo

    def _level(self, X, y):
        level = _Level(X, y, seed=self.rng)
        level.thetamin, level.thetamax, level.pmin, level.pmax = self.thetamin, self.thetamax, self.pmin, self.pmax
        return level

    def train(self, optimizer='pso'):
        """Maximum-likelihood estimates for the cheap model (eq. 3.6), then for theta_d, p_d and rho (eq. 3.10)."""
        self.cheap = self._level(self.Xc, self.yc)
        self.cheap.train(optimizer)
        # y_c at X_e: observed where the points coincide, else the cheap model's prediction (paper, footnote 3)
        gap = np.abs(self.Xe[:, None, :] - self.Xc[None, :, :]).max(axis=2)
        match = gap.argmin(axis=1)
        collocated = gap[np.arange(self.ne), match] < 1e-12
        self._recursive = self.mean == 'koh' and bool(collocated.all())
        self.yc_e = np.where(collocated, self.yc[match], self.cheap.predict_normalized(self.Xe))
        # a (nearly) constant y_c(X_e) carries no information on rho: then rho = 0 and y_e is kriged on its own
        self._use_rho = np.ptp(self.yc_e) > np.sqrt(EPS) * max(1.0, np.abs(self.yc_e).max())
        self.diff = self._level(self.Xe, self.ye - self.yc_e)
        lo, hi = self.diff._bounds()
        self.update(_search(self._difference_nll, lo, hi, self.k, self.rng, optimizer, kriging._search_settings[optimizer])[0])

    def _gls(self, U):
        """With U the upper Cholesky factor of Psi_d: MLEs [mu_d, rho] (GLS of y_e on [1, y_c(X_e)]) and the
        squared norm of the whitened residual U^-T (y_e - mu_d - rho y_c(X_e))."""
        a = lapack.dtrtrs(U, np.column_stack([np.ones(self.ne), self.yc_e, self.ye]), lower=0, trans=1)[0]
        F = a[:, :2] if self._use_rho else a[:, :1]
        beta = np.linalg.lstsq(F, a[:, 2], rcond=None)[0]
        res = a[:, 2] - F @ beta
        return np.r_[beta, 0.0][:2], res @ res

    def _difference_nll(self, candidates):
        """Concentrated -ln L of the difference model (eq. 3.10) for candidates [theta_d, p_d], with mu_d,
        rho and sigma_d^2 at their MLEs; PENALTY where Psi_d is not positive definite."""
        k, d, out = self.k, self.diff, np.full(len(candidates), PENALTY)
        work = np.empty((self.ne, self.ne))
        step = max(1, 4_000_000 // max(1, d._d.size))  # bound the correlation block, as kriging does
        for s in range(0, len(candidates), step):
            for i, r in enumerate(d._pair_correlations(candidates[s:s + step, :k], candidates[s:s + step, k:2 * k]), s):
                U = d._cholesky(r, EPS, work)
                if U is not None:
                    nll = 0.5 * self.ne * np.log(self._gls(U)[1] / self.ne) + np.sum(np.log(U.diagonal()))
                    out[i] = nll if np.isfinite(nll) else PENALTY
        return out

    def update(self, values):
        """Set [theta_d (k), p_d (k)], estimate rho and refit the difference model and the joint covariance C (eq. 3.2).

        C is factorised by block elimination on the cheap block A = sigma_c^2 Psi_c, whose Cholesky factor U_c
        the cheap model already holds. The Schur complement S = rho^2 sigma_c^2 G + sigma_d^2 Psi_d involves the
        cheap model's posterior correlation G at X_e, which is positive semi-definite (and ~0 where X_e lies in
        X_c); it is projected onto that cone so rounding cannot make C indefinite."""
        values = np.asarray(values, dtype=float)[:2 * self.k]
        self.diff.update(values)  # fixes Psi_d and its factor; mu_d, rho and y_d follow from the GLS fit
        (mud, rho), _ = self._gls(self.diff.U)
        self.rho = float(rho)
        self.diff.y = self.ye - self.rho * self.yc_e
        self.diff.update(values)
        c, d, rho = self.cheap, self.diff, self.rho
        # y_e exactly affine in y_c leaves no difference process (sigma_d^2 = 0) and C singular: keep a trace of it
        d.SigmaSqr = max(d.SigmaSqr, 1e-12 * rho ** 2 * c.SigmaSqr)
        self.thetac, self.pc, self.muc, self.SigmaSqrc = c.theta, c.pl, c.mu, c.SigmaSqr
        self.thetad, self.pd, self.mud, self.SigmaSqrd = d.theta, d.pl, d.mu, d.SigmaSqr
        if self._recursive:
            self.mu = rho * c.mu + d.mu  # mean of the expensive process
            return
        self._Pce = Pce = correlation(self.Xc, self.Xe, c.theta, c.pl)
        Pee = correlation(self.Xe, self.Xe, c.theta, c.pl) + np.eye(self.ne) * EPS
        self._V = lapack.dtrtrs(c.U, Pce, lower=0, trans=1)[0]  # U_c^-T Psi_c(X_c, X_e)
        lam, Q = np.linalg.eigh(Pee - self._V.T @ self._V)
        S = rho ** 2 * c.SigmaSqr * (Q * np.maximum(lam, 0.0)) @ Q.T + d.SigmaSqr * d.Psi
        self._US, info = lapack.dpotrf(S, lower=0, clean=1)
        if info:
            raise LinAlgError('co-kriging covariance matrix is not positive definite')
        # generalised least-squares means: E[y] = F beta with F = 1 (eq. 3.11: one mu) or, for Kennedy & O'Hagan's
        # per-level means, rows [1, 0] for cheap data and [rho, 1] for expensive data (beta = [mu_c, mu_d])
        y, nc, ne = np.concatenate([self.yc, self.ye]), self.nc, self.ne
        F = np.ones((nc + ne, 1)) if self.mean == 'paper' else np.r_[np.c_[np.ones(nc), np.zeros(nc)], np.c_[np.full(ne, rho), np.ones(ne)]]
        CiF, Ciy = np.split(self._solve(np.column_stack([F, y])), [F.shape[1]], axis=1)
        beta = np.linalg.solve(F.T @ CiF, F.T @ Ciy[:, 0])
        self._weights = self._solve((y - F @ beta)[:, None])[:, 0]  # C^-1 (y - F beta), without cancellation
        self.mu = beta[0] if self.mean == 'paper' else rho * beta[0] + beta[1]  # mean of the expensive process

    @property
    def C(self):
        """The joint covariance matrix of eq. (3.2) (built on request; predictions use its block factorisation)."""
        c, d, rho = self.cheap, self.diff, self.rho
        Pce = correlation(self.Xc, self.Xe, c.theta, c.pl)
        Pee = correlation(self.Xe, self.Xe, c.theta, c.pl) + np.eye(self.ne) * EPS
        return np.block([[c.SigmaSqr * c.Psi, rho * c.SigmaSqr * Pce],
                         [rho * c.SigmaSqr * Pce.T, rho ** 2 * c.SigmaSqr * Pee + d.SigmaSqr * d.Psi]])

    def _solve(self, v):
        """C^-1 v for v (nc + ne, m) by block elimination."""
        c, rho, nc = self.cheap, self.rho, self.nc
        t = lapack.dtrtrs(c.U, lapack.dtrtrs(c.U, v[:nc], lower=0, trans=1)[0], lower=0)[0] / c.SigmaSqr  # A^-1 v1
        u = lapack.dpotrs(self._US, v[nc:] - rho * c.SigmaSqr * self._Pce.T @ t, lower=0)[0]
        return np.vstack([t - rho * lapack.dtrtrs(c.U, self._V @ u, lower=0)[0], u])

    def _predict(self, x):
        """Predictor (eq. 3.11) and its standard error (eq. 3.12) in model units at points x (m, k):
        s^2 = rho^2 sigma_c^2 (1 - |v|^2) + sigma_d^2 - |U_S^-T r|^2 with v = U_c^-T psi_c(X_c, x) and
        r = rho^2 sigma_c^2 (psi_c(X_e, x) - V^T v) + sigma_d^2 psi_d(X_e, x)."""
        c, d, rho = self.cheap, self.diff, self.rho
        if self._recursive:
            (fc, sc), (fd, sd) = c._predict(x), d._predict(x)
            return rho * fc + fd, np.sqrt(rho ** 2 * sc ** 2 + sd ** 2)
        step = max(1, 2_000_000 // ((self.nc + self.ne) * self.k))
        f, s = np.empty(len(x)), np.empty(len(x))
        for i in range(0, len(x), step):
            xi = x[i:i + step]
            psi_c, psi_ce, psi_de = (correlation(xi, self.Xc, c.theta, c.pl), correlation(xi, self.Xe, c.theta, c.pl),
                                     correlation(xi, self.Xe, d.theta, d.pl))
            f[i:i + step] = self.mu + np.hstack([rho * c.SigmaSqr * psi_c, rho ** 2 * c.SigmaSqr * psi_ce
                                                  + d.SigmaSqr * psi_de]) @ self._weights
            v = lapack.dtrtrs(c.U, psi_c.T, lower=0, trans=1)[0]
            r = rho ** 2 * c.SigmaSqr * (psi_ce.T - self._V.T @ v) + d.SigmaSqr * psi_de.T
            w = lapack.dtrtrs(self._US, r, lower=0, trans=1)[0]
            s2 = rho ** 2 * c.SigmaSqr * (1 - np.einsum('ij,ij->j', v, v)) + d.SigmaSqr - np.einsum('ij,ij->j', w, w)
            s[i:i + step] = np.sqrt(np.abs(s2))
        return f, s

    def _fitted(self):
        if self.rho is None:
            raise RuntimeError('the model is not fitted yet: call train() first')

    def predict(self, X):
        """Co-kriging prediction of the expensive function at real-world point(s) X."""
        self._fitted()
        x, single = matrixops._as_points(self.normX(X), self.k)
        f = self.inversenormy(self._predict(x)[0])
        return float(f[0]) if single else f

    def predict_var(self, X):
        """Standard error of the prediction (model units of y) at real-world point(s) X."""
        self._fitted()
        x, single = matrixops._as_points(self.normX(X), self.k)
        s = self._predict(x)[1]
        return float(s[0]) if single else s

    def expimp(self, x):
        """Expected improvement over the best expensive observation at normalised point(s) x (section 5)."""
        self._fitted()
        x, single = matrixops._as_points(x, self.k)
        ei = expected_improvement(*self._predict(x), np.min(self.ye))
        return float(ei[0]) if single else ei
