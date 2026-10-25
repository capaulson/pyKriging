"""
Regression Kriging: Kriging with a regression constant Lambda added to the diagonal of the correlation
matrix, so noisy observations are filtered rather than interpolated (Forrester et al. 2006b; 2007, sec. 5a).
Lambda is tuned together with theta and p; the error estimate becomes sigma^2 (1 + Lambda - psi^T Psi^-1 psi).
Expected improvement and error-based infill use the re-interpolation error instead (Forrester et al. 2006b;
2007, section 5b), which excludes the noise, so infill does not re-sample existing points.
"""
__author__ = 'chrispaulson'
import numpy as np
from scipy.linalg import LinAlgError, lapack

from .krige import kriging
from .matrixops import EPS


class regression_kriging(kriging):
    _search_settings = {'pso': dict(pop=150, hood=20), 'ga': dict(pop=50)}
    _U0 = None

    def __init__(self, X, y, testfunction=None, name='', testPoints=None, **kwargs):
        self.Lambda = 0.0
        kriging.__init__(self, X, y, testfunction, name, testPoints, **kwargs)
        self.thetamin, self.pmin = 1e-4, 1.81231

    def _hyper(self):
        return np.r_[self.theta, self.pl, self.Lambda]

    def _nugget(self):
        return self.Lambda

    def _err_offset(self):
        return self.Lambda

    def _bounds(self):
        lo, hi = kriging._bounds(self)
        return np.append(lo, 0.0), np.append(hi, 1.0)

    def _candidate_nuggets(self, candidates):
        return candidates[:, 2 * self.k]

    def update(self, values):
        """Set [theta_1..theta_k, p_1..p_k, Lambda] and refit."""
        values = np.asarray(values, dtype=float).ravel()
        if len(values) != 2 * self.k + 1:
            raise ValueError(f'expected {2 * self.k + 1} hyper-parameters (theta, p, Lambda), got {len(values)}')
        self.Lambda = float(values[2 * self.k])
        kriging.update(self, values)

    def updatePsi(self):
        kriging.updatePsi(self)
        self._U0 = None  # re-interpolation factor, rebuilt on demand

    def neglikelihood(self):
        kriging.neglikelihood(self)
        # re-interpolation variance (Forrester et al. 2007, eq. 5.4) = w^T Psi_0 w / n, with w = (Psi_0 + Lambda I)^-1
        # (y - 1 mu); held fixed, like sigma^2, while multi-point infill imputes points
        self._s2ri = self.SigmaSqr - self.Lambda * (self._weights @ self._weights) / self.n

    def _incumbent(self):
        """Best regressed response: min of y - Lambda Psi^-1 (y - 1 mu), the trend through the noisy data."""
        return np.min(self.y - self.Lambda * self._weights)

    def _improvement_variance(self, psi):
        """Re-interpolation variance sigma_ri^2 (1 - psi^T (Psi_0 + delta I)^-1 psi): the error of an interpolation
        of the regressed data."""
        if self._U0 is None:
            for delta in ((10 + self.n) * EPS, 1e-10, 1e-8, 1e-6):  # Psi_0 alone may be numerically singular
                U0, info = lapack.dpotrf(self.Psi + (delta - self.Lambda - self._jitter) * np.eye(self.n),
                                         lower=0, clean=1)
                if not info:
                    break
            else:
                raise LinAlgError('correlation matrix is not positive definite')
            self._U0 = U0  # only a successful factorisation is cached
        v = lapack.dtrtrs(self._U0, psi.T, lower=0, trans=1)[0]
        return self._s2ri * (1 - np.einsum('ij,ij->j', v, v))

    # names used by earlier releases
    regupdatePsi = updatePsi
    regneglikelihood = neglikelihood
    regression_predicterr_normalized = kriging.predicterr_normalized
