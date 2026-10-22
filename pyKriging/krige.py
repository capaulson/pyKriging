"""
Kriging (Gaussian-process) surrogate model.

    model = kriging(X, y, seed=0)  # X: (n, k) samples, y: (n,) responses (normalised internally to [0, 1])
    model.train()                  # maximum-likelihood theta and p
    model.predict(x)               # x: one point (k,) -> float, or many points (m, k) -> (m,) array
    model.predicterr(x)            # standard error of the prediction (same shapes, units of y)
    model.infill(2, method='ei')   # where to sample next (error reduction or expected improvement)

Notation and equations follow Forrester, Sobester & Keane (2007), Proc. R. Soc. A 463:3251.
"""
__author__ = 'chrispaulson'
import warnings

import numpy as np
from scipy.linalg import LinAlgError, lapack
from scipy.special import ndtr

from .matrixops import EPS, PENALTY, matrixops
from .samplingplan import samplingplan


def _pso(f, lo, hi, rng, pop, max_evals, hood=20, patience=10, inertia=0.5, cognitive=2.1, social=2.1, x0=None):
    """Minimise a batch objective f((pop, d) -> (pop,)) on the box [lo, hi] with a ring-topology particle swarm
    (the update rule of inspyred's PSO), starting from uniform random particles (or x0). Stops after `patience`
    generations without improving the best value by 1e-4, or after max_evals evaluations. Returns the best point
    and value found."""
    x = lo + (hi - lo) * rng.random((pop, len(lo))) if x0 is None else np.array(x0, dtype=float)
    fx = f(x)
    prev, best_x, best_f = x.copy(), x.copy(), fx.copy()
    ring = (np.arange(pop)[:, None] + np.arange(hood) - hood // 2) % pop
    evals, stall, record = pop, 0, np.round(fx.min(), 4)
    while evals < max_evals and stall < patience:
        leader = best_x[ring[np.arange(pop), np.argmin(best_f[ring], axis=1)]]
        x, prev = np.clip(x + inertia * (x - prev) + cognitive * rng.random(x.shape) * (best_x - x)
                          + social * rng.random(x.shape) * (leader - x), lo, hi), x
        fx = f(x)
        evals += pop
        better = fx <= best_f
        best_x[better], best_f[better] = x[better], fx[better]
        current = np.round(best_f.min(), 4)
        stall = stall + 1 if current >= record else 0
        record = min(record, current)
    i = np.argmin(best_f)
    return best_x[i], best_f[i]


def _ga(f, lo, hi, rng, pop, max_evals, elites=10, mutation_rate=0.05, patience=10):
    """Real-coded generational GA (rank selection, uniform crossover, uniform-reset mutation, elitism) with the
    same batch interface and stopping rule as _pso."""
    x = lo + (hi - lo) * rng.random((pop, len(lo)))
    fx = f(x)
    rank_prob = np.arange(pop, 0, -1.0) / (pop * (pop + 1) / 2)
    evals, stall, record = pop, 0, np.round(fx.min(), 4)
    while evals < max_evals and stall < patience:
        order = np.argsort(fx)
        x, fx = x[order], fx[order]
        mum, dad = x[rng.choice(pop, (2, pop - elites), p=rank_prob)]
        kids = np.where(rng.random(mum.shape) < 0.5, mum, dad)
        mutate = rng.random(kids.shape) < mutation_rate
        kids[mutate] = (lo + (hi - lo) * rng.random(kids.shape))[mutate]
        x, fx = np.vstack([x[:elites], kids]), np.concatenate([fx[:elites], f(kids)])
        evals += len(kids)
        current = np.round(fx.min(), 4)
        stall = stall + 1 if current >= record else 0
        record = min(record, current)
    i = np.argmin(fx)
    return x[i], fx[i]


def _search(f, lo, hi, k, rng, optimizer, settings, max_evals=30000):
    """
    Global minimisation of the batch objective f over the hyper-parameter box [lo, hi] = [theta (k), p (k), extras].
    Three searches, each polished by SLSQP, keep the best result: the chosen optimiser ('pso' or 'ga') over the box;
    a particle swarm over log10(theta); and a swarm over the isotropic model (one theta and one p for all dimensions)
    polished in the full box. No single search is reliable: the first misses small-theta optima, the second the
    p -> 2 ridge of smooth problems, and both miss nearly isotropic optima on that ridge in higher dimensions.
    """
    from scipy.optimize import minimize  # deferred: costs ~0.1 s to import and only training needs it

    def polish(g, z, fz, zlo, zhi):
        with warnings.catch_warnings():  # scipy < 1.10 warns when SLSQP steps past a bound (it clips)
            warnings.filterwarnings('ignore', message='Values in x were outside bounds')
            local = minimize(lambda v: float(g(v[None])[0]), z, method='SLSQP', bounds=list(zip(zlo, zhi)))
        return (local.x, local.fun) if local.fun < fz else (z, fz)

    def log_box(Z):  # [log10 theta (k), p (k), extras] -> box
        return np.column_stack([10.0 ** Z[:, :k], Z[:, k:]])

    def iso_box(Z):  # [log10 theta, p, extras] -> box
        return np.column_stack([np.repeat(10.0 ** Z[:, :1], k, axis=1), np.repeat(Z[:, 1:2], k, axis=1), Z[:, 2:]])

    def f_log(Z):
        return f(log_box(Z))

    def f_iso(Z):
        return f(iso_box(Z))

    first = _pso if optimizer == 'pso' else _ga
    found = [polish(f, *first(f, lo, hi, rng, max_evals=max_evals, patience=20, **settings), lo, hi)]
    log_lo = np.log10(np.maximum(lo[:k], 1e-300))
    zlo, zhi = np.r_[log_lo, lo[k:]], np.r_[np.log10(hi[:k]), hi[k:]]
    swarm = {key: settings[key] for key in ('pop', 'hood') if key in settings}
    z, fz = polish(f_log, *_pso(f_log, zlo, zhi, rng, max_evals=max_evals, patience=10, **swarm), zlo, zhi)
    found.append((log_box(z[None])[0], fz))
    if k > 1:  # in 1-D the isotropic model is the log10(theta) one
        zlo, zhi = np.r_[log_lo[0], lo[k], lo[2 * k:]], np.r_[np.log10(hi[0]), hi[k], hi[2 * k:]]
        z, fz = _pso(f_iso, zlo, zhi, rng, pop=50, max_evals=max_evals, patience=10)
        found.append(polish(f, iso_box(z[None])[0], fz, lo, hi))
    return min(found, key=lambda t: t[1])


def _evaluate(func, pts):
    """Evaluate a user test function on points (m, k): vectorised if it supports it, else point by point."""
    try:
        values = np.asarray(func(pts), dtype=float).ravel()
        if values.size == len(pts):
            return values
    except Exception:
        pass
    return np.array([np.ravel(func(p))[0] for p in pts], dtype=float)


class kriging(matrixops):
    _search_settings = {'pso': dict(pop=300, hood=20), 'ga': dict(pop=300)}

    def __init__(self, X, y, testfunction=None, name='', testPoints=None, seed=None, **kwargs):
        """
        X: training inputs (n, k); y: responses (n,); testfunction: optional ground truth f(x) used for history
        tracking and plots; testPoints: number of random points at which snapshot() tracks convergence;
        seed: seed for the stochastic hyper-parameter and infill searches (None = random).
        """
        X = np.array(X, dtype=float)
        self.X = X.reshape(len(X), -1)
        self.y = np.array(y, dtype=float).ravel()
        if len(self.y) != len(self.X):
            raise ValueError(f'X has {len(self.X)} samples but y has {len(self.y)} values')
        if not (np.isfinite(self.X).all() and np.isfinite(self.y).all()):
            raise ValueError('X and y must be finite (remove failed evaluations: NaN or inf)')
        self.testfunction, self.name = testfunction, name
        self.n, self.k = self.X.shape
        self.theta, self.pl = np.ones(self.k), np.full(self.k, 2.0)
        self.thetamin, self.thetamax, self.pmin, self.pmax = 1e-5, 100.0, 1.0, 2.0
        self.rng = np.random.default_rng(seed)
        self.sp = samplingplan(self.k, random_state=self.rng)
        self.normalizeData()

        self.history = {'points': [], 'neglnlike': [], 'theta': [], 'p': [], 'rsquared': [0], 'adjrsquared': [0],
                        'chisquared': [1000], 'lastPredictedPoints': [], 'avgMSE': [], 'pointData': None}
        self.testPoints = None
        if testPoints:  # real-world points spread over the data's bounding box
            self.testPoints = self.inversenormX(self._unit_points(testPoints))
            self.history['pointData'] = [
                {'point': p, 'actual': self.testfunction(p)[0] if self.testfunction else None,
                 'predicted': [], 'mse': [], 'gradient': []} for p in self.testPoints]
        matrixops.__init__(self)

    # ------------------------------------------------------------------ normalisation (model units are [0, 1])
    def normX(self, X):
        """Real-world point(s) -> model units [0, 1] per dimension."""
        return (np.asarray(X, dtype=float) - self._xlo) / self._xspan

    def inversenormX(self, X):
        """Model units -> real-world point(s)."""
        return np.asarray(X, dtype=float) * self._xspan + self._xlo

    def _yscale(self):
        lo, hi = self.ynormRange
        return hi - lo if hi > lo else 1.0

    def normy(self, y):
        """Real-world response(s) -> model units [0, 1]."""
        return (y - self.ynormRange[0]) / self._yscale()

    def inversenormy(self, y):
        """Model units -> real-world response(s)."""
        return y * self._yscale() + self.ynormRange[0]

    def _unit_points(self, m):
        """m random Latin hypercube points in model units, with constant input columns kept at their value."""
        pts = self.sp.rlh(m)
        pts[:, self._fixed] = 0.0
        return pts

    def normalizeData(self):
        """Record the data ranges and map X and y to [0, 1]."""
        self._xlo, hi = self.X.min(axis=0), self.X.max(axis=0)
        self._fixed = hi == self._xlo  # constant input columns
        self._xspan = np.where(self._fixed, 1.0, hi - self._xlo)
        self.normRange = [[lo, hi] for lo, hi in zip(self._xlo, hi)]
        self.ynormRange = [self.y.min(), self.y.max()]
        self.X, self.y = self.normX(self.X), self.normy(self.y)

    # ------------------------------------------------------------------ model state
    def addPoint(self, newX, newy, norm=True):
        """Add one observation (real-world units if norm=True) and refit; retrains if the current
        hyper-parameters no longer give a positive-definite correlation matrix."""
        if not (np.isfinite(newX).all() and np.isfinite(newy).all()):
            raise ValueError('newX and newy must be finite (NaN or inf)')
        if norm:
            newX, newy = self.normX(newX), self.normy(newy)
        self.X = np.vstack([self.X, np.reshape(newX, (1, self.k))])
        self.y = np.append(self.y, newy)
        self.updateData()
        try:
            self.updateModel()
        except Exception:
            self.train()

    def update(self, values):
        """Set the hyper-parameters [theta_1..theta_k, p_1..p_k] and refit."""
        values = np.asarray(values, dtype=float).ravel()
        if len(values) != len(self._bounds()[0]):
            raise ValueError(f'expected {len(self._bounds()[0])} hyper-parameters, got {len(values)}')
        self.theta, self.pl = values[:self.k].copy(), values[self.k:2 * self.k].copy()
        self.updateModel()

    def updateModel(self):
        """Refit Psi, its Cholesky factor and the likelihood estimates for the current data/hyper-parameters
        (raises Exception('bad params') if Psi is not positive definite)."""
        try:
            self.updatePsi()
            self.neglikelihood()
        except (LinAlgError, ValueError, FloatingPointError) as err:  # LinAlgError is a ValueError only from numpy 2
            raise Exception('bad params') from err

    # ------------------------------------------------------------------ predictions
    def predict(self, X, norm=True):
        """Prediction at real-world point(s) X (model units if norm=False): one point (k,) -> float,
        points (m, k) -> (m,) array."""
        x, single = self._as_points(X, self.k)
        f = self.inversenormy(self._predict(self.normX(x) if norm else x, error=False)[0])
        return float(f[0]) if single else f

    def predict_var(self, X, norm=True):
        """Standard error of the prediction at real-world point(s) X, in normalised units of y (y range = 1);
        see predicterr() for real-world units."""
        x, single = self._as_points(X, self.k)
        s = self._predict(self.normX(x) if norm else x, mean=False)[1]
        return float(s[0]) if single else s

    def predicterr(self, X):
        """Standard error of the prediction at real-world point(s) X, in the units of y."""
        return self.predict_var(X) * self._yscale()

    def infill_ei(self, X):
        """Expected improvement (in the units of y) at real-world point(s) X."""
        return self.expimp(self.normX(X)) * self._yscale()

    def expimp(self, x):
        """Expected improvement at normalised point(s) x, in normalised units of y."""
        x, single = self._as_points(x, self.k)
        ei = np.exp(self._log_expected_improvement(x))
        return float(ei[0]) if single else ei

    def weightedexpimp(self, x, w):
        """Weighted expected improvement w (y_min - f) Phi(u) + (1 - w) s phi(u) (Sobester et al. 2005) at
        normalised point(s) x; w = 0.5 is half the standard expected improvement."""
        x, single = self._as_points(x, self.k)
        f, s = self._predict(x, error=False)[0], self._improvement_error(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            imp = self._incumbent() - f
            u = imp / s
            wei = np.where(s > 0, w * imp * ndtr(u) + (1 - w) * s * np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi), 0.0)
        return float(wei[0]) if single else wei

    def infill_objective_mse(self, candidates, args=None):
        """Batch infill objective: minus the prediction error at normalised candidates (m, k)."""
        return -self._improvement_error(np.reshape(candidates, (-1, self.k)))

    def infill_objective_ei(self, candidates, args=None):
        """Batch infill objective: minus the expected improvement at normalised candidates (m, k)."""
        return -self.expimp(np.reshape(candidates, (-1, self.k)))

    def infill(self, points, method='error', addPoint=True):
        """
        Find `points` new sample locations (real-world units) maximising the prediction error
        (method='error') or the expected improvement (method='ei'). The searches run on ln(error) and ln(EI),
        which stay informative where EI underflows; for EI a tenth of the swarm starts near the best sample.
        Several points are found by imputing the model prediction at each new point in turn; the model is
        restored afterwards.
        """
        method = method.lower()
        if method not in ('ei', 'error'):
            raise ValueError(f"method must be 'ei' or 'error', not {method!r}")
        X0, y0, hyper = self.X.copy(), self.y.copy(), self._hyper()
        lo, hi, pop = np.zeros(self.k), np.where(self._fixed, 0.0, 1.0), 155  # constant input columns stay put
        new = np.zeros((points, self.k))

        def objective(c):
            return -self._log_expected_improvement(c) if method == 'ei' else -np.log(self._improvement_error(c))

        for i in range(points):
            x0 = None
            if method == 'ei':
                x0 = lo + (hi - lo) * self.rng.random((pop, self.k))
                x0[:pop // 10] = np.clip(self.X[np.argmin(self.y)] + 0.02 * self.rng.standard_normal((pop // 10, self.k)), lo, hi)
            with np.errstate(divide='ignore'):
                best, _ = _pso(objective, lo, hi, self.rng, pop=pop, max_evals=20000, hood=30, x0=x0)
            new[i] = self.inversenormX(best)
            if addPoint and i < points - 1:
                self._believe(best)
        self.X, self.y = X0, y0
        self.updateData()
        self.update(hyper)
        return new

    def _hyper(self):
        """The hyper-parameter vector that update() takes."""
        return np.r_[self.theta, self.pl]

    def _believe(self, x):
        """Kriging-believer imputation for multi-point infill: add normalised point x with its predicted value,
        keeping mu, sigma^2 and the hyper-parameters fixed. A point that leaves Psi numerically singular (typically
        one next to a sample, where the model is nearly exact) is imputed with a small jitter on the diagonal."""
        mu, s2, f = self.mu, self.SigmaSqr, self.predict_normalized(x)
        X, y = self.X, self.y
        self.X, self.y = np.vstack([X, x]), np.append(y, f)
        self.updateData()
        try:
            for self._jitter in (0.0, 1e-10, 1e-8, 1e-6):
                try:
                    self.updatePsi()
                    break
                except LinAlgError:
                    continue
            else:  # cannot impute: keep the model as it was
                self.X, self.y = X, y
                self.updateData()
                self.updatePsi()
        finally:
            del self._jitter
        self.mu, self.SigmaSqr = mu, s2
        self._weights = lapack.dtrtrs(self.U, lapack.dtrtrs(self.U, self.y - mu, lower=0, trans=1)[0], lower=0)[0]

    # ------------------------------------------------------------------ training
    def _bounds(self):
        return (np.array([self.thetamin] * self.k + [self.pmin] * self.k, dtype=float),
                np.array([self.thetamax] * self.k + [self.pmax] * self.k, dtype=float))

    def _candidate_nuggets(self, candidates):
        return np.full(len(candidates), EPS)

    def fittingObjective(self, candidates, args=None):
        """NegLnLike for a batch of hyper-parameter vectors (m, 2k); PENALTY where Psi is not positive definite."""
        c = np.atleast_2d(np.asarray(candidates, dtype=float))
        return self._batch_nll(c[:, :self.k], c[:, self.k:2 * self.k], self._candidate_nuggets(c))

    def fittingObjective_local(self, entry):
        """NegLnLike of a single hyper-parameter vector (for the local optimiser)."""
        return float(self.fittingObjective(entry)[0])

    def train(self, optimizer='pso'):
        """Maximum-likelihood hyper-parameters theta and p (see _search: 'pso' or 'ga', then particle swarms over
        log10(theta) and over the isotropic model, each polished by SLSQP)."""
        if optimizer not in self._search_settings:
            raise ValueError(f"optimizer must be 'pso' or 'ga', not {optimizer!r}")
        self.updateData()
        lo, hi = self._bounds()
        if np.ptp(self.y) == 0:  # constant data: every likelihood is -inf; any PD fit predicts the constant
            return self.update(hi)
        best, f_best = _search(self.fittingObjective, lo, hi, self.k, self.rng, optimizer, self._search_settings[optimizer])
        if f_best >= PENALTY:
            raise RuntimeError('no hyper-parameters gave a positive-definite correlation matrix')
        self.update(best)

    # ------------------------------------------------------------------ plots and diagnostics
    def plot(self, labels=False, show=True):
        """Plot 2-D models (matplotlib: error, prediction and surface) or 3-D models (mayavi iso-surfaces)."""
        if self.k == 3:
            import mayavi.mlab as mlab
            pts = np.mgrid[0:1:25j, 0:1:25j, 0:1:25j].reshape(3, -1).T
            if self.testfunction:
                truth = mlab.contour3d(_evaluate(self.testfunction, self.inversenormX(pts)).reshape(25, 25, 25), contours=15,
                                       transparent=True, figure=mlab.figure(figure='test'))
                truth.compute_normals = False
            pred = mlab.contour3d(self.predict_normalized(pts).reshape(25, 25, 25), contours=15, transparent=True,
                                  figure=mlab.figure(figure='predict'))
            pred.compute_normals = False
            if show:
                mlab.show()
        if self.k == 2:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8, 6))
            X, Y = np.meshgrid(*(np.linspace(lo, hi, 61) for lo, hi in self.normRange))
            pts = np.column_stack([X.ravel(), Y.ravel()])
            Z, Ze = self.predict(pts).reshape(X.shape), self.predict_var(pts).reshape(X.shape)
            spx, spy = self.inversenormX(self.X).T
            levels = 25
            fig.add_subplot(222)
            plt.contourf(X, Y, Ze, levels)
            plt.colorbar()
            plt.plot(spx, spy, 'ow')
            fig.add_subplot(221)
            if self.testfunction:
                ZT = _evaluate(self.testfunction, pts).reshape(X.shape)
                cs = plt.contour(X, Y, ZT, levels, colors='k', zorder=2).levels
                delta = abs(cs[0] - cs[1])
                levels = np.concatenate([[cs[0] - delta], cs, [cs[-1] + delta]])
            plt.contourf(X, Y, Z, levels, zorder=1)
            plt.plot(spx, spy, 'ow', zorder=3)
            plt.colorbar()
            ax = fig.add_subplot(212, projection='3d')
            ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.4)
            if self.testfunction:
                ax.plot_wireframe(X, Y, ZT, rstride=3, cstride=3)
            if show:
                plt.show()

    def saveFigure(self, name=None):
        """k=3: save prediction/error/truth iso-surfaces as VRML files '<name>_*.wrl' (mayavi).
        k=2: return (sample x, sample y, grid X, grid Y, prediction Z, error Ze): coordinates and Z in real-world
        units, Ze in the normalised units of predict_var()."""
        if self.k == 3:
            import mayavi.mlab as mlab
            mlab.options.offscreen = True
            pts = np.mgrid[0:1:75j, 0:1:75j, 0:1:75j].reshape(3, -1).T
            f, s = self._predict(pts)
            figures = {'prediction': f, 'error': s}
            if self.testfunction:
                figures['actual'] = _evaluate(self.testfunction, self.inversenormX(pts))
            for label, values in figures.items():
                fig = mlab.figure(figure=label)
                mlab.clf(figure=label)
                mlab.contour3d(values.reshape(75, 75, 75), contours=15, transparent=True, figure=fig).compute_normals = False
                mlab.savefig('%s_%s.wrl' % (name, label), figure=fig)
            mlab.close(all=True)
        if self.k == 2:
            g = np.linspace(0, 1, 61)
            pts = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
            f, s = self._predict(pts)
            X, Y = self.inversenormX(pts).T.reshape(2, 61, 61)
            spx, spy = self.inversenormX(self.X).T
            return spx, spy, X, Y, self.inversenormy(f).reshape(61, 61), s.reshape(61, 61)

    def calcuatemeanMSE(self, p2s=200, points=None):
        """Mean and standard deviation of predict_var() over real-world `points` (default: p2s random points
        spread over the data's bounding box)."""
        values = self.predicterr_normalized(self._unit_points(p2s)) if points is None else self.predict_var(points)
        return np.mean(values), np.std(values)

    def snapshot(self):
        """Record the current model state in self.history (for convergence tracking during infill)."""
        self.history['points'].append(self.n)
        self.history['neglnlike'].append(self.NegLnLike)
        self.history['theta'].append(self.theta.copy())
        self.history['p'].append(self.pl.copy())
        self.history['avgMSE'].append(self.calcuatemeanMSE(points=self.testPoints)[0])
        current = []
        if self.history['pointData'] is not None:
            pts = np.array([d['point'] for d in self.history['pointData']])
            current = list(self.predict(pts))
            for d, f, s in zip(self.history['pointData'], current, self.predict_var(pts)):
                d['predicted'].append(f)
                d['mse'].append(s)
                if len(d['predicted']) > 1:
                    d['gradient'] = np.gradient(d['predicted'])
        if self.history['lastPredictedPoints']:
            last = self.history['lastPredictedPoints']
            self.history['chisquared'].append(self.chisquared(last, current))
            self.history['rsquared'].append(self.rsquared(last, current))
            self.history['adjrsquared'].append(self.adjrsquares(self.history['rsquared'][-1], len(current)))
        self.history['lastPredictedPoints'] = current

    def rsquared(self, actual, observed):
        return np.corrcoef(observed, actual)[0, 1] ** 2

    def adjrsquares(self, rsquared, obs):
        return 1 - (1 - rsquared) * ((obs - 1) / (obs - self.k))

    def chisquared(self, actual, observed):
        actual, observed = np.asarray(actual), np.asarray(observed)
        return np.sum(np.abs((observed - actual) ** 2 / actual))
