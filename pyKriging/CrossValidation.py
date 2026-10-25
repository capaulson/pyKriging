"""
Cross-validation of Kriging models (Jones, Schonlau & Welch 1998, J. Global Optim. 13:455-492).

Leave-one-out (LOO) predictions with the hyper-parameters held fixed need no refitting (Dubrule 1983, Math. Geol.
15:687-699). With the correlation matrix R = Psi (nugget on its diagonal), Q = R^-1, q = Q 1, c = 1^T Q 1 and the
predictor weights w = Q (y - 1 mu) = (Q - q q^T / c) y, the fit to the n - 1 samples other than i (mu and sigma^2
re-estimated) predicts y_i with residual, process variance and standard error

    e_i = y_i - y_-i = w_i / (Q_ii - q_i^2 / c),        sigma^2_-i = (n sigma^2 - w_i e_i) / (n - 1),
    s_-i = sqrt(sigma^2_-i (1 / Q_ii - nugget + offset)),

where 1 / Q_ii = R_ii - r^T R_-i^-1 r (r: correlations of sample i with the others; R_-i: R without row and column
i), so s_-i is what predict_var gives for the n - 1 point fit (offset = Lambda for regression kriging, else 0). The
standardised cross-validated residuals SCVR_i = e_i / s_-i of a valid model lie roughly within [-3, 3].

@author: Giorgos
"""
import copy

import numpy as np
from scipy.linalg import lapack

from .utilities import mse, splitArrays

_TRANSFORMS = {'logarithmic': (np.log, 'ln(y)'), 'inverse': (lambda y: -1.0 / y, '-1/y')}


def _loo(model):
    """Closed-form LOO residuals e and standard errors s (module docstring) of a fitted model, in normalised units
    of y. Costs one triangular solve with n right-hand sides (Q = U^-1 U^-T). Needs n >= 3, as a fit to one sample
    has sigma^2 = 0."""
    if model.n < 3:
        raise ValueError('leave-one-out cross-validation needs at least 3 samples')
    # U^-T by a triangular solve: dtrtri crashes in the OpenBLAS of the scipy 1.7/1.8 wheels for macOS arm64
    Vt = lapack.dtrtrs(model.U, np.eye(model.n), lower=0, trans=1)[0]
    a = Vt.sum(axis=1)  # U^-T 1
    q = lapack.dtrtrs(model.U, a, lower=0)[0]  # Q 1
    Qii, w = np.einsum('ij,ij->j', Vt, Vt), model._weights  # diag(Q) = column norms of U^-T
    e = w / (Qii - q * q / (a @ a))
    sigma2 = (model.n * model.SigmaSqr - w * e) / (model.n - 1)
    return e, np.sqrt(np.abs(sigma2 * (1.0 / Qii - model._nugget() + model._err_offset())))


class Cross_Validation():

    def __init__(self, model, name=None):
        """model: a kriging or regression_kriging model, which calculate_SCVR, calculate_RMSE_Rsquared and QQ_plot
        train first if it is not fitted (the other methods train copies); name: title for the plots."""
        self.model, self.name = model, name
        self.X, self.y = model.X, model.y  # normalised data, as in earlier releases
        self.n, self.k = model.X.shape
        self.predict_list, self.predict_varr, self.scvr = [], [], []

    def _fitted(self, optimiser):
        """The model, trained with optimiser ('pso' or 'ga') first if it has not been fitted."""
        if self.model.mu is None:
            self.model.train(optimiser)
        return self.model

    def _copy(self):
        """Independent copy of the model; the user's test function is shared, as it may hold resources (e.g. locks)
        that cannot be copied."""
        return copy.deepcopy(self.model, {id(self.model.testfunction): self.model.testfunction})

    def calculate_RMSE_Rsquared(self, optimiser, nt):
        """
        Root mean squared error RMSE = sqrt(sum (y_i - y_-i)^2 / nt), in the units of y, and squared correlation
        coefficient R^2 of y_i and the leave-one-out predictions y_-i (see calculate_SCVR) at nt samples chosen at
        random (np.random). optimiser ('pso' or 'ga') trains the model first if it has not been fitted.
        Returns ['RMSE = ...', 'Rsquared = ...'].
        """
        m = self._fitted(optimiser)
        i = np.random.choice(m.n, nt, replace=False)
        y, f = m.inversenormy(m.y[i]), m.inversenormy(m.y[i] - _loo(m)[0][i])
        return ['RMSE = %f' % np.sqrt(np.sum((y - f) ** 2) / nt), 'Rsquared = %f' % (np.corrcoef(y, f)[0, 1] ** 2)]

    def calculate_SCVR(self, optimiser='pso', plot=0):
        """
        Leave-one-out predictions y_-i of every sample and their standardised cross-validated residuals
        SCVR_i = (y_i - y_-i) / s_-i, from the closed form in the module docstring. The hyper-parameters stay at the
        model's values, so optimiser ('pso' or 'ga') is only used to train a model that has not been fitted.
        plot=1 also plots the SCVR of each sample, and y_i against y_-i.
        Returns lists (predict_list, predict_varr, scvr): y_-i in the units of y, s_-i in normalised units of y
        (as predict_var) and the SCVR; they are kept as attributes for QQ_plot().
        """
        if plot not in (0, 1):
            raise ValueError('value for plot should be either 0 or 1')
        m = self._fitted(optimiser)
        return self._scvr(m, m.inversenormy(m.y), 'y', plot)

    def calculate_transformed_SCVR(self, transformation, optimiser='pso', plot=0):
        """
        SCVR (see calculate_SCVR) of a copy of the model fitted to transformed responses, 'logarithmic' (ln y) or
        'inverse' (-1/y), with hyper-parameters trained by optimiser ('pso' or 'ga'); a transformation that brings
        the SCVR within [-3, 3] can make a better model (Jones et al. 1998). predict_list holds the leave-one-out
        predictions of the transformed responses. The model itself is not changed.
        """
        if transformation not in _TRANSFORMS:
            raise ValueError("transformation should be 'logarithmic' or 'inverse'")
        if plot not in (0, 1):
            raise ValueError('value for plot should be either 0 or 1')
        func, label = _TRANSFORMS[transformation]
        with np.errstate(divide='ignore', invalid='ignore'):
            y = func(self.model.inversenormy(self.model.y))
        if not np.all(np.isfinite(y)):
            raise ValueError(f'the {transformation} transformation is undefined for some responses')
        m = self._copy()
        m.ynormRange = [y.min(), y.max()]
        m.y = m.normy(y)
        m.train(optimiser)
        return self._scvr(m, y, label, plot)

    def _scvr(self, m, y, label, plot):
        """Closed-form leave-one-out results of fitted model m, kept as attributes; plot=1 plots them against the
        observations y (called label)."""
        e, s = _loo(m)
        self.predict_list = m.inversenormy(m.y - e).tolist()
        with np.errstate(divide='ignore', invalid='ignore'):  # a duplicated sample has e = s = 0: no SCVR (NaN)
            scvr = np.where(s > 1e-10 * np.sqrt(m.SigmaSqr), e / s, np.nan)
        self.predict_varr, self.scvr = s.tolist(), scvr.tolist()
        if plot:
            self._plot(y, label)
        return self.predict_list, self.predict_varr, self.scvr

    def _plot(self, y, label):
        """SCVR of each sample with the [-3, 3] band, and the observations y against the leave-one-out predictions."""
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.scatter(np.arange(1, len(self.scvr) + 1), self.scvr, alpha=0.5, edgecolor='black', facecolor='b')
        for bound in (-3, 3):
            ax1.axhline(bound, color='r')
        ax1.set(xlabel='DoE individual', ylabel='SCVR')
        ax2.scatter(self.predict_list, y, alpha=0.5, edgecolor='black', facecolor='b')
        ax2.axline((y[0], y[0]), slope=1, color='.3')
        ax2.set(xlabel='predicted ' + label, ylabel=label)
        if self.name:
            fig.suptitle(self.name)
        plt.show()

    def QQ_plot(self):
        """Normal probability (Q-Q) plot of the SCVR from the last calculate_SCVR or calculate_transformed_SCVR call
        (calculate_SCVR() if there was none): for a valid model the points lie close to a straight line."""
        import matplotlib.pyplot as plt
        from scipy import stats
        if not self.scvr:
            self.calculate_SCVR()
        plt.figure(figsize=(8, 6))
        stats.probplot(self.scvr, dist='norm', plot=plt)
        plt.xlabel('standard normal quantile')
        plt.ylabel('ordered SCVR')
        if self.name:
            plt.title(self.name)
        plt.show()

    def leave_n_out(self, q=5):
        """
        q-fold cross-validation: the samples are split at random (np.random) into q groups, and a copy of the model
        trained (train()) on the other groups predicts each group. Returns the mean and standard deviation of the
        squared prediction errors, in normalised units of y.
        """
        if not 2 <= q <= self.model.n:
            raise ValueError('q should be between 2 and the number of samples')
        errors = []
        for X, y, Xq, yq in splitArrays(self.model, q):
            m = self._copy()
            m.X, m.y = X, y
            m.updateData()
            m.train()
            errors.append(mse(yq, m.predict_normalized(Xq)))
        errors = np.concatenate(errors)
        return np.average(errors), np.std(errors)
