"""
Variance-based global sensitivity indices (Sobol' 1993, 2001) computed on a Kriging surrogate.

For Y = f(X) with independent inputs, uniform on the bounding box of the samples, the first-order index
S_i = V[E(Y | X_i)] / V(Y) is the fraction of the output variance due to X_i alone and the total-order index
ST_i = E[V(Y | X_~i)] / V(Y) the fraction due to X_i including all its interactions: 0 <= S_i <= ST_i <= 1 and
sum S_i <= 1 <= sum ST_i, with S_i = ST_i and sum S_i = 1 for an additive f. Both are estimated from two base
samples A, B (N, k) and the k radial samples AB_i (A with column i taken from B), N (k + 2) predictions in all:

    S_i  = mean(f(B) (f(AB_i) - f(A))) / V      (Saltelli et al. 2010, Comput. Phys. Commun. 181:259, table 2b)
    ST_i = mean((f(A) - f(AB_i))^2) / (2 V)     (Jansen 1999; Saltelli et al. 2010, table 2f)

where V is the variance of f over A and B. f(B) is first centred on the mean of f over A and B: the expectation
of the S_i estimator does not depend on the mean of f, but its variance grows with it. [A | B] is a scrambled
2k-dimensional Sobol' sequence (scipy.stats.qmc): every point is uniform, so the estimates stay unbiased, and
for a smooth surrogate they converge far faster than with plain Monte Carlo samples.
"""
import warnings

import numpy as np

from .krige import kriging


class SobolSensitivity():
    def __init__(self, X, y, model=None, n_samples=2 ** 13, seed=None):
        """
        X: samples (n, k), whose bounding box, sampled uniformly, is the input distribution; y: responses (n,).
        model: fitted surrogate whose predict() maps points (m, k) to (m,) responses (None: fit and train
        kriging(X, y) here). n_samples: base sample size N, a power of 2; an analysis costs N (k + 2) predictions
        and the default gives errors of about 1e-2 or less. seed: seed for the kriging fit and the scrambling.
        """
        X = np.array(X, dtype=float)
        self.X, self.y = X.reshape(len(X), -1), np.array(y, dtype=float).ravel()
        self.n, self.k = self.X.shape
        if model is None:
            model = kriging(self.X, self.y, seed=seed)
            model.train()
        self.model, self.n_samples, self.rng = model, n_samples, np.random.default_rng(seed)
        self.Sobol, self.S, self.ST = [], None, None

    def sensitivity_gamma(self, V):
        """Deprecated no-op returning []: the 'gamma' main effects of earlier releases were not Sobol'
        quantities, and sensitivity_Sobol() needs no preparation."""
        warnings.warn('sensitivity_gamma() is deprecated and does nothing: call sensitivity_Sobol() directly',
                      DeprecationWarning, stacklevel=2)
        return []

    def sensitivity_Sobol(self, Model='', plot=0):
        """
        First-order indices S_i, returned as a list and stored as self.Sobol and self.S, and total-order indices
        ST_i (self.ST), from a freshly scrambled sample. plot=1 draws the first-order indices and the interactions
        share 1 - sum S_i as a pie chart titled Model.
        """
        from scipy.stats import qmc  # deferred: scipy.stats costs ~0.3 s to import
        try:
            sobol = qmc.Sobol(2 * self.k, rng=self.rng)
        except TypeError:  # scipy < 1.15 calls the argument seed
            sobol = qmc.Sobol(2 * self.k, seed=self.rng)
        U = sobol.random(self.n_samples)
        lo, span = self.X.min(axis=0), np.ptp(self.X, axis=0)
        A, B = lo + span * U[:, :self.k], lo + span * U[:, self.k:]
        fA, fB = self.model.predict(A), self.model.predict(B)
        f = np.concatenate([fA, fB])
        V, fB = f.var(), fB - f.mean()
        self.S, self.ST = np.empty(self.k), np.empty(self.k)
        for i in range(self.k):
            AB = A.copy()
            AB[:, i] = B[:, i]
            d = self.model.predict(AB) - fA
            self.S[i], self.ST[i] = np.mean(fB * d) / V, np.mean(d * d) / (2 * V)
        self.Sobol = self.S.tolist()
        if plot:
            import matplotlib.pyplot as plt
            S = np.clip(self.S, 0.0, None)
            colors = ['lightskyblue', 'gold', 'yellowgreen', 'lightcoral']
            plt.figure(figsize=(12, 8), facecolor='w', edgecolor='k', linewidth=2.0, frameon=True)
            plt.pie(np.append(S, max(0.0, 1.0 - S.sum())),
                    labels=['S_%d' % i for i in range(1, self.k + 1)] + ['interactions'],
                    colors=[colors[i % 4] for i in range(self.k)] + ['lightgrey'],
                    explode=[0.01 + 0.02 * (i % 3) for i in range(self.k + 1)], autopct='%1.1f%%', shadow=True,
                    startangle=90)
            plt.axis('equal')
            plt.title(Model)
            plt.show()
        return self.Sobol
