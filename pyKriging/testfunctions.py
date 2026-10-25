"""
Analytic benchmark functions for surrogate-model testing. Every method accepts one point (k,) or many
points (m, k) and returns an (m,) array, except rastrigin/rosenbrock which take one point and return [f].
"""
import numpy as np


def _points(X, k=None):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if k is not None and X.shape[1] != k:
        raise ValueError(f'expected {k}-dimensional points, got {X.shape[1]}')
    return X


def _styblinski_tang(X):
    return np.sum(X ** 4 - 16 * X ** 2 + 5 * X, axis=1) / 2.0


class testfunctions():
    def linear(self, X):
        """f(x) = sum(x)."""
        return _points(X).sum(axis=1)

    def squared(self, X, offset=.25):
        """Euclidean distance from (offset, ..., offset)."""
        return np.sum((_points(X) - offset) ** 2, axis=1) ** 0.5

    def cubed(self, X, offset=.25):
        """(sum((x - offset)^3))^(1/3); NaN where the sum is negative."""
        return np.sum((_points(X) - offset) ** 3, axis=1) ** (1 / 3.0)

    def branin(self, X):
        """Branin function on [0, 1]^2 (scaled to [-5, 10] x [0, 15]) plus 5 x_1, which leaves one global minimum."""
        X = _points(X, 2)
        x1, x2 = 15 * X[:, 0] - 5, 15 * X[:, 1]
        return ((x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2
                + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10) + 5 * X[:, 0]

    def branin_noise(self, X):
        """branin() plus N(0, 15^2) noise, for testing regression kriging."""
        f = self.branin(X)
        return f + np.random.standard_normal(f.shape) * 15

    def paulson(self, X, hz=5):
        """0.5 sin(hz x_1) + 0.5 cos(hz x_2) on [0, 1]^2."""
        X = _points(X, 2)
        return .5 * np.sin(X[:, 0] * hz) + .5 * np.cos(X[:, 1] * hz)

    def paulson1(self, X, hz=10):
        """sin(hz x_1)/(x_1 + 0.2) + cos(hz x_2)/(x_2 + 0.2) on [0, 1]^2."""
        X = _points(X, 2)
        return np.sin(X[:, 0] * hz) / (X[:, 0] + .2) + np.cos(X[:, 1] * hz) / (X[:, 1] + .2)

    def runge(self, X, offset=0.0):
        """1 / (1 + |x - offset|^2)."""
        return 1 / (1 + np.sum((_points(X) - offset) ** 2, axis=1))

    def stybtang(self, X):
        """Styblinski-Tang function on [-5, 5]^d; minimum -39.166 d at x_i = -2.9035."""
        return _styblinski_tang(_points(X))

    def stybtang_norm(self, X):
        """Styblinski-Tang with inputs in [0, 1]^d."""
        return _styblinski_tang(_points(X) * 10 - 5)

    def curretal88exp(self, X):
        """Currin et al. (1988) exponential function on [0, 1]^2."""
        X = _points(X)
        x1, x2 = X[:, 0], X[:, 1]
        return (1 - np.exp(-1 / (2 * x2))) * (2300 * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60) / (100 * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20)

    def cosine(self, X):
        """cos(sum(x))."""
        return np.cos(_points(X).sum(axis=1))

    def rastrigin(self, x):
        """2-D Rastrigin function at one point; global minimum 0 at (0, 0). Returns [f]."""
        return [20 + x[0] ** 2 + x[1] ** 2 - 10 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))]

    def rosenbrock(self, x):
        """Rosenbrock function at one point of any dimension; global minimum 0 at (1, ..., 1). Returns [f]."""
        x = np.asarray(x, dtype=float)
        return [float(np.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2))]
