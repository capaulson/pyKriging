"""
Sobol' sensitivity indices of kriging surrogates against analytic values: the Ishigami function
f = sin x1 + a sin^2 x2 + b x3^4 sin x1 on [-pi, pi]^3 with a = 7, b = 0.1 (Ishigami & Homma 1990), and the
additive f = x1 + 3 x2 on [0, 1]^2, whose variance (1 + 9) / 12 gives S = ST = [0.1, 0.9].
"""
from types import SimpleNamespace

import numpy as np
import pytest
from pyKriging import samplingplan
from pyKriging.GlobalSensitivity import SobolSensitivity

BOX = np.pi * np.array([[-1.0] * 3, [1.0] * 3])  # two corners spanning [-pi, pi]^3


def ishigami(X, a=7.0, b=0.1):
    return np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])


def ishigami_indices(a=7.0, b=0.1):
    """(S, ST) from the partial variances V_1 = (1 + b pi^4 / 5)^2 / 2, V_2 = a^2 / 8 and
    V_13 = b^2 pi^8 (1/18 - 1/50): S = [0.3139, 0.4424, 0], ST = [0.5576, 0.4424, 0.2437]."""
    V1, V2, V13 = (1 + b * np.pi ** 4 / 5) ** 2 / 2, a ** 2 / 8, b ** 2 * np.pi ** 8 * (1 / 18 - 1 / 50)
    return np.array([V1, V2, 0.0]) / (V1 + V2 + V13), np.array([V1 + V13, V2, V13]) / (V1 + V2 + V13)


def linear(X):
    return X[:, 0] + 3 * X[:, 1]


def analyse(X, y):
    sa = SobolSensitivity(X, y, seed=0)
    sa.sensitivity_Sobol()
    return sa


def assert_variance_fractions(sa, tol=0.02):
    """0 <= S_i <= ST_i <= 1 and sum S_i <= 1, up to the Monte Carlo error tol."""
    assert np.all(sa.S > -tol) and np.all(sa.S < sa.ST + tol) and np.all(sa.ST < 1 + tol) and sa.S.sum() < 1 + tol


@pytest.fixture(scope='module')
def plan2d():
    return samplingplan(2, random_state=0).optimallhc(20)


def test_default_sample_gives_1e_2_accuracy():
    """With the exact function as the model only the sampling error of the default sample size remains."""
    model = SimpleNamespace(predict=ishigami)
    sa = SobolSensitivity(BOX, ishigami(BOX), model=model, seed=0)
    S = sa.sensitivity_Sobol('Ishigami')
    assert sa.model is model and isinstance(S, list) and S == sa.Sobol == sa.S.tolist()
    np.testing.assert_allclose(sa.S, ishigami_indices()[0], atol=0.01)
    np.testing.assert_allclose(sa.ST, ishigami_indices()[1], atol=0.01)


def test_ishigami_surrogate():
    """The plan is stretched to fill [-pi, pi]^3, the box analysed: its points are cell centres, whose bounding
    box alone is 1/150 narrower and moves the exact ST_1 from 0.5576 to 0.5503."""
    U = samplingplan(3, random_state=0).optimallhc(150)
    X = np.pi * (2 * (U - U.min(axis=0)) / np.ptp(U, axis=0) - 1)
    sa = analyse(X, ishigami(X))
    S, ST = ishigami_indices()
    np.testing.assert_allclose(sa.S, S, atol=0.05)
    np.testing.assert_allclose(sa.ST, ST, atol=0.05)
    assert_variance_fractions(sa)


def test_additive_surrogate(plan2d):
    sa = analyse(plan2d, linear(plan2d))
    np.testing.assert_allclose(sa.S, [0.1, 0.9], atol=1e-3)
    np.testing.assert_allclose(sa.ST, [0.1, 0.9], atol=1e-3)
    assert_variance_fractions(sa)


def test_one_variable_explains_all_the_variance():
    x = np.linspace(0, 1, 8)  # 1-D X: a one-variable problem
    sa = analyse(x, np.sin(3 * x))
    np.testing.assert_allclose(np.r_[sa.S, sa.ST], 1.0, atol=1e-3)


def test_row_order_does_not_matter(plan2d):
    """The gradient-based numbers of earlier releases changed with the row order: [-1.93, 2.93] to [0.24, 0.76] here."""
    y = linear(plan2d)
    orders = [np.arange(20), np.arange(20)[::-1]] + [np.random.default_rng(s).permutation(20) for s in (1, 2)]
    results = [np.r_[sa.S, sa.ST] for sa in (analyse(plan2d[o], y[o]) for o in orders)]
    np.testing.assert_allclose(results, np.tile(results[0], (4, 1)), atol=1e-3)


@pytest.mark.parametrize('active', [0, 1])
def test_an_inert_variable_has_no_effect(plan2d, active):
    """f depends on one input only: S = ST = 1 for it and 0 for the other (earlier releases gave [1.06, -0.06])."""
    sa = analyse(plan2d, np.sin(3 * plan2d[:, active]))
    np.testing.assert_allclose(sa.S[[active, 1 - active]], [1.0, 0.0], atol=1e-2)
    np.testing.assert_allclose(sa.ST[[active, 1 - active]], [1.0, 0.0], atol=1e-2)
    assert_variance_fractions(sa)


def test_sensitivity_gamma_is_a_deprecated_no_op(plan2d):
    sa = SobolSensitivity(plan2d, linear(plan2d), model=SimpleNamespace(predict=linear))
    with pytest.warns(DeprecationWarning):
        assert sa.sensitivity_gamma(1.0) == []


def test_pie_chart_shows_first_order_and_interaction_shares(monkeypatch):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
    monkeypatch.setattr(plt, 'show', lambda: None)
    sa = SobolSensitivity(BOX, ishigami(BOX), model=SimpleNamespace(predict=ishigami), seed=0)
    S = np.clip(sa.sensitivity_Sobol('Ishigami', plot=1), 0, None)  # S_3 = 0 can come out slightly negative
    ax = plt.gca()
    shares = [(w.theta2 - w.theta1) / 360 for w in ax.patches if isinstance(w, Wedge)]
    np.testing.assert_allclose(shares, np.append(S, 1 - S.sum()), atol=1e-9)
    assert ax.get_title() == 'Ishigami'
    plt.close('all')
