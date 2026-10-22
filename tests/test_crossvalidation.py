"""
Cross-validation tests. The closed-form leave-one-out (LOO) results must equal explicit refits of the model to the
other n - 1 samples with the same hyper-parameters (Dubrule 1983); the other diagnostics are checked against them.
"""
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

import pyKriging
from pyKriging import kriging, regression_kriging, samplingplan
from pyKriging.CrossValidation import Cross_Validation

KRIGING = [2.0, 4.0, 1.7, 2.0]  # theta_1, theta_2, p_1, p_2 giving a well-conditioned Psi
REGRESSION = [4.0, 2.0, 2.0, 1.9, 1e-2]  # ... and Lambda


class nugget_kriging(kriging):
    """Kriging with a nugget that, unlike regression kriging's Lambda, is left out of the error estimate."""

    def _nugget(self):
        return 1e-3


def smooth(X):
    """Positive, so that it can be log- or inverse-transformed."""
    return 40 + 25 * np.sin(1.5 * X[:, 0]) * np.cos(X[:, 1])


class Simulator:
    """Test function bound to a resource (a lock) that cannot be deep-copied."""

    def __init__(self):
        self.lock = threading.Lock()

    def evaluate(self, X):
        with self.lock:
            return smooth(np.atleast_2d(X))


def fitted(cls=kriging, values=KRIGING, n=15):
    """Model of smooth() on [0, 2]^2 fitted with fixed hyper-parameters (no training)."""
    X = 2 * samplingplan(2, random_state=1).rlh(n)
    model = cls(X, smooth(X), seed=0)
    model.update(values)
    return model


def explicit_loo(model, values):
    """Normalised prediction and standard error at each sample of the model refitted without it."""
    f, s = np.empty(model.n), np.empty(model.n)
    for i in range(model.n):
        keep = np.arange(model.n) != i
        loo = type(model)(model.X[keep], model.y[keep])
        loo.X, loo.y = model.X[keep], model.y[keep]  # the full model's normalisation, as the closed form
        loo.updateData()
        loo.update(values)
        f[i], s[i] = loo.predict_normalized(model.X[i]), loo.predicterr_normalized(model.X[i])
    return f, s


@pytest.mark.parametrize('cls, values', [(kriging, KRIGING), (regression_kriging, REGRESSION),
                                         (nugget_kriging, KRIGING)])
def test_closed_form_loo_equals_explicit_refits(cls, values):
    model = fitted(cls, values)
    f, s = explicit_loo(model, values)
    predict_list, predict_varr, scvr = Cross_Validation(model).calculate_SCVR()
    np.testing.assert_allclose(predict_list, model.inversenormy(f), rtol=1e-12)
    np.testing.assert_allclose(predict_varr, s, rtol=1e-10)
    np.testing.assert_allclose(scvr, (model.y - f) / s, rtol=1e-10, atol=1e-12)


def test_untrained_model_is_trained_and_its_scvr_is_standardised():
    X = 2 * samplingplan(2, random_state=2).rlh(20)
    model = kriging(X, smooth(X), seed=0)
    predict_list, predict_varr, scvr = Cross_Validation(model).calculate_SCVR(optimiser='ga')
    assert model.mu is not None
    assert np.all(np.isfinite(scvr)) and np.all(np.array(predict_varr) > 0)
    assert np.mean(np.abs(scvr) <= 3) >= 0.9
    f, s = explicit_loo(model, np.r_[model.theta, model.pl])  # trained: Psi is ill-conditioned
    np.testing.assert_allclose(scvr, (model.y - f) / s, rtol=1e-6, atol=1e-6)


def test_rmse_and_rsquared_are_those_of_the_loo_predictions():
    model = fitted()
    cv = Cross_Validation(model)
    y, f = model.inversenormy(model.y), np.array(cv.calculate_SCVR()[0])
    rmse, rsquared = (float(s.split('=')[1]) for s in cv.calculate_RMSE_Rsquared('pso', model.n))
    assert rmse == pytest.approx(np.sqrt(np.mean((y - f) ** 2)), abs=1e-6)
    assert rsquared == pytest.approx(np.corrcoef(y, f)[0, 1] ** 2, abs=1e-6)
    assert rmse > 0.01 * np.ptp(y)  # out-of-sample: in-sample predictions interpolate (RMSE ~ 0, R^2 ~ 1)
    rmse = float(cv.calculate_RMSE_Rsquared('pso', 7)[0].split('=')[1])
    assert np.min(np.abs(y - f)) - 1e-6 <= rmse <= np.max(np.abs(y - f)) + 1e-6


@pytest.mark.parametrize('cls, values', [(kriging, KRIGING), (regression_kriging, REGRESSION)])
def test_leave_n_out_honours_q(monkeypatch, cls, values):
    folds = []

    def train(self, optimizer='pso'):  # fixed hyper-parameters instead of a search; records the fold model
        folds.append((type(self), self.n))
        self.update(values)

    monkeypatch.setattr(kriging, 'train', train)
    model = fitted(cls, values)
    cv = Cross_Validation(model)
    cv.leave_n_out(q=4)
    assert {c for c, _ in folds} == {cls} and sorted(n for _, n in folds) == [11, 11, 11, 12]
    folds.clear()
    mean, std = cv.leave_n_out(q=model.n)  # leave-one-out with fixed hyper-parameters is the closed form
    assert folds == [(cls, model.n - 1)] * model.n
    e2 = (model.y - model.normy(np.array(cv.calculate_SCVR()[0]))) ** 2
    np.testing.assert_allclose([mean, std], [e2.mean(), e2.std()], rtol=1e-9)


def test_transformed_scvr_is_that_of_a_model_of_the_transformed_responses(monkeypatch):
    monkeypatch.setattr(kriging, 'train', lambda self, optimizer='pso': self.update(KRIGING))
    model = fitted()
    direct = kriging(model.inversenormX(model.X), np.log(model.inversenormy(model.y)))
    direct.update(KRIGING)
    np.testing.assert_allclose(Cross_Validation(model).calculate_transformed_SCVR('logarithmic'),
                               Cross_Validation(direct).calculate_SCVR(), rtol=1e-9)


@pytest.mark.parametrize('transformation, func', [('logarithmic', np.log), ('inverse', lambda y: -1 / y)])
def test_transformed_scvr_trains_a_copy(transformation, func):
    model = fitted()
    model.testfunction = Simulator().evaluate  # shared by the copy, not deep-copied
    state = [model.y.copy(), model.theta.copy(), model.pl.copy(), model.mu]
    predict_list, predict_varr, scvr = Cross_Validation(model).calculate_transformed_SCVR(transformation, 'ga')
    assert len(scvr) == model.n and np.all(np.isfinite([predict_list, predict_varr, scvr]))
    assert np.corrcoef(predict_list, func(model.inversenormy(model.y)))[0, 1] > 0.95
    for before, after in zip(state, [model.y, model.theta, model.pl, model.mu]):
        np.testing.assert_array_equal(before, after)


def test_plot0_returns_lists_that_repeated_calls_replace():
    model = fitted()
    cv = Cross_Validation(model)
    out = cv.calculate_SCVR(plot=0)
    assert type(out) is tuple and len(out) == 3
    assert all(type(v) is list and len(v) == model.n and all(type(x) is float for x in v) for v in out)
    again = cv.calculate_SCVR()
    assert again == out == (cv.predict_list, cv.predict_varr, cv.scvr) and all(len(v) == model.n for v in again)


def test_invalid_arguments():
    model = fitted()
    cv = Cross_Validation(model)
    for call in (lambda: cv.calculate_SCVR(plot=2), lambda: cv.calculate_transformed_SCVR('square'),
                 lambda: cv.calculate_RMSE_Rsquared('pso', model.n + 1), lambda: cv.leave_n_out(1),
                 lambda: cv.leave_n_out(model.n + 1), lambda: Cross_Validation(fitted(n=2)).calculate_SCVR(),
                 lambda: Cross_Validation(kriging(model.X, model.y - 0.5)).calculate_transformed_SCVR('logarithmic')):
        with pytest.raises(ValueError):
            call()


def test_plots(monkeypatch):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    cv = Cross_Validation(fitted(), name='smooth')
    try:
        assert cv.calculate_SCVR(plot=1) == cv.calculate_SCVR()
        ax1, ax2 = plt.gcf().axes
        assert (ax1.get_ylabel(), ax2.get_xlabel(), ax2.get_ylabel()) == ('SCVR', 'predicted y', 'y')
        cv.calculate_transformed_SCVR('inverse', 'ga', plot=1)
        assert (plt.gcf().axes[1].get_xlabel(), plt.gcf().axes[1].get_ylabel()) == ('predicted -1/y', '-1/y')
        cv.QQ_plot()
        assert plt.gca().get_ylabel() == 'ordered SCVR'
        fresh = Cross_Validation(fitted())
        fresh.QQ_plot()  # no SCVR yet: computes them first
        assert fresh.scvr == cv.calculate_SCVR()[2]
    finally:
        plt.close('all')


def test_matplotlib_is_imported_lazily():
    root = os.path.dirname(os.path.dirname(os.path.abspath(pyKriging.__file__)))  # the package under test
    code = f'import sys; sys.path.insert(0, {root!r}); import pyKriging.CrossValidation; ' \
           'sys.exit("matplotlib" in sys.modules)'
    # -B: no .pyc files in the package (-I ignores PYTHONDONTWRITEBYTECODE)
    assert subprocess.run([sys.executable, '-I', '-B', '-c', code]).returncode == 0
