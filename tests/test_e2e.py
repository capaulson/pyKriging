"""
End-to-end workflows: sampling plan -> training -> batch predictions -> accuracy against the true function, error
estimates, infill (error, expected improvement, Kriging-believer multi-point infill), optimisation, plots and
convergence history. Accuracy thresholds sit between the values of correctly trained models and those of the same
models at the default (untrained) hyper-parameters, which each test checks fail them.
"""
import functools

import numpy as np
import pytest

import _reference as R
from pyKriging import kriging, samplingplan

T = R.lhs(1000, 2, 999)  # test points, independent of the library's random numbers


def sine2(X):
    X = np.atleast_2d(X)
    return np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])


def branin(X):
    X = np.atleast_2d(X)
    x1, x2 = 15 * X[:, 0] - 5, 15 * X[:, 1]
    return (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) \
        + 10


@functools.lru_cache(maxsize=None)
def trained(name):
    """(model, X, y) trained once per session; tests must not modify the model (see fresh())."""
    f, n = {'sine': (sine2, 20), 'branin': (branin, 30)}[name]
    X = samplingplan(2, random_state=1).optimallhc(n)
    model = kriging(X, f(X), seed=1)
    model.train()
    return model, X, f(X)


def fresh(name, **kwargs):
    """A new model of the same data at the trained hyper-parameters (no training)."""
    model, X, y = trained(name)
    copy = kriging(X, y, seed=0, **kwargs)
    copy.update(np.r_[model.theta, model.pl])
    return copy, X, y


def untrained(X, y):
    model = kriging(X, y)
    model.update(np.r_[np.ones(model.k), np.full(model.k, 2.0)])  # the defaults: theta = 1, p = 2
    return model


def rmse(model, f, X=T):
    return np.sqrt(np.mean((model.predict(X) - f(X)) ** 2))


# ------------------------------------------------------------------ accuracy
def test_readme_quick_start():
    from pyKriging import testfunctions
    X = samplingplan(k=2).optimallhc(20)
    y = testfunctions().branin(X)
    model = kriging(X, y, seed=0)
    model.train()
    assert isinstance(model.predict([0.5, 0.5]), float) and isinstance(model.predicterr([0.5, 0.5]), float)
    assert model.predict(np.random.rand(1000, 2)).shape == (1000,)
    assert abs(model.predict([0.5, 0.5]) - testfunctions().branin([0.5, 0.5])[0]) < 0.02 * np.ptp(y)
    new = model.infill(2, method='ei')
    assert new.shape == (2, 2)
    model.addPoint(new[0], testfunctions().branin(new[0])[0])
    model.train()
    assert model.n == 21


def test_sine_model_is_accurate():
    model, X, y = trained('sine')
    assert rmse(model, sine2) < 0.13 < rmse(untrained(X, y), sine2)


def test_branin_model_is_accurate():
    model, X, y = trained('branin')
    yt = branin(T)
    mae = np.mean(np.abs(model.predict(T) - yt)) / np.ptp(yt)
    assert mae < 0.005 < np.mean(np.abs(untrained(X, y).predict(T) - yt)) / np.ptp(yt)
    assert rmse(model, branin) < 0.1 * rmse(untrained(X, y), branin)


def test_5000_predictions_in_one_call():
    model, X, y = trained('sine')
    Xt = samplingplan(2, random_state=0).rlh(5000)
    f = model.predict(Xt)
    assert f.shape == (5000,)
    np.testing.assert_allclose(f[:50], [model.predict(x) for x in Xt[:50]], rtol=1e-12)
    assert np.mean(np.abs(f - sine2(Xt))) < 0.065 < np.mean(np.abs(untrained(X, y).predict(Xt) - sine2(Xt)))


def test_five_variables():
    """The legacy suite only checked that a 5-D prediction is finite. Its likelihood optimum is nearly isotropic, which
    0.2.0 missed (RMSE 0.65 against 0.05 at the optimum)."""
    def f(X):
        return np.sum(np.sin(np.pi * X), axis=1) + X[:, 0] * X[:, 1]

    X = R.MAXIMIN_LHS_40x5
    model = kriging(X, f(X), seed=0)
    model.train()
    Xt = R.lhs(500, 5, 4)
    assert rmse(model, f, Xt) < 0.5 * rmse(untrained(X, f(X)), f, Xt)


def test_error_estimates_are_calibrated():
    """|error| <= 3 s(x) almost everywhere (s in the units of y), and s is larger where the errors are."""
    model, X, y = trained('sine')
    errors, s = np.abs(model.predict(T[:400]) - sine2(T[:400])), model.predicterr(T[:400])
    assert np.mean(errors <= 3 * s) >= 0.9
    rank = np.argsort(np.argsort(s))
    assert np.corrcoef(rank, np.argsort(np.argsort(errors)))[0, 1] > 0.3


# ------------------------------------------------------------------ infill
def test_error_infill_finds_the_largest_error_and_restores_the_model():
    model, X, y = fresh('sine')
    state = [model.X.copy(), model.y.copy(), model.theta.copy(), model.pl.copy(), model.mu, model.SigmaSqr]
    new = model.infill(1, method='error', addPoint=False)
    assert new.shape == (1, 2) and np.all((new >= X.min(0)) & (new <= X.max(0)))
    grid = np.stack(np.meshgrid(*[np.linspace(lo, hi, 101) for lo, hi in model.normRange]), -1).reshape(-1, 2)
    assert model.predict_var(new[0]) >= 0.98 * model.predict_var(grid).max()
    for before, after in zip(state, [model.X, model.y, model.theta, model.pl, model.mu, model.SigmaSqr]):
        np.testing.assert_array_equal(before, after)


def test_ei_infill_finds_the_largest_expected_improvement():
    model, X, y = fresh('branin')
    new = model.infill(1, method='ei')
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201)), -1).reshape(-1, 2)
    assert model.infill_ei(new[0]) >= 0.98 * model.infill_ei(grid).max() > 0


def test_multi_point_infill_imputes_the_kriging_prediction():
    """Kriging believer: after each point the model is refitted with its own prediction there (mu, sigma^2 and the
    hyper-parameters fixed), so the next point maximises the EI of that model; the model is restored afterwards."""
    model, X, y = fresh('sine')
    state = [model.X.copy(), model.y.copy(), model.theta.copy(), model.pl.copy(), model.mu, model.SigmaSqr,
             model.NegLnLike]
    new = model.infill(3, method='ei')
    assert new.shape == (3, 2) and len({tuple(np.round(p, 6)) for p in new}) == 3
    assert np.min(np.abs(X[:, None, :] - new[None, :, :]).sum(axis=2)) > 1e-3  # no sample is proposed again
    for before, after in zip(state, [model.X, model.y, model.theta, model.pl, model.mu, model.SigmaSqr,
                                     model.NegLnLike]):
        np.testing.assert_array_equal(before, after)
    # the believer model after the first point, built independently
    Xn, x1 = model.X, model.normX(new[0])
    ref = R.RefKriging(np.vstack([Xn, x1]), np.r_[model.y, model.predict_normalized(x1)], model.theta, model.pl)
    ref.mu, ref.sigma2 = model.mu, model.SigmaSqr
    ref.w = np.linalg.solve(ref.Psi, ref.y - ref.mu)
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 151), np.linspace(0, 1, 151)), -1).reshape(-1, 2)
    believer_ei = ref.ei(grid)
    assert ref.ei(model.normX(new[1]))[0] >= 0.95 * believer_ei.max() > 0
    assert model.expimp(model.normX(new[1])) < model.expimp(x1)  # it is not the best point of the original model


# the samples of the optimisation in test_efficient_global_optimisation_of_the_papers_function after 12 updates:
# converged, with samples 1e-4 apart at the minimum
CONVERGED = np.array([0.0, 0.5, 1.0, 0.5056476382808011, 0.23777338634579007, 0.3184161223009613, 0.2240358361040277,
                      0.18630281995480763, 0.14424379058505263, 0.7264171203024836, 0.7684068714538498,
                      0.7580067876654707, 0.7571631305164176, 0.7572481052750895, 0.9341122313582141])


def test_multi_point_infill_restores_the_model_even_when_imputation_retrains():
    """Once an optimisation has converged the next points lie ~1e-6 from samples and imputing them makes Psi
    numerically singular; infill must still leave theta, p, mu and sigma^2 exactly as they were."""
    y = R.forrester_fe(CONVERGED)
    changed = []
    for seed in range(6):
        model = kriging(CONVERGED, y, seed=seed)
        model.train()
        before = np.r_[model.theta, model.pl, model.mu, model.SigmaSqr]
        model.infill(3, method='ei')
        if not np.array_equal(before, np.r_[model.theta, model.pl, model.mu, model.SigmaSqr]):
            changed.append((seed, before[0], model.theta[0]))
    assert not changed, f'(seed, theta before, theta after infill): {changed}'


def test_infill_is_reproducible_given_a_seed():
    a, b = fresh('sine')[0], fresh('sine')[0]
    np.testing.assert_array_equal(a.infill(2, method='ei'), b.infill(2, method='ei'))
    np.testing.assert_array_equal(a.infill(1), b.infill(1))


def test_expected_improvement_steers_to_the_best_candidate():
    """Three EI updates over a fixed candidate set must improve on the best sample and reach the best candidate."""
    model, X, y = fresh('sine')
    candidates = T[:300]
    best = y.min()
    for _ in range(3):
        x = candidates[int(np.argmax(model.infill_ei(candidates)))]
        value = float(sine2(x)[0])
        best = min(best, value)
        model.addPoint(x, value)
    assert best < y.min() - 0.01
    assert best <= sine2(candidates).min() + 0.02


def test_efficient_global_optimisation_of_the_papers_function():
    """Train, add the point of maximum EI, repeat (Jones et al. 1998): from 3 samples, 10 updates find the global
    minimum of f_e(x) = (6x - 2)^2 sin(12x - 4), -6.0207 at x = 0.7572, where 13 random samples do not."""
    X = np.array([0.0, 0.5, 1.0])
    y = R.forrester_fe(X)
    for _ in range(10):
        model = kriging(X, y, seed=0)
        model.train()
        x = model.infill(1, method='ei')[0, 0]
        X, y = np.r_[X, x], np.r_[y, R.forrester_fe(x)]
    assert y.min() == pytest.approx(-6.020740, abs=1e-3)
    assert X[np.argmin(y)] == pytest.approx(0.75725, abs=2e-3)
    assert R.forrester_fe(np.random.default_rng(0).random((20, 13))).min(axis=1).max() > -5.9


# ------------------------------------------------------------------ plots and history
def test_plot_draws_error_prediction_and_surface():
    import matplotlib.pyplot as plt
    model, X, y = fresh('sine', testfunction=sine2)
    model.plot(show=False)
    axes = plt.gcf().axes
    assert len(axes) >= 3 and any(ax.name == '3d' for ax in axes)


def test_plot_shows_the_figure(monkeypatch):
    import matplotlib.pyplot as plt
    shown = []
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: shown.append(1))
    fresh('sine')[0].plot()
    assert shown == [1]


def test_saveFigure_returns_the_plotted_grids():
    model, X, y = fresh('sine')
    spx, spy, gx, gy, Z, Ze = model.saveFigure()
    np.testing.assert_allclose(np.c_[spx, spy], X, atol=1e-12)
    assert gx.shape == gy.shape == Z.shape == Ze.shape == (61, 61)
    pts = np.c_[gx.ravel(), gy.ravel()]
    np.testing.assert_allclose(Z.ravel(), model.predict(pts), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Ze.ravel(), model.predict_var(pts), rtol=1e-8, atol=1e-12)


def test_snapshot_without_test_points():
    model, X, y = fresh('sine')
    model.snapshot()
    h = model.history
    assert h['points'] == [20] and h['neglnlike'] == [model.NegLnLike] and len(h['avgMSE']) == 1
    np.testing.assert_array_equal(h['theta'][0], model.theta)


def test_snapshot_tracks_convergence_at_test_points():
    model, X, y = fresh('sine', testfunction=sine2, testPoints=15)
    model.snapshot()
    model.addPoint(np.array([0.5, 0.5]), float(sine2(np.array([0.5, 0.5]))[0]))
    model.snapshot()
    h = model.history
    assert h['points'] == [20, 21] and len(h['avgMSE']) == 2 and len(h['rsquared']) == 2
    assert all(len(d['predicted']) == 2 and len(d['mse']) == 2 for d in h['pointData'])
    assert 0.9 < h['rsquared'][-1] <= 1.0  # the predictions at the test points barely move


def test_mean_error_over_the_domain():
    model, X, y = trained('sine')
    mean, std = model.calcuatemeanMSE(p2s=200)
    assert np.isfinite(mean) and 0 < mean < 0.2 and std >= 0
    at_samples = model.calcuatemeanMSE(points=X)
    assert at_samples[0] < 1e-4
