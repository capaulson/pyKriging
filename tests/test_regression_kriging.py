"""
Regression kriging: Psi + Lambda I (Forrester, Keane & Bressloff 2006; Forrester et al. 2007, section 5a) filters
noise instead of interpolating it. Expected improvement and error-based infill use the re-interpolation error
(section 5b): the error of an interpolation of the regressed data, which vanishes at the samples, so infill does not
re-sample them. Everything is checked against the reference in _reference.py.
"""
import functools

import numpy as np
import pytest

import _reference as R
from pyKriging import kriging, regression_kriging

TH, P, LAMBDA = [2.0, 4.0], [2.0, 1.9], 0.01


def f(X):
    return np.sin(2 * np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1])


@functools.lru_cache(maxsize=None)
def noisy(n=30):
    X = R.lhs(n, 2, 21)
    return X, f(X) + 0.15 * np.random.default_rng(5).standard_normal(n)


@pytest.fixture
def fitted():
    X, y = noisy()
    model = regression_kriging(X, y)
    model.update(np.r_[TH, P, LAMBDA])
    Xn, yn, (xlo, xhi), (ylo, yhi) = R.normalise(X, y)
    return model, R.RefKriging(Xn, yn, TH, P, lam=LAMBDA), X, y, (xlo, xhi, ylo, yhi)


def test_fixed_hyperparameters_match_the_reference(fitted):
    model, ref, X, y, (xlo, xhi, ylo, yhi) = fitted
    assert model.Lambda == LAMBDA
    assert (model.NegLnLike, model.mu, model.SigmaSqr) == pytest.approx((ref.nll, ref.mu, ref.sigma2), rel=1e-9)
    Q = R.lhs(10, 2, 8)
    Qn = (Q - xlo) / (xhi - xlo)
    np.testing.assert_allclose(model.predict(Q), ref.predict(Qn) * (yhi - ylo) + ylo, rtol=1e-9, atol=1e-9)
    # the regression error sigma^2 (1 + Lambda - psi' Psi^-1 psi) (eq. 5.3 for one level)
    np.testing.assert_allclose(model.predict_var(Q), ref.std(Qn, offset=LAMBDA), rtol=1e-8)
    np.testing.assert_allclose(model.predicterr(Q), ref.std(Qn, offset=LAMBDA) * (yhi - ylo), rtol=1e-8)


def test_noisy_data_are_regressed_not_interpolated(fitted):
    model, ref, X, y, (_, _, ylo, yhi) = fitted
    np.testing.assert_allclose(model.predict(X), ref.regressed() * (yhi - ylo) + ylo, rtol=1e-9, atol=1e-9)
    assert np.min(np.abs(model.predict(X) - y)) > 1e-6
    assert np.all(model.predict_var(X) > 0.05 * np.sqrt(ref.sigma2))  # the noise stays in the regression error


def test_expected_improvement_uses_the_reinterpolation_error(fitted):
    """EI of the re-interpolation: its error (Forrester et al. 2006b) and the best regressed value as incumbent."""
    model, ref, X, y, (xlo, xhi, _, _) = fitted
    Qn = (R.lhs(25, 2, 13) - xlo) / (xhi - xlo)
    s = ref.reinterpolation_std(Qn, delta=(10 + ref.n) * R.EPS)
    np.testing.assert_allclose(-model.infill_objective_mse(Qn), s, rtol=1e-6)
    ymin = float(np.min(ref.regressed()))
    np.testing.assert_allclose(model.expimp(Qn), R.ei(ref.predict(Qn), s, ymin), rtol=1e-5, atol=1e-12)
    u = (ymin - ref.predict(Qn)) / s
    w = 0.3
    want = w * (ymin - ref.predict(Qn)) * R.ndtr(u) + (1 - w) * s * np.exp(-u * u / 2) / np.sqrt(2 * np.pi)
    np.testing.assert_allclose(model.weightedexpimp(Qn, w), want, rtol=1e-5, atol=1e-12)


def test_no_expected_improvement_at_the_samples(fitted):
    """The re-interpolation error vanishes at the samples, so neither EI nor error-based infill re-samples them,
    although the regression error does not vanish there."""
    model, ref, X, y, _ = fitted
    Xn = model.normX(X)
    assert np.all(-model.infill_objective_mse(Xn) < 1e-6 * np.sqrt(ref.sigma2))
    assert np.all(model.expimp(Xn) < 1e-6 * np.sqrt(ref.sigma2))
    new = model.infill(2, method='ei')
    assert np.min(np.abs(X[:, None, :] - new[None, :, :]).sum(axis=2)) > 1e-3


def test_hyperparameter_vector_includes_lambda():
    X, y = noisy()
    model = regression_kriging(X, y, seed=0)
    assert model.Lambda == 0.0 and isinstance(model, kriging)
    for bad in ([1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0, 0.1, 0.1]):
        with pytest.raises(ValueError):
            model.update(bad)
    for name in ('regupdatePsi', 'regneglikelihood', 'regression_predicterr_normalized'):  # earlier names
        assert callable(getattr(model, name))
    model.update(np.r_[TH, P, LAMBDA])
    x = np.array([[0.3, 0.6]])
    np.testing.assert_array_equal(model.regression_predicterr_normalized(x), model.predicterr_normalized(x))


def test_a_correlation_matrix_that_is_not_positive_definite_is_not_ignored(fitted):
    """A non-PD Psi must surface (an exception or a fresh likelihood), never leave the previous fit's likelihood."""
    model, *_ = fitted
    good = model.NegLnLike
    try:
        model.update([1e-9, 1e-9, 2.0, 2.0, 0.0])  # Psi ~ all ones and no regression constant
    except Exception:
        return
    assert not np.isclose(model.NegLnLike, good), 'stale likelihood returned for a non-PD Psi'


# ------------------------------------------------------------------ training
@functools.lru_cache(maxsize=None)
def reference_optimum():
    """Dense reference search over regression kriging's own box: theta in [1e-4, 100], p in [1.81231, 2],
    Lambda in [0, 1]."""
    Xn, yn, _, _ = R.normalise(*noisy())
    return R.nll_optimum(Xn, yn, (1e-4, 100.0), (1.81231, 2.0), lam_bounds=(0.0, 1.0))


@pytest.mark.parametrize('optimizer', ['pso', 'ga'])
def test_trained_likelihood_is_the_reference_optimum(optimizer):
    X, y = noisy()
    model = regression_kriging(X, y, seed=0)
    assert (model.thetamin, model.thetamax, model.pmin, model.pmax) == (1e-4, 100.0, 1.81231, 2.0)
    model.train(optimizer=optimizer)
    Xn, yn, _, _ = R.normalise(X, y)
    got = R.concentrated_nll(Xn, yn, model.theta, model.pl, lam=model.Lambda)
    best = reference_optimum()
    assert model.NegLnLike == pytest.approx(got, rel=1e-8)
    assert got <= best[0] + 0.05, \
        f'trained {got:.4f} (Lambda {model.Lambda:.3g}) vs reference {best[0]:.4f} (Lambda {best[3]:.3g})'
    assert 0 <= model.Lambda <= 1


def test_regression_filters_noise_better_than_interpolation():
    X, y = noisy()
    G = R.lhs(400, 2, 77)
    regression = regression_kriging(X, y, seed=0)
    regression.train()
    interpolation = kriging(X, y, seed=0)
    interpolation.train()
    rmse_r = np.sqrt(np.mean((regression.predict(G) - f(G)) ** 2))
    rmse_k = np.sqrt(np.mean((interpolation.predict(G) - f(G)) ** 2))
    assert 1e-3 < regression.Lambda < 0.5
    assert rmse_r < 0.85 * rmse_k, f'regression {rmse_r:.4f} vs interpolation {rmse_k:.4f}'
    assert rmse_r < 0.15  # below the noise level


def test_replicated_observations():
    """Repeated noisy measurements at the same points, which ordinary kriging cannot interpolate: the regression
    passes near the replicate means, closer to the truth than single measurements, and does not resample."""
    X0 = R.lhs(12, 2, 3)
    X = np.vstack([X0, X0, X0])
    y = f(X) + 0.1 * np.random.default_rng(0).standard_normal(36)
    model = regression_kriging(X, y, seed=0)
    model.train()
    assert model.Lambda > 1e-4
    error = model.predict(X0) - f(X0)
    assert np.sqrt(np.mean(error ** 2)) < 0.7 * np.sqrt(np.mean((y - f(X)) ** 2))
    assert np.max(np.abs(model.predict(X0) - y.reshape(3, 12).mean(axis=0))) < 0.1
    new = model.infill(1, method='ei')
    assert np.min(np.abs(X0 - new).sum(axis=1)) > 1e-3
