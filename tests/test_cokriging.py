"""
Co-kriging (Forrester, Sobester & Keane 2007, section 3) on the paper's one-variable example (section 3a),

    f_e(x) = (6x - 2)^2 sin(12x - 4),   f_c(x) = A f_e(x) + B (x - 0.5) - C,   A = 0.5, B = 10, C = -5,
    X_c = {0, 0.1, ..., 1},   X_e = {0, 0.4, 0.6, 1},

and on two-variable problems. The paper's GA found rho = 1.87 ("close to the true value of 2"); the exact MLE is
~2.0. Predictions and errors are checked against _reference.py: a from-scratch fit, and the joint covariance matrix C
(eq. 3.2) solved directly at the library's hyper-parameters, for Kennedy & O'Hagan's per-level means (the default,
computed level by level when X_e is a subset of X_c) and for the single mean of eq. (3.11) (mean='paper').
"""
import functools

import numpy as np
import pytest
from scipy.optimize import minimize

import _reference as R
from pyKriging import kriging, samplingplan, testfunctions
from pyKriging.coKriging import coKriging

XC, XE = R.PAPER_XC, R.PAPER_XE
GRID = np.linspace(0, 1, 101)


def fe(x):
    return R.forrester_fe(x)


def fc(x, A=0.5, B=10.0, C=-5.0):
    return R.forrester_fc(x, A, B, C)


@functools.lru_cache(maxsize=None)
def paper_model(mean='koh', A=0.5, B=10.0, C=-5.0):
    ck = coKriging(XC, fc(XC, A, B, C), XE, fe(XE), seed=0, mean=mean)
    ck.train()
    return ck


@pytest.fixture(scope='module')
def reference():
    return R.RefCoKriging(XC, fc(XC), XE, fe(XE))


def rmse_of_kriging_through_ye():
    """Ordinary kriging through the four expensive points only (p = 2, MLE theta): the paper's foil."""
    grid = np.linspace(-3, 2, 501)
    best = min((R.RefKriging(XE, fe(XE), [10 ** t], [2.0]).nll, t) for t in grid)
    model = R.RefKriging(XE, fe(XE), [10 ** best[1]], [2.0])
    return np.sqrt(np.mean((model.predict(GRID) - fe(GRID)) ** 2))


def fe2(X):
    return np.sin(3 * X[:, 0]) + X[:, 1] ** 2


def fc2(X):
    return 0.8 * fe2(X) + 0.3 * X[:, 0]


def scaling(Xc, yc, Xe, ye):
    """The co-kriging units: one min-max scaling of x and one of y shared by both levels."""
    X, y = np.r_[np.reshape(Xc, (len(Xc), -1)), np.reshape(Xe, (len(Xe), -1))], np.r_[yc, ye]
    xlo, xspan, ylo, yspan = X.min(0), np.ptp(X, 0), y.min(), np.ptp(y)
    return lambda x: (np.reshape(x, (len(x), -1)) - xlo) / xspan, lambda v: (v - ylo) / yspan, ylo, yspan


def assert_matches_the_joint_reference(ck, Xc, yc, Xe, ye, x, tol=1e-6):
    """Predictor, mean squared error and EI of the library equal those of the joint matrix C at the library's
    hyper-parameters. The MSE is a difference of terms of the size of the prior variance rho^2 sigma_c^2 + sigma_d^2,
    so its rounding error scales with that; tol allows for the conditioning of C (formula errors are far larger)."""
    nx, ny, ylo, yspan = scaling(Xc, yc, Xe, ye)
    f, mse, mu = R.cokriging_predict(nx(Xc), ny(yc), nx(Xe), ny(ye), ck.thetac, ck.pc, ck.thetad, ck.pd, ck.rho,
                                     ck.SigmaSqrc, ck.SigmaSqrd, nx(x), mean=ck.mean)
    prior = ck.rho ** 2 * ck.SigmaSqrc + ck.SigmaSqrd
    assert ck.mu == pytest.approx(mu, rel=100 * tol, abs=tol)  # the mean of the expensive process
    np.testing.assert_allclose(ck.predict(x), f * yspan + ylo, rtol=tol, atol=tol * yspan)
    np.testing.assert_allclose(ck.predict_var(x) ** 2, mse, rtol=tol, atol=1e-4 * tol * prior)
    s = np.sqrt(np.abs(mse))
    ei = R.ei(f, s, ny(ye).min())
    keep = (s > 1e-3 * s.max()) & (ei > 1e-10 * ei.max())  # EI jumps where s rounds to 0; its tail is ill-conditioned
    np.testing.assert_allclose(ck.expimp(nx(x[keep])), ei[keep], rtol=100 * tol)


# ------------------------------------------------------------------ the reference reproduces the paper
def test_reference_reproduces_the_paper(reference):
    assert np.log10(reference.theta_c) == pytest.approx(1.2073, abs=2e-3)  # the value in examples/coKriging.py
    assert 1.8 <= reference.rho <= 2.1
    np.testing.assert_allclose(reference.predict(XE), fe(XE), atol=1e-6 * np.ptp(fe(XE)))
    rmse = np.sqrt(np.mean((reference.predict(GRID) - fe(GRID)) ** 2))
    assert rmse < 0.1 * rmse_of_kriging_through_ye()


# ------------------------------------------------------------------ the paper's example
def test_constructor_is_quiet(capsys):
    coKriging(XC, fc(XC), XE, fe(XE))
    assert capsys.readouterr().out == ''


def test_rho_matches_the_paper():
    assert 1.8 <= paper_model().rho <= 2.2  # paper 1.87, exact MLE ~2.0, truth 1/A = 2


@pytest.mark.parametrize('A', [0.25, 1.0, -0.5])
def test_rho_tracks_one_over_A(A):
    """Paper fig. 3: rho ~ 1/A (B and C have no effect on rho)."""
    assert paper_model(A=A).rho == pytest.approx(1.0 / A, rel=0.1)


def test_interpolates_the_expensive_data_and_its_error_vanishes_there():
    ck = paper_model()
    np.testing.assert_allclose(ck.predict(XE), fe(XE), atol=1e-6 * np.ptp(fe(XE)))
    error = ck.predict_var(GRID)
    assert np.all(error >= 0) and np.max(ck.predict_var(XE)) < 1e-4
    assert np.max(ck.predict_var(XC)) < 1e-2 * error.max()  # 'the estimated error reduces almost to 0 at X_c' (fig. 2)


def test_beats_kriging_through_the_expensive_data():
    ck = paper_model()
    rmse = np.sqrt(np.mean((ck.predict(GRID) - fe(GRID)) ** 2))
    assert rmse < 0.02 * np.ptp(fe(GRID))
    assert rmse < 0.1 * rmse_of_kriging_through_ye()
    k = kriging(XE, fe(XE), seed=0)  # and the library's own kriging through y_e
    k.train()
    assert rmse < 0.1 * np.sqrt(np.mean((k.predict(GRID) - fe(GRID)) ** 2))


@pytest.mark.parametrize('mean', ['koh', 'paper'])
def test_close_to_the_reference_fit(reference, mean):
    """Both at (near-)maximum-likelihood hyper-parameters: predictions agree to a few % of the range."""
    want = reference.predict(GRID)
    assert np.max(np.abs(paper_model(mean).predict(GRID) - want)) < 0.05 * np.ptp(want)


@pytest.mark.parametrize('mean', ['koh', 'paper'])
def test_joint_covariance_predictor_and_error_at_the_trained_hyperparameters(mean):
    """Eqs. (3.11) and (3.12). With per-level means and X_e a subset of X_c the library computes rho y_c(x) + y_d(x)
    and rho^2 s_c^2 + s_d^2 level by level (Le Gratiet & Garnier 2014): they must equal the joint solution."""
    assert_matches_the_joint_reference(paper_model(mean), XC, fc(XC), XE, fe(XE), GRID)


@pytest.mark.parametrize('nested', [True, False], ids=['nested', 'not_nested'])
@pytest.mark.parametrize('mean', ['koh', 'paper'])
def test_joint_covariance_predictor_and_error_at_set_hyperparameters(nested, mean):
    """The same at well-conditioned hyper-parameters set by hand (cheap level first), to near machine precision."""
    Xc = R.lhs(15, 2, 4)
    Xe = Xc[::3] if nested else R.lhs(6, 2, 5) * 0.8 + 0.1
    ck = coKriging(Xc, fc2(Xc), Xe, fe2(Xe), seed=0, mean=mean)
    ck.train()
    ck.cheap.update([6.0, 3.0, 2.0, 1.8])
    ck.update([4.0, 8.0, 1.9, 2.0])
    np.testing.assert_array_equal(np.r_[ck.thetac, ck.pc, ck.thetad, ck.pd], [6.0, 3.0, 2.0, 1.8, 4.0, 8.0, 1.9, 2.0])
    assert_matches_the_joint_reference(ck, Xc, fc2(Xc), Xe, fe2(Xe), R.lhs(30, 2, 6), tol=1e-9)


def test_rho_and_mu_d_are_the_generalised_least_squares_estimates():
    """For given theta_d, p_d (set here with update(); the trained theta_d of the paper's example sits at its lower
    bound, where Psi_d is too ill-conditioned for a 1e-8 comparison) the MLEs of mu_d and rho are the GLS fit of
    y_e on [1, y_c(X_e)], and sigma_d^2 its residual quadratic form / n_e."""
    ck = coKriging(XC, fc(XC), XE, fe(XE), seed=0)
    ck.train()
    ck.update([10.0, 1.7])
    nx, ny, _, _ = scaling(XC, fc(XC), XE, fe(XE))
    Psi_d = R.corr(nx(XE), nx(XE), [10.0], [1.7]) + R.EPS * np.eye(len(XE))
    (mud, rho), q = R.gls(Psi_d, np.c_[np.ones(len(XE)), ny(fc(XE))], ny(fe(XE)))
    assert (ck.rho, ck.mud, ck.SigmaSqrd) == pytest.approx((rho, mud, q / len(XE)), rel=1e-8)
    np.testing.assert_array_equal(np.r_[ck.thetad, ck.pd], [10.0, 1.7])
    assert_matches_the_joint_reference(ck, XC, fc(XC), XE, fe(XE), GRID)


def test_difference_hyperparameters_maximise_the_likelihood():
    """theta_d, p_d (with rho and mu_d profiled) reach the optimum of eq. (3.10) found by a dense reference search
    over the library's box (theta in [1e-4, 100], p in [1, 2])."""
    ck = paper_model()
    nx, ny, _, _ = scaling(XC, fc(XC), XE, fe(XE))
    args = nx(XE), ny(fe(XE)), ny(fc(XE))
    grid = [(R.difference_nll(*args, [10 ** t], [p])[0], t, p)
            for t in np.linspace(-4, 2, 121) for p in np.linspace(1, 2, 21)]
    best = min(minimize(lambda v: min(R.difference_nll(*args, [10 ** v[0]], [v[1]])[0], 1e300), [t, p],
                        method='L-BFGS-B', bounds=[(-4, 2), (1, 2)]).fun for _, t, p in sorted(grid)[:5])
    got = R.difference_nll(*args, ck.thetad, ck.pd)[0]
    assert got <= best + 0.05


def test_a_purely_scaled_cheap_function_leaves_no_difference_process():
    """B = 0: d = y_e - rho y_c is constant at rho = 1/A, so sigma_d^2 -> 0 (paper fig. 4) and the expensive
    function is predicted as well as the cheap one, despite the singular joint covariance."""
    ck = paper_model(B=0.0)
    assert ck.rho == pytest.approx(2.0, rel=1e-3)
    assert ck.SigmaSqrd < 1e-6 * ck.SigmaSqrc
    np.testing.assert_allclose(ck.predict(XE), fe(XE), atol=1e-5 * np.ptp(fe(XE)))
    assert np.sqrt(np.mean((ck.predict(GRID) - fe(GRID)) ** 2)) < 0.02 * np.ptp(fe(GRID))


# ------------------------------------------------------------------ API
def test_output_shapes_and_types():
    ck = paper_model()
    assert isinstance(ck.predict(0.3), float) and isinstance(ck.predict_var([0.3]), float)
    assert isinstance(ck.expimp(0.3), float) and isinstance(ck.rho, float)
    assert ck.predict(np.linspace(0, 1, 7)).shape == (7,) and ck.predict_var(np.linspace(0, 1, 7)).shape == (7,)
    ei = ck.expimp(np.linspace(0, 1, 50))
    assert ei.shape == (50,) and np.all(ei >= 0)


def test_unknown_mean_is_rejected():
    with pytest.raises(ValueError):
        coKriging(XC, fc(XC), XE, fe(XE), mean='universal')


def test_training_is_reproducible_given_a_seed():
    a, b = (coKriging(XC, fc(XC), XE, fe(XE), seed=3) for _ in range(2))
    a.train()
    b.train()
    assert a.rho == b.rho
    np.testing.assert_array_equal(a.predict(GRID), b.predict(GRID))


# ------------------------------------------------------------------ two variables
def test_two_variables_with_nested_designs():
    Xc = R.lhs(20, 2, 1)
    Xe = Xc[::4]
    ck = coKriging(Xc, fc2(Xc), Xe, fe2(Xe), seed=0)
    ck.train()
    np.testing.assert_allclose(ck.predict(Xe), fe2(Xe), atol=1e-4 * np.ptp(fe2(Xe)))
    assert_matches_the_joint_reference(ck, Xc, fc2(Xc), Xe, fe2(Xe), R.lhs(40, 2, 2), tol=1e-5)  # cond(Psi_c) ~ 1e13


def test_two_variables_without_collocated_points():
    """X_e is not a subset of X_c: y_c(X_e) comes from the cheap model (paper, footnote 3) and predictions from the
    joint matrix C. Co-kriging must still beat kriging of the expensive data alone."""
    branin = testfunctions().branin

    def cheap(X):
        return 0.5 * branin(X) + 5 * (X[:, 0] - 0.5) + 3 * X[:, 1]

    Xc = samplingplan(2, random_state=1).optimallhc(30)
    Xe = samplingplan(2, random_state=2).rlh(8) * 0.9 + 0.05
    ck = coKriging(Xc, cheap(Xc), Xe, branin(Xe), seed=1)
    ck.train()
    Xt = np.random.default_rng(2).random((500, 2))
    assert_matches_the_joint_reference(ck, Xc, cheap(Xc), Xe, branin(Xe), Xt[:40])
    np.testing.assert_allclose(ck.predict(Xe), branin(Xe), atol=1e-5 * np.ptp(branin(Xe)))
    k = kriging(Xe, branin(Xe), seed=1)
    k.train()
    co_err, kr_err = np.abs(ck.predict(Xt) - branin(Xt)).mean(), np.abs(k.predict(Xt) - branin(Xt)).mean()
    assert co_err < 0.25 * kr_err
