"""
Golden numerics at fixed hyper-parameters, checked against the independent float64 reference in _reference.py
rather than frozen numbers: a refactor may reorganise the linear algebra (Cholesky factors, cached weights, batched
predictions) and pass; changing the mathematics (kernel, nugget, the MLEs of mu and sigma^2, normalisation, expected
improvement) fails.
"""
import math

import numpy as np
import pytest

import _reference as R
from pyKriging import kriging
from pyKriging.matrixops import correlation, expected_improvement, log_expected_improvement

CASES = {  # name: (n, k, x scale, theta, p); non-unit input ranges exercise the normalisation
    'n12_k2': (12, 2, [4.0, 10.0], [3.0, 7.0], [2.0, 1.5]),
    'n25_k3': (25, 3, [4.0, 10.0, 1.0], [1.0, 5.0, 0.3], [2.0, 2.0, 1.2]),
    'n8_k1': (8, 1, [3.0], [10.0], [1.9]),
    'n30_k5': (30, 5, [1.0] * 5, [2.0, 3.0, 1.0, 0.5, 4.0], [2.0, 1.5, 2.0, 1.8, 2.0]),
    'n3_k2': (3, 2, [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]),  # the fewest points that make sense in 2-D
}


def _data(n, k, scale, seed=None):
    X = R.lhs(n, k, n if seed is None else seed) * np.asarray(scale) - 1.0
    y = np.sin(3 * X[:, 0]) + (X[:, 1] ** 2 if k > 1 else 0.0) + (X[:, 2] if k > 2 else 0.0)
    return X, y


def _queries(k, scale, m=15, seed=5):
    return np.random.default_rng(seed).random((m, k)) * np.asarray(scale) - 1.0


def _fixed(X, y, theta, p):
    model = kriging(np.array(X, dtype=float), np.array(y, dtype=float))
    model.update(np.r_[theta, p])
    return model


@pytest.fixture(params=sorted(CASES))
def golden(request):
    n, k, scale, theta, p = CASES[request.param]
    X, y = _data(n, k, scale)
    model = _fixed(X, y, theta, p)
    Xn, yn, (xlo, xhi), (ylo, yhi) = R.normalise(X, y)
    ref = R.RefKriging(Xn, yn, theta, p)
    assert np.linalg.cond(ref.Psi) < 1e8, 'the golden cases must be well conditioned'
    Q = _queries(k, scale)
    return model, ref, X, y, Q, (Q - xlo) / (xhi - xlo), (ylo, yhi)


def test_likelihood_mu_and_sigma2(golden):
    model, ref, *_ = golden
    assert model.NegLnLike == pytest.approx(ref.nll, rel=1e-9, abs=1e-9)
    assert model.mu == pytest.approx(ref.mu, rel=1e-9, abs=1e-12)
    assert model.SigmaSqr == pytest.approx(ref.sigma2, rel=1e-9)
    assert model.LnDetPsi == pytest.approx(2 * np.sum(np.log(np.diag(ref.L))), rel=1e-9, abs=1e-9)


def test_predict(golden):
    model, ref, _, _, Q, Qn, (ylo, yhi) = golden
    np.testing.assert_allclose(model.predict(Q), ref.predict(Qn) * (yhi - ylo) + ylo, rtol=1e-9,
                               atol=1e-9 * (yhi - ylo))


def test_standard_errors(golden):
    """predict_var() is (legacy semantics) the standard error sqrt|s^2| in normalised units of y,
    s^2 = sigma^2 (1 - psi' Psi^-1 psi); predicterr() is the same in the units of y."""
    model, ref, _, _, Q, Qn, (ylo, yhi) = golden
    np.testing.assert_allclose(model.predict_var(Q), ref.std(Qn), rtol=1e-7, atol=1e-7 * math.sqrt(ref.sigma2))
    np.testing.assert_allclose(model.predicterr(Q), ref.std(Qn) * (yhi - ylo), rtol=1e-7,
                               atol=1e-7 * math.sqrt(ref.sigma2) * (yhi - ylo))


def test_expected_improvement(golden):
    """expimp() takes normalised points (as infill() uses it) and returns EI [Jones et al. 1998, eq. 15] in
    normalised units with y_min = min(y); infill_ei() takes real-world points and returns EI in the units of y."""
    model, ref, _, _, Q, Qn, (ylo, yhi) = golden
    np.testing.assert_allclose(model.expimp(Qn), ref.ei(Qn), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(model.infill_ei(Q), ref.ei(Qn) * (yhi - ylo), rtol=1e-6, atol=1e-12 * (yhi - ylo))


@pytest.mark.parametrize('w', [0.0, 0.3, 0.5, 1.0])
def test_weighted_expected_improvement(golden, w):
    """Weighted EI (Sobester et al. 2005): w (y_min - f) Phi(u) + (1 - w) s phi(u)."""
    model, ref, _, _, _, Qn, _ = golden
    f, s, ymin = ref.predict(Qn), ref.std(Qn), float(np.min(ref.y))
    u = (ymin - f) / s
    want = w * (ymin - f) * 0.5 * (1 + np.vectorize(math.erf)(u / math.sqrt(2))) \
        + (1 - w) * s * np.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)
    np.testing.assert_allclose(model.weightedexpimp(Qn, w), want, rtol=1e-6, atol=1e-12)
    if w == 0.5:
        np.testing.assert_allclose(model.weightedexpimp(Qn, w), 0.5 * model.expimp(Qn), rtol=1e-6, atol=1e-12)


def test_interpolates_the_data_with_no_error_there(golden):
    model, ref, X, y, *_ = golden
    np.testing.assert_allclose(model.predict(X), y, rtol=0, atol=1e-8 * np.ptp(y))
    assert np.all(model.predict_var(X) <= 1e-4 * math.sqrt(ref.sigma2))
    assert np.all(model.expimp(model.normX(X)) <= 1e-4 * math.sqrt(ref.sigma2))


def test_expected_improvement_is_nonnegative(golden):
    model, *_ = golden
    ei = model.expimp(np.random.default_rng(0).random((200, model.k)))
    assert np.all(ei >= 0) and np.all(np.isfinite(ei))


# ------------------------------------------------------------------ correlation kernel
@pytest.mark.parametrize('p', [[2.0], [2.0, 2.0], [1.5, 2.0], [2.0, 1.0, 1.7, 2.0, 1.99], [2.0] * 9],
                         ids=['k1', 'k2', 'k2_mixed', 'k5_mixed', 'k9'])
def test_correlation_kernel(p):
    """correlation() works on blocks of rows (eq. 2.2), spread over threads for large inputs: single points and
    multi-block inputs must match the reference, and a row must not depend on the block that computed it."""
    k = len(p)
    rng = np.random.default_rng(k)
    A, B, theta = rng.random((3000, k)), rng.random((60, k)), 10 ** rng.uniform(-1, 1.5, k)  # 3000 rows: 2+ blocks
    got = correlation(A, B, theta, p)
    np.testing.assert_allclose(got, R.corr(A, B, theta, p), rtol=1e-12, atol=0)
    np.testing.assert_array_equal(got[::299], np.vstack([correlation(A[i:i + 1], B, theta, p) for i in range(0, 3000, 299)]))
    np.testing.assert_array_equal(np.diag(correlation(B, B, theta, p)), 1.0)


# ------------------------------------------------------------------ expected improvement far in the tail
def test_log_expected_improvement_is_accurate_far_into_the_tail():
    """ln EI must stay accurate where EI itself underflows and the textbook formula cancels (Forrester et al.
    2007, appendix B): compared with a continued-fraction evaluation, from u = 5 down to u = -1e8."""
    u = -np.r_[-5, -1, 0, np.geomspace(1e-3, 1e8, 300)]
    np.testing.assert_allclose(log_expected_improvement(-u, 1.0, 0.0), R.log_h(u), rtol=1e-12)
    rng = np.random.default_rng(1)
    f, s, ymin = rng.normal(0, 1, 200), 10 ** rng.uniform(-6, 1, 200), 0.3
    np.testing.assert_allclose(log_expected_improvement(f, s, ymin), np.log(s) + R.log_h((ymin - f) / s), rtol=1e-12)


def test_expected_improvement_is_zero_without_error_and_matches_the_textbook_formula():
    f, s = np.array([0.1, 0.5, 0.2, 0.3]), np.array([0.0, 0.0, 0.05, 0.4])
    np.testing.assert_array_equal(log_expected_improvement(f, s, 0.2)[:2], [-np.inf, -np.inf])
    np.testing.assert_allclose(expected_improvement(f, s, 0.2), R.ei(f, s, 0.2), rtol=1e-12, atol=0)


# ------------------------------------------------------------------ invariances
def test_affine_invariance_in_y():
    """y is min-max normalised: a y + b gives a f + b, the same likelihood and the same normalised error."""
    n, k, scale, th, p = CASES['n12_k2']
    X, y = _data(n, k, scale)
    m1, m2 = _fixed(X, y, th, p), _fixed(X, 3.5 * y - 40.0, th, p)
    Q = _queries(k, scale, m=5)
    assert m2.NegLnLike == pytest.approx(m1.NegLnLike, rel=1e-9)
    np.testing.assert_allclose(m2.predict(Q), 3.5 * m1.predict(Q) - 40.0, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(m2.predict_var(Q), m1.predict_var(Q), rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(m2.predicterr(Q), 3.5 * m1.predicterr(Q), rtol=1e-7, atol=1e-12)


def test_affine_invariance_in_x():
    n, k, scale, th, p = CASES['n12_k2']
    X, y = _data(n, k, scale)
    a, b = np.array([0.01, 250.0]), np.array([7.0, -3.0])
    m1, m2 = _fixed(X, y, th, p), _fixed(X * a + b, y, th, p)
    Q = _queries(k, scale, m=5)
    np.testing.assert_allclose(m2.predict(Q * a + b), m1.predict(Q), rtol=1e-9, atol=1e-9)


def test_permutation_invariance():
    n, k, scale, th, p = CASES['n25_k3']
    X, y = _data(n, k, scale)
    order = np.random.default_rng(3).permutation(len(y))
    m1, m2 = _fixed(X, y, th, p), _fixed(X[order], y[order], th, p)
    Q = _queries(k, scale, m=5)
    assert m2.NegLnLike == pytest.approx(m1.NegLnLike, rel=1e-9)
    np.testing.assert_allclose(m2.predict(Q), m1.predict(Q), rtol=1e-9, atol=1e-9)
