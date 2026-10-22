"""
Ordinary kriging: the public API, input handling and the consistency of the model state (normalisation, update,
addPoint). Numerical values are checked against the independent float64 reference in _reference.py.
"""
import numpy as np
import pytest

import _reference as R
import pyKriging
from pyKriging import gpu_backend, kriging

TH, P = [0.5, 2.0], [2.0, 2.0]


def _data(n=8, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2)) * 10
    return X, np.sin(X[:, 0]) + X[:, 1]


def fixed(X, y, theta=TH, p=P, **kwargs):
    """Model of (X, y) fitted at fixed hyper-parameters (no training)."""
    model = kriging(X, y, **kwargs)
    model.update(np.r_[theta, p])
    return model


def reference(X, y, theta=TH, p=P):
    """Reference model of (X, y) and its maps real-world x -> normalised x and normalised f -> real-world f."""
    Xn, yn, (xlo, xhi), (ylo, yhi) = R.normalise(X, y)
    return (R.RefKriging(Xn, yn, theta, p), lambda x: (np.asarray(x, dtype=float) - xlo) / (xhi - xlo),
            lambda f: f * (yhi - ylo) + ylo)


# ------------------------------------------------------------------ API surface
LEGACY_METHODS = ['train', 'predict', 'predict_var', 'expimp', 'weightedexpimp', 'infill', 'addPoint', 'update',
                  'updateModel', 'updateData', 'updatePsi', 'neglikelihood', 'snapshot', 'plot', 'saveFigure',
                  'calcuatemeanMSE', 'normX', 'inversenormX', 'normy', 'inversenormy', 'normalizeData',
                  'predict_normalized', 'predicterr_normalized', 'fittingObjective', 'fittingObjective_local',
                  'infill_objective_mse', 'infill_objective_ei', 'rsquared', 'adjrsquares', 'chisquared',
                  'predicterr', 'infill_ei']
LEGACY_ATTRIBUTES = ['theta', 'pl', 'mu', 'SigmaSqr', 'NegLnLike', 'LnDetPsi', 'Psi', 'U', 'X', 'y', 'n', 'k',
                     'normRange', 'ynormRange', 'history', 'thetamin', 'thetamax', 'pmin', 'pmax', 'name',
                     'testfunction', 'testPoints']


def test_public_surface():
    X, y = _data()
    model = fixed(X, y)
    for name in LEGACY_METHODS:
        assert callable(getattr(kriging, name, None)), f'kriging.{name} is missing'
    for name in LEGACY_ATTRIBUTES:
        assert hasattr(model, name), f'kriging instance lacks .{name}'
    kriging(X, y, testfunction=None, name='n', testPoints=None, seed=0)  # the documented signature


def test_package_names_and_star_import():
    for name in ('kriging', 'regression_kriging', 'samplingplan', 'testfunctions', 'saveModel', 'loadModel',
                 'splitArrays', 'mse', 'norm', 'configure_gpu', 'get_device_info', 'is_gpu_available',
                 'get_backend', 'to_cpu', 'to_gpu'):
        assert hasattr(pyKriging, name), f'pyKriging.{name} is missing'
    namespace = {}
    exec('from pyKriging import *', namespace)
    assert set(pyKriging.__all__) <= set(namespace)
    assert namespace['kriging'] is kriging
    from pyKriging.coKriging import coKriging
    from pyKriging.CrossValidation import Cross_Validation
    from pyKriging.GlobalSensitivity import SobolSensitivity
    assert namespace['coKriging'].coKriging is coKriging  # the last three names of __all__ are submodules
    assert namespace['CrossValidation'].Cross_Validation is Cross_Validation
    assert namespace['GlobalSensitivity'].SobolSensitivity is SobolSensitivity


def test_construction():
    X, y = _data(n=10)
    model = kriging(X, y, name='test')
    assert (model.n, model.k, model.name) == (10, 2, 'test')
    assert model.X.shape == (10, 2) and model.y.shape == (10,)
    np.testing.assert_array_equal(model.theta, [1.0, 1.0])  # defaults before training: theta = 1, p = 2
    np.testing.assert_array_equal(model.pl, [2.0, 2.0])
    assert model.mu is None


# ------------------------------------------------------------------ normalisation
def test_inputs_and_responses_are_normalised_to_the_unit_range():
    X, y = np.array([[0, 0], [10, 20], [5, 10]]), np.array([100, 200, 150])  # integers, as users pass them
    model = kriging(X, y)
    np.testing.assert_array_equal(model.X.min(axis=0), [0, 0])
    np.testing.assert_array_equal(model.X.max(axis=0), [1, 1])
    np.testing.assert_allclose(model.X, [[0, 0], [1, 1], [0.5, 0.5]])
    np.testing.assert_allclose(model.y, [0, 1, 0.5])
    assert np.allclose(model.normRange, [[0, 10], [0, 20]]) and np.allclose(model.ynormRange, [100, 200])
    x, f = np.array([[2.5, 5.0], [7.0, 1.0]]), np.array([120.0, 180.0])
    np.testing.assert_allclose(model.inversenormX(model.normX(x)), x)
    np.testing.assert_allclose(model.normX(x), [[0.25, 0.25], [0.7, 0.05]])
    np.testing.assert_allclose(model.normy(f), [0.2, 0.8])
    np.testing.assert_allclose(model.inversenormy(model.normy(f)), f)


def test_integer_inputs_give_the_same_model_as_float_inputs():
    Xi, yi = np.array([[0, 0], [10, 20], [3, 10], [7, 4]]), np.array([100, 200, 130, 170])
    mi, mf = fixed(Xi, yi), fixed(Xi.astype(float), yi.astype(float))
    np.testing.assert_allclose(mi.X, mf.X)
    q = np.array([[3, 10], [5.5, 7.25], [9, 1]])
    np.testing.assert_allclose(mi.predict(q), mf.predict(q), rtol=1e-12)
    assert mi.predict([3, 10]) == pytest.approx(130.0, abs=1e-6)


def test_caller_arrays_are_not_modified():
    X, y = _data()
    X0, y0 = X.copy(), y.copy()
    model = fixed(X, y)
    model.addPoint(X[0] * 0.5 + X[1] * 0.5, 3.0)
    q = np.array([3.3, 4.4])
    model.predict(q), model.predict_var(q), model.predicterr(q), model.infill_ei(q), model.expimp(model.normX(q))
    np.testing.assert_array_equal(X, X0)
    np.testing.assert_array_equal(y, y0)
    np.testing.assert_array_equal(q, [3.3, 4.4])


@pytest.mark.parametrize('variant', ['X_list', 'y_column', 'y_list', 'float32'])
def test_array_like_inputs_are_accepted(variant):
    X, y = _data()
    want = fixed(X, y).predict([3.3, 4.4])
    Xa, ya = {'X_list': (X.tolist(), y), 'y_column': (X, y.reshape(-1, 1)), 'y_list': (X, y.tolist()),
              'float32': (X.astype(np.float32), y.astype(np.float32))}[variant]
    model = fixed(Xa, ya)
    assert model.predict([3.3, 4.4]) == pytest.approx(want, rel=1e-5)
    for name in ('X', 'y', 'theta', 'pl', 'Psi', 'U'):  # the model state is float64 whatever the input
        assert getattr(model, name).dtype == np.float64, name


def test_constant_input_column_is_ignored():
    """A constant input has no range: it normalises to 0 instead of 0/0, and the model is the 1-D model of the
    other input."""
    X, y = _data()
    X[:, 1] = 5.0
    model = fixed(X, y, theta=[0.5, 7.0])
    assert np.all(np.isfinite(model.X))
    one_d = fixed(X[:, :1], y, theta=[0.5], p=[2.0])
    x = np.linspace(0.5, 9.5, 7)
    np.testing.assert_allclose(model.predict(np.c_[x, np.full(7, 5.0)]), one_d.predict(x), rtol=1e-12)


def test_mismatched_data_is_rejected():
    X, y = _data()
    with pytest.raises(ValueError):
        kriging(X, y[:-1])


# ------------------------------------------------------------------ predictions: shapes and units
def test_one_point_gives_floats_and_many_points_give_arrays():
    X, y = _data()
    model = fixed(X, y)
    x = [3.3, 4.4]
    for method in (model.predict, model.predict_var, model.predicterr, model.infill_ei):
        value = method(x)
        assert isinstance(value, float), method.__name__
    for method in (model.expimp, model.predict_normalized, model.predicterr_normalized):
        assert isinstance(method(model.normX(x)), float), method.__name__
    assert isinstance(model.weightedexpimp(model.normX(x), 0.5), float)
    assert model.predict(np.array([x])).shape == (1,)  # an (1, k) array is a batch of one


@pytest.mark.parametrize('m', [1, 3, 8])  # 8 == n
def test_batches_equal_point_by_point_results(m):
    X, y = _data()
    model = fixed(X, y)
    Q = np.random.default_rng(2).random((m, 2)) * 10
    for method, normalised in ((model.predict, False), (model.predict_var, False), (model.predicterr, False),
                               (model.infill_ei, False), (model.expimp, True), (model.weightedexpimp, True)):
        points = model.normX(Q) if normalised else Q
        args = (0.3,) if method == model.weightedexpimp else ()
        batch = method(points, *args)
        assert batch.shape == (m,), method.__name__
        np.testing.assert_allclose(batch, [method(q, *args) for q in points], rtol=1e-12, atol=1e-15)


def test_real_world_and_normalised_units():
    """predict_var and expimp keep the legacy normalised units (y range = 1, expimp takes normalised x);
    predicterr and infill_ei are in the units of y and take real-world x."""
    X, y = _data()
    y = 50.0 * y + 7.0
    model = fixed(X, y)
    ref, normx, _ = reference(X, y)
    Q = np.random.default_rng(3).random((6, 2)) * 10
    span = np.ptp(y)
    np.testing.assert_allclose(model.predict_var(Q), ref.std(normx(Q)), rtol=1e-9)
    np.testing.assert_allclose(model.predicterr(Q), span * ref.std(normx(Q)), rtol=1e-9)
    np.testing.assert_allclose(model.expimp(normx(Q)), ref.ei(normx(Q)), rtol=1e-7, atol=1e-15)
    np.testing.assert_allclose(model.infill_ei(Q), span * ref.ei(normx(Q)), rtol=1e-7, atol=1e-15)
    np.testing.assert_allclose(model.predict(normx(Q), norm=False), model.predict(Q), rtol=1e-12)
    np.testing.assert_allclose(model.predict_var(normx(Q), norm=False), model.predict_var(Q), rtol=1e-12)
    np.testing.assert_allclose(model.predict_normalized(normx(Q)), model.normy(model.predict(Q)), rtol=1e-12)
    np.testing.assert_allclose(model.predicterr_normalized(normx(Q)), model.predict_var(Q), rtol=1e-12)


def test_one_dimensional_models_take_scalars():
    X = np.linspace(0, 2, 7)  # 1-D X: a one-variable problem
    model = fixed(X, np.sin(3 * X), theta=[3.0], p=[2.0])
    assert model.k == 1
    want = model.predict([0.5])
    for x in (0.5, np.float64(0.5), np.array([0.5]), [0.5]):
        assert model.predict(x) == pytest.approx(want, rel=1e-12)
    np.testing.assert_allclose(model.predict(np.array([0.5, 1.5])), [want, model.predict(1.5)], rtol=1e-12)
    assert model.predict(np.array([[0.5], [1.5]])).shape == (2,)


def test_wrong_point_shapes_are_rejected():
    X, y = _data()
    model = fixed(X, y)
    for bad in ([1.0, 2.0, 3.0], np.ones((4, 3)), np.ones((2, 2, 2))):
        with pytest.raises(ValueError):
            model.predict(bad)


def test_an_unfitted_model_cannot_predict():
    X, y = _data()
    with pytest.raises(RuntimeError, match='not fitted'):
        kriging(X, y).predict([3.3, 4.4])


# ------------------------------------------------------------------ model state
def test_update_makes_the_model_usable():
    X, y = _data()
    model = kriging(X.copy(), y.copy())
    q = np.array([3.3, 4.4])
    for theta, p in (([2.0, 3.0], [2.0, 2.0]), ([20.0, 0.3], [2.0, 1.5])):
        model.update(np.r_[theta, p])
        ref, normx, realf = reference(X, y, theta, p)
        np.testing.assert_array_equal(np.r_[model.theta, model.pl], np.r_[theta, p])
        assert (model.mu, model.SigmaSqr, model.NegLnLike) == pytest.approx((ref.mu, ref.sigma2, ref.nll), rel=1e-9)
        assert model.predict(q) == pytest.approx(realf(ref.predict(normx(q))[0]), rel=1e-9)


def test_update_rejects_the_wrong_number_of_hyperparameters():
    X, y = _data()
    with pytest.raises(ValueError):
        kriging(X, y).update([1.0, 2.0, 2.0])


def test_hyperparameters_without_a_positive_definite_correlation_matrix_raise():
    X, y = _data()
    model = kriging(X, y)
    with pytest.raises(Exception, match='bad params'):
        # p = 4 is not a valid kernel: here min eig(Psi) = -0.36. (A Psi singular only to round-off, such as
        # theta ~ 0, passes or fails depending on the LAPACK library.)
        model.update([1.0, 1.0, 4.0, 4.0])


def test_default_hyperparameters_give_the_reference_likelihood():
    X, y = _data()
    model = kriging(X, y)
    model.updateModel()  # theta = 1, p = 2
    ref, _, _ = reference(X, y, [1.0, 1.0], [2.0, 2.0])
    assert np.isfinite(model.NegLnLike)
    assert model.NegLnLike == pytest.approx(ref.nll, rel=1e-9)
    assert model.LnDetPsi == pytest.approx(2 * np.sum(np.log(np.diag(ref.L))), rel=1e-9)


def test_addPoint_refits_the_model_to_the_augmented_data():
    """addPoint keeps the normalisation and hyper-parameters, refits mu and sigma^2 and interpolates the new point."""
    X, y = _data(n=10, seed=0)
    theta = [0.2, 0.3]
    model = fixed(X.copy(), y.copy(), theta=theta)
    x_new, y_new = np.array([5.0, 5.0]), float(np.sin(5.0) + 5.0)  # inside the data's box
    model.addPoint(x_new, y_new)
    assert model.n == 11 and model.X.shape == (11, 2)
    xlo, xhi, ylo, yhi = X.min(0), X.max(0), y.min(), y.max()
    ref = R.RefKriging((np.vstack([X, x_new]) - xlo) / (xhi - xlo), (np.r_[y, y_new] - ylo) / (yhi - ylo), theta, P)
    assert model.predict(x_new) == pytest.approx(y_new, abs=1e-8 * (yhi - ylo))
    assert (model.mu, model.SigmaSqr, model.NegLnLike) == pytest.approx((ref.mu, ref.sigma2, ref.nll), rel=1e-9)
    q = np.array([2.0, 7.0])
    assert model.predict(q) == pytest.approx(ref.predict((q - xlo) / (xhi - xlo))[0] * (yhi - ylo) + ylo, rel=1e-9)
    np.testing.assert_array_equal(np.r_[model.theta, model.pl], np.r_[theta, P])


def test_addPoint_outside_the_original_box():
    X, y = _data(n=10, seed=0)
    model = fixed(X.copy(), y.copy(), theta=[0.2, 0.3])
    x_new = X.max(0) + 1.0
    model.addPoint(x_new, 42.0)
    assert model.predict(x_new) == pytest.approx(42.0, rel=1e-8)
    assert 0 <= model.predict_var(x_new) < 1e-4


def test_addPoint_retrains_when_the_hyperparameters_no_longer_fit(monkeypatch):
    """If Psi is not positive definite at the current hyper-parameters once the point is added (simulated here: a
    new point next to a sample can do that), addPoint retrains instead of failing."""
    X, y = _data(n=10, seed=0)
    model = fixed(X, y)
    refit, failures, trained = kriging.updatePsi, [np.linalg.LinAlgError('not positive definite')], []

    def updatePsi(self):
        if failures:
            raise failures.pop()
        refit(self)

    monkeypatch.setattr(kriging, 'updatePsi', updatePsi)
    monkeypatch.setattr(kriging, 'train',
                        lambda self, optimizer='pso': trained.append(self.n) or self.update([1.0, 1.0, 2.0, 2.0]))
    model.addPoint([5.0, 5.0], 1.0)
    assert trained == [11] and model.n == 11
    assert model.predict([5.0, 5.0]) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------ the suite's own isolation
def test_isolation_part1_configures_a_cuda_backend(monkeypatch):
    monkeypatch.setattr(gpu_backend, '_cuda_available', lambda: True)
    backend = pyKriging.configure_gpu('auto')
    assert backend.backend_type == 'cuda' and pyKriging.get_backend() is backend


def test_the_default_backend_is_the_cpu_even_with_cuda(monkeypatch):
    """The GPU is opt-in (configure_gpu): a fresh process computes on the CPU although PyTorch finds CUDA, which
    is_gpu_available() reports."""
    monkeypatch.setattr(gpu_backend, '_backend', None)
    monkeypatch.setattr(gpu_backend, '_cuda_available', lambda: True)
    assert pyKriging.get_backend().backend_type == 'cpu' and pyKriging.is_gpu_available()


def test_isolation_part2_starts_on_the_cpu_again():
    """Runs after part 1: the autouse fixture in conftest.py must have restored the CPU backend."""
    assert pyKriging.get_device_info()['backend'] == 'cpu' and not pyKriging.is_gpu_available()


@pytest.mark.parametrize('installed', [True, False], ids=['threadpoolctl', 'without'])
def test_predictions_do_not_depend_on_the_optional_blas_thread_limit(monkeypatch, installed):
    from pyKriging import matrixops
    if installed:
        pytest.importorskip('threadpoolctl')
    X, y = _data()
    model = kriging(X, y)
    model.update([1.0, 1.0, 2.0, 2.0])
    pts = np.random.default_rng(3).random((20_000, 2)) * 10  # large enough to limit BLAS threads
    expected = model.predict(pts), model.predict_var(pts)
    monkeypatch.setattr(matrixops, '_blas', None if installed else False)  # None: look threadpoolctl up again
    np.testing.assert_array_equal(model.predict(pts), expected[0])
    np.testing.assert_array_equal(model.predict_var(pts), expected[1])
