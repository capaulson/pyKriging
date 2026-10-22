"""Utilities: saving and loading models, cross-validation folds, normalisation and squared errors."""
import numpy as np
import pytest

import _reference as R
from pyKriging import kriging, loadModel, mse, norm, regression_kriging, saveModel, splitArrays


def fitted(cls=kriging, n=8, **kwargs):
    X = R.lhs(n, 2, 3) * [4.0, 2.0]
    model = cls(X, np.sin(X[:, 0]) * np.cos(X[:, 1]), seed=0, **kwargs)
    model.update([5.0, 5.0, 2.0, 2.0] + ([0.01] if cls is regression_kriging else []))
    return model


@pytest.mark.parametrize('cls', [kriging, regression_kriging])
def test_save_and_load_round_trip(cls, tmp_path):
    model = fitted(cls, testfunction=lambda x: np.sin(x[..., 0]))  # dill handles functions such as lambdas
    path = tmp_path / 'model.pkl'
    saveModel(model, str(path))
    loaded = loadModel(str(path))
    assert type(loaded) is cls and loaded.n == model.n
    Q = R.lhs(10, 2, 9) * [4.0, 2.0]
    np.testing.assert_array_equal(loaded.predict(Q), model.predict(Q))
    np.testing.assert_array_equal(loaded.predicterr(Q), model.predicterr(Q))
    np.testing.assert_array_equal(loaded.testfunction(Q), np.sin(Q[:, 0]))
    loaded.addPoint([2.0, 1.0], 0.5)  # a loaded model is a working model
    assert loaded.n == model.n + 1
    if cls is kriging:  # regression kriging regresses rather than interpolates
        assert loaded.predict([2.0, 1.0]) == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize('q', [2, 3, 5, 8])
def test_split_arrays_partitions_the_samples(q):
    model = fitted()
    folds = list(splitArrays(model, q))
    assert len(folds) == q
    tests = np.vstack([f[2] for f in folds])
    assert sorted(map(tuple, tests)) == sorted(map(tuple, model.X))  # each sample is tested exactly once
    for train_X, train_y, test_X, test_y in folds:
        assert len(train_X) + len(test_X) == model.n and len(train_y) == len(train_X) and len(test_y) == len(test_X)
        assert abs(len(test_X) - model.n / q) < 1
        assert not set(map(tuple, train_X)) & set(map(tuple, test_X))
        for x, y in zip(test_X, test_y):  # responses stay with their inputs
            assert y == model.y[np.flatnonzero((model.X == x).all(axis=1))[0]]


def test_norm_and_mse():
    x = np.array([3.0, -1.0, 7.0, 2.0])
    np.testing.assert_allclose(norm(x), (x - x.min()) / np.ptp(x))
    assert mse(3.0, 1.0) == 4.0
    np.testing.assert_array_equal(mse(np.array([1.0, 2.0]), np.array([0.5, 4.0])), [0.25, 4.0])
