"""Benchmark functions: shapes, known optima and values from the textbook formulas, finiteness on the domain."""
import numpy as np
import pytest

from pyKriging import testfunctions

BATCH = ['linear', 'squared', 'cubed', 'branin', 'paulson', 'paulson1', 'runge', 'stybtang', 'stybtang_norm',
         'curretal88exp', 'cosine']


@pytest.fixture
def tf():
    return testfunctions()


@pytest.mark.parametrize('name', BATCH)
def test_one_point_and_batches(tf, name):
    f = getattr(tf, name)
    X = np.random.default_rng(0).random((7, 2)) * 0.9 + 0.3  # cubed() is real where sum((x - 0.25)^3) >= 0
    assert np.shape(f(X)) == (7,) and np.shape(f(X[0])) == (1,)
    np.testing.assert_allclose(f(X)[3], f(X[3])[0])


@pytest.mark.parametrize('name', [n for n in BATCH if n != 'cubed'])
def test_finite_on_the_unit_square(tf, name):
    X = np.random.default_rng(1).random((200, 2)) * 0.98 + 0.01
    assert np.all(np.isfinite(getattr(tf, name)(X)))


def test_cubed_is_the_documented_real_cube_root(tf):
    """(sum((x - offset)^3))^(1/3), NaN where the sum is negative (documented)."""
    X = np.random.default_rng(2).random((200, 2))
    s = np.sum((X - 0.25) ** 3, axis=1)
    with np.errstate(invalid='ignore'):
        got = tf.cubed(X)
    np.testing.assert_allclose(got[s >= 0], np.cbrt(s[s >= 0]))
    assert np.all(np.isnan(got[s < 0])) and np.any(s < 0)


def test_textbook_values(tf):
    rng = np.random.default_rng(3)
    X = rng.random((20, 2))
    x1, x2 = X[:, 0], X[:, 1]
    np.testing.assert_allclose(tf.linear(X), x1 + x2)
    np.testing.assert_allclose(tf.squared(X), np.hypot(x1 - 0.25, x2 - 0.25))
    np.testing.assert_allclose(tf.paulson(X), 0.5 * np.sin(5 * x1) + 0.5 * np.cos(5 * x2))
    np.testing.assert_allclose(tf.paulson1(X), np.sin(10 * x1) / (x1 + 0.2) + np.cos(10 * x2) / (x2 + 0.2))
    np.testing.assert_allclose(tf.runge(X), 1 / (1 + x1 ** 2 + x2 ** 2))
    np.testing.assert_allclose(tf.cosine(X), np.cos(x1 + x2))
    np.testing.assert_allclose(tf.stybtang_norm(X), tf.stybtang(10 * X - 5))
    currin = (1 - np.exp(-1 / (2 * x2))) * (2300 * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60) \
        / (100 * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20)
    np.testing.assert_allclose(tf.curretal88exp(X), currin)


def test_known_optima(tf):
    assert np.ravel(tf.rosenbrock([1.0] * 10))[0] == 0.0 and np.ravel(tf.rosenbrock([0.0, 0.0]))[0] == 1.0
    assert np.ravel(tf.rastrigin([0.0, 0.0]))[0] == pytest.approx(0.0, abs=1e-12)
    assert np.ravel(tf.rastrigin([1.0, 1.0]))[0] == pytest.approx(2.0)
    assert tf.stybtang(np.full((1, 3), -2.903534))[0] == pytest.approx(-39.16599 * 3, rel=1e-5)
    assert tf.runge(np.array([[0.3, 0.3]]), offset=0.3)[0] == pytest.approx(1.0)
    assert tf.squared(np.array([[0.25, 0.25]]))[0] == pytest.approx(0.0)


def test_branin_is_the_classic_function_plus_5_x1(tf):
    """branin() rescales [0, 1]^2 to [-5, 10] x [0, 15] and adds 5 x_1, which leaves a single global minimum; without
    that term its three minima are the classic 0.397887."""
    xs = np.array([[(-np.pi + 5) / 15, 12.275 / 15], [(np.pi + 5) / 15, 2.275 / 15], [(9.42478 + 5) / 15, 2.475 / 15]])
    np.testing.assert_allclose(tf.branin(xs) - 5 * xs[:, 0], 0.397887, atol=1e-5)
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 301), np.linspace(0, 1, 301)), -1).reshape(-1, 2)
    assert np.linalg.norm(grid[np.argmin(tf.branin(grid))] - xs[0]) < 0.02  # near the first classic minimum
    with pytest.raises(ValueError):
        tf.branin(np.ones((4, 3)))


def test_branin_noise_is_reproducible_and_noisy(tf):
    X = np.random.default_rng(2).random((400, 2))
    np.random.seed(0)
    a = tf.branin_noise(X)
    np.random.seed(0)
    np.testing.assert_array_equal(a, tf.branin_noise(X))
    assert np.std(a - tf.branin(X)) == pytest.approx(15, rel=0.15)
