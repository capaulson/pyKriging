"""
Training = maximum-likelihood estimation of theta and p. The trained hyper-parameters are scored with the reference
concentrated likelihood (Forrester et al. 2007, eq. 3.6) and compared with a dense reference optimisation over the
model's own search box; "the hyper-parameters changed" would be satisfied by a random draw.
"""
import functools

import numpy as np
import pytest

import _reference as R
from pyKriging import kriging
from pyKriging.matrixops import PENALTY

NLL_TOL = 0.05  # nats: correctly trained models land within 1e-3 of the reference optimum

PROBLEMS = {
    'n12_k2': (12, 2, 11, lambda X: np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1]) + 0.3 * X[:, 0]),
    'n8_k1': (8, 1, 3, lambda X: (6 * X[:, 0] - 2) ** 2 * np.sin(12 * X[:, 0] - 4)),
    'n20_k3': (20, 3, 5, lambda X: np.sin(4 * X[:, 0]) + X[:, 1] ** 2 + 0.1 * X[:, 2]),  # x_3 nearly inactive
    'n30_k2': (30, 2, 100, lambda X: np.abs(np.sin(5 * X[:, 0])) + 0.5 * np.abs(np.cos(4 * X[:, 1]))),  # kinks: p < 2
    # on a maximin plan; the optimum is nearly isotropic (theta ~ 0.016 for all j, p just below 2) and was missed by
    # 0.2.0 and by searches over the full box alone, which stop at -60.60 with x_3 and x_4 "inactive"
    'n40_k5': (40, 5, None, lambda X: np.sum(np.sin(np.pi * X), axis=1) + X[:, 0] * X[:, 1]),
}


def problem(name):
    n, k, seed, f = PROBLEMS[name]
    X = R.MAXIMIN_LHS_40x5 if seed is None else R.lhs(n, k, seed)
    return X, f(X)


@functools.lru_cache(maxsize=None)
def reference_optimum(name, box=(1e-5, 100.0, 1.0, 2.0)):
    """(NegLnLike, theta, p) of the dense reference search over kriging's default box."""
    Xn, yn, _, _ = R.normalise(*problem(name))
    return R.nll_optimum(Xn, yn, box[:2], box[2:])


def reference_nll(name, model):
    Xn, yn, _, _ = R.normalise(*problem(name))
    return R.concentrated_nll(Xn, yn, model.theta, model.pl)


@pytest.mark.parametrize('optimizer', ['pso', 'ga'])
@pytest.mark.parametrize('name', ['n12_k2', 'n8_k1', 'n20_k3', 'n30_k2', 'n40_k5'])
def test_trained_likelihood_is_the_reference_optimum(name, optimizer):
    model = kriging(*problem(name), seed=0)
    model.train(optimizer=optimizer)
    best = reference_optimum(name)
    got = reference_nll(name, model)
    assert got <= best[0] + NLL_TOL, (f'trained NegLnLike {got:.4f} vs reference optimum {best[0]:.4f} '
                                      f'(theta={model.theta}, p={model.pl}; reference theta={best[1]}, p={best[2]})')


def test_training_improves_on_the_default_hyperparameters():
    X, y = problem('n12_k2')
    model = kriging(X, y, seed=1)
    model.updateModel()  # theta = 1, p = 2
    default = model.NegLnLike
    model.train()
    assert model.NegLnLike < default - 1.0
    assert not np.allclose(np.r_[model.theta, model.pl], [1.0, 1.0, 2.0, 2.0])


def test_train_leaves_the_model_fitted_at_the_returned_hyperparameters():
    X, y = problem('n12_k2')
    model = kriging(X, y, seed=4)
    model.train(optimizer='ga')
    Xn, yn, _, _ = R.normalise(X, y)
    ref = R.RefKriging(Xn, yn, model.theta, model.pl)
    assert (model.mu, model.SigmaSqr, model.NegLnLike) == pytest.approx((ref.mu, ref.sigma2, ref.nll), rel=1e-8)
    Q = R.lhs(20, 2, 9)
    np.testing.assert_allclose(model.predict_normalized(Q), ref.predict(Q), rtol=1e-8, atol=1e-10)


def test_hyperparameters_depend_on_the_data():
    X = R.lhs(14, 2, 5)
    trained = []
    for f in (lambda X: np.sin(2 * np.pi * X[:, 0]) + 0.1 * X[:, 1], lambda X: np.cos(5 * X[:, 1]) * X[:, 0]):
        model = kriging(X, f(X), seed=0)
        model.train()
        trained.append(np.r_[model.theta, model.pl])
    assert not np.allclose(trained[0], trained[1], rtol=1e-3)
    assert trained[0][0] > trained[0][1]  # the first data vary along x_1, the second mostly along x_2
    assert trained[1][1] > trained[1][0]


def test_trained_hyperparameters_respect_the_bounds():
    X, y = problem('n20_k3')
    model = kriging(X, y, seed=3)
    model.train()
    assert np.all((model.theta >= model.thetamin) & (model.theta <= model.thetamax))
    assert np.all((model.pl >= model.pmin) & (model.pl <= model.pmax))
    model = kriging(X, y, seed=3)
    model.thetamin, model.thetamax, model.pmin, model.pmax = 0.5, 20.0, 1.5, 1.9  # user-set bounds are used
    model.train()
    assert np.all((model.theta >= 0.5 * (1 - 1e-12)) & (model.theta <= 20.0 * (1 + 1e-12)))
    assert np.all((model.pl >= 1.5 - 1e-12) & (model.pl <= 1.9 + 1e-12))


def test_fitting_objective_is_the_concentrated_likelihood():
    """The batch objective that train() minimises is NegLnLike, with PENALTY where Psi is not positive definite."""
    X, y = problem('n12_k2')
    model = kriging(X, y)
    Xn, yn, _, _ = R.normalise(X, y)
    rng = np.random.default_rng(0)
    candidates = np.c_[10 ** rng.uniform(-1, 2, (30, 2)), rng.uniform(1, 2, (30, 2))]
    candidates[::7, :2] = -50.0  # correlations > 1: Psi is not positive definite
    got = model.fittingObjective(candidates)
    want = [R.concentrated_nll(Xn, yn, c[:2], c[2:]) for c in candidates]
    assert got.shape == (30,) and np.all(got[::7] == PENALTY) and np.all(np.isinf(want[::7]))
    keep = np.arange(30) % 7 > 0
    np.testing.assert_allclose(got[keep], np.array(want)[keep], rtol=1e-9)
    assert model.fittingObjective_local(candidates[1]) == pytest.approx(want[1], rel=1e-9)


def test_batched_and_single_fits_agree_exactly():
    """The population objective and update() share one Cholesky routine: a candidate gets bit-for-bit the
    likelihood its fit has, and near-singular candidates fail in both or in neither."""
    X, y = problem('n12_k2')
    model = kriging(X, y)
    rng = np.random.default_rng(1)
    p = np.where(rng.random((60, 2)) < 0.5, 2.0, rng.uniform(1, 2, (60, 2)))
    candidates = np.c_[10 ** rng.uniform(-12, 2, (60, 2)), p]  # theta < 1e-8 with p = 2: Psi numerically singular
    batch = model.fittingObjective(candidates)
    assert 0 < np.sum(batch == PENALTY) < 30  # both outcomes occur
    for c, value in zip(candidates, batch):
        try:
            model.update(c)
        except Exception:
            assert value == PENALTY
        else:
            assert value == model.NegLnLike


# ------------------------------------------------------------------ reproducibility
@pytest.mark.parametrize('optimizer', ['pso', 'ga'])
def test_training_is_reproducible_given_a_seed(optimizer):
    X, y = problem('n12_k2')
    runs = []
    for _ in range(2):
        model = kriging(X, y, seed=2024)
        model.train(optimizer=optimizer)
        runs.append(np.r_[model.theta, model.pl, model.NegLnLike])
    np.testing.assert_array_equal(runs[0], runs[1])


def test_a_numpy_generator_is_accepted_as_seed():
    X, y = problem('n12_k2')
    a, b = kriging(X, y, seed=np.random.default_rng(99)), kriging(X, y, seed=np.random.default_rng(99))
    a.train(optimizer='ga')
    b.train(optimizer='ga')
    np.testing.assert_array_equal(np.r_[a.theta, a.pl], np.r_[b.theta, b.pl])


def test_training_does_not_touch_numpy_global_random_state():
    """Users seed np.random for their own sampling plans: neither seeded nor unseeded training may consume it."""
    X, y = problem('n12_k2')
    np.random.seed(5)
    expected = np.random.random(3)
    for seed in (None, 7):
        np.random.seed(5)
        kriging(X, y, seed=seed).train()
        np.testing.assert_array_equal(np.random.random(3), expected)


# ------------------------------------------------------------------ special cases
def test_unknown_optimizer_is_rejected():
    X, y = problem('n12_k2')
    with pytest.raises(ValueError, match='pso'):
        kriging(X, y).train(optimizer='bogus')


def test_an_optimizer_name_built_at_run_time_is_recognised():
    """train() must compare names by value: a string built at run time is not the interned literal 'ga'."""
    X, y = problem('n12_k2')
    model = kriging(X, y, seed=0)
    model.train(optimizer=''.join(['g', 'a']))
    assert model.NegLnLike <= reference_optimum('n12_k2')[0] + NLL_TOL


def test_constant_responses_give_a_constant_model():
    X = R.lhs(10, 2, 2)
    model = kriging(X, np.full(10, 5.0), seed=0)
    assert np.all(np.isfinite(model.y))
    model.train()
    Q = R.lhs(25, 2, 3)
    np.testing.assert_allclose(model.predict(Q), 5.0, rtol=1e-12)
    np.testing.assert_allclose(model.predicterr(Q), 0.0, atol=1e-12)
    assert np.all(model.infill_ei(Q) == 0)


def test_minimal_designs():
    model = kriging([0.0, 1.0], [1.0, 3.0], seed=0)  # two points in one dimension
    model.train()
    assert model.predict(0.0) == pytest.approx(1.0, abs=1e-6) and model.predict(1.0) == pytest.approx(3.0, abs=1e-6)
    assert np.isfinite(model.predict(0.5)) and model.predicterr(0.5) > 0
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # three points in two dimensions
    model = kriging(X, X.sum(axis=1), seed=0)
    model.train(optimizer='ga')
    np.testing.assert_allclose(model.predict(X), X.sum(axis=1), atol=1e-6)
    assert np.isfinite(model.predict([0.5, 0.5])) and np.isfinite(model.NegLnLike)
