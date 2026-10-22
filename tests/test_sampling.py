"""
Sampling plans: random and Morris-Mitchell optimal Latin hypercubes, full factorial grids and space-filling subsets
(Forrester et al. 2007, section 4; Morris & Mitchell 1995; McKay et al. 1979). Exact properties instead of "close
enough": every Latin hypercube column is a permutation of the bin centres, the criteria agree with brute force, and
the optimisers are compared with exhaustive searches.
"""
import itertools
import warnings

import numpy as np
import pytest

import _reference as R
from pyKriging import samplingplan


def is_latin(X, edges=False):
    n = len(X)
    levels = np.arange(n) / (n - 1) if edges else (np.arange(n) + 0.5) / n
    return all(np.allclose(np.sort(X[:, j]), levels) for j in range(X.shape[1]))


def maximin_of_all_latin_hypercubes(n):
    """(d_1, J_1) of the best of all n! two-dimensional n-point Latin hypercubes (the first two entries of the
    Morris-Mitchell key): the largest smallest L1 distance between points and the fewest pairs at that distance."""
    perms = np.array(list(itertools.permutations(range(n))))
    i, j = np.triu_indices(n, 1)
    d = np.abs(i - j) + np.abs(perms[:, i] - perms[:, j])  # L1 distances in rank units
    d1 = d.min(axis=1)
    J1 = (d == d1[:, None]).sum(axis=1)
    best = d1.max()
    return best / n, J1[d1 == best].min()


# ------------------------------------------------------------------ random Latin hypercubes
@pytest.mark.parametrize('n, k', [(10, 2), (20, 5), (17, 3), (2, 3)])
def test_rlh_is_exactly_latin(n, k):
    X = samplingplan(k).rlh(n)
    assert X.shape == (n, k) and np.all((X > 0) & (X < 1)) and is_latin(X)


def test_rlh_with_edges():
    X = samplingplan(3).rlh(9, Edges=1)
    assert is_latin(X, edges=True) and X.min() == 0 and X.max() == 1


def test_random_state_makes_plans_repeatable():
    a, b, c = (samplingplan(2, random_state=s).rlh(10) for s in (7, 7, 8))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    gen = np.random.default_rng(3)
    assert is_latin(samplingplan(2, random_state=gen).rlh(5))  # a Generator is accepted too


def test_without_random_state_plans_follow_numpy_global_seed():
    np.random.seed(11)
    a = samplingplan(2).rlh(10), samplingplan(2).optimallhc(10)
    np.random.seed(11)
    b = samplingplan(2).rlh(10), samplingplan(2).optimallhc(10)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


# ------------------------------------------------------------------ criteria
@pytest.mark.parametrize('q', [1, 2, 5, 10])
@pytest.mark.parametrize('p', [1, 2])
def test_mmphi_matches_brute_force(q, p):
    X = R.lhs(15, 3, 1)
    assert samplingplan(3).mmphi(X, q=q, p=p) == pytest.approx(R.mmphi(X, q=q, p=p), rel=1e-12)


def test_jd_counts_match_brute_force():
    """Latin hypercube distances lie on a 1/n lattice, so ties are the rule: rounding noise must not split them."""
    X = samplingplan(2, random_state=0).rlh(9)
    J, d = samplingplan(2).jd(X, p=1)
    D = np.abs(X[:, None, :] - X[None, :, :]).sum(-1)[np.triu_indices(9, 1)]
    values, counts = np.unique(np.round(D * 9).astype(int), return_counts=True)
    np.testing.assert_allclose(d, values / 9, atol=1e-12)
    np.testing.assert_array_equal(J, counts)
    assert J.sum() == 9 * 8 // 2


def test_jd_small_examples():
    X = np.array([[0, 0], [1, 0], [0, 1]])
    J, d = samplingplan(2).jd(X, p=1)
    np.testing.assert_array_equal(d, [1, 2])
    np.testing.assert_array_equal(J, [2, 1])
    J, d = samplingplan(2).jd(X, p=2)
    np.testing.assert_allclose(d, [1, np.sqrt(2)])
    np.testing.assert_array_equal(J, [2, 1])


def test_mm_compares_plans_by_definition_1_2B():
    sp = samplingplan(2, random_state=0)
    plans = [sp.rlh(10) for _ in range(8)]
    compared = 0
    for a, b in itertools.combinations(range(8), 2):
        ka, kb = R.mm_key(plans[a]), R.mm_key(plans[b])
        if ka != kb:
            assert sp.mm(plans[a], plans[b]) == (1 if ka < kb else 2)
            compared += 1
    assert compared >= 20
    assert sp.mm(plans[0], plans[0][::-1].copy()) == 0  # the same plan in another row order


@pytest.mark.parametrize('jitter', [False, True], ids=['latin', 'jittered'])
def test_mmsort_ranks_plans_best_first(jitter):
    sp = samplingplan(2, random_state=0)
    plans = [sp.rlh(10) for _ in range(6)]
    if jitter:
        rng = np.random.default_rng(0)
        plans = [np.clip(P + 0.01 * rng.standard_normal(P.shape), 0, 1) for P in plans]
    order = list(sp.mmsort(np.stack(plans, axis=2)))
    assert order == sorted(range(6), key=lambda i: R.mm_key(plans[i]))


# ------------------------------------------------------------------ optimal Latin hypercubes
@pytest.mark.parametrize('n, k', [(10, 2), (15, 2), (12, 3), (30, 4)])
def test_optimallhc_is_latin_and_beats_random_plans(n, k):
    sp = samplingplan(k, random_state=n)
    X = sp.optimallhc(n)
    assert X.shape == (n, k) and is_latin(X)
    best_random = min((sp.rlh(n) for _ in range(50)), key=R.mm_key)
    assert R.mm_key(X) < R.mm_key(best_random)
    assert R.mmphi(X, q=5) < R.mmphi(best_random, q=5)


@pytest.mark.parametrize('n', [5, 6, 7, 8])
def test_optimallhc_finds_the_maximin_plan_of_small_problems(n):
    """Exhaustive search over all n! two-dimensional plans gives the Morris-Mitchell optimum (d_1, J_1)."""
    best = maximin_of_all_latin_hypercubes(n)
    found = [R.mm_key(samplingplan(2, random_state=seed).optimallhc(n))[:2] for seed in range(10)]
    assert all(np.isclose(-d1, best[0]) and J1 == best[1] for d1, J1 in found), (best, found)


def test_optimallhc_is_reproducible_given_a_random_state():
    a, b = (samplingplan(3, random_state=5).optimallhc(12) for _ in range(2))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, samplingplan(3, random_state=6).optimallhc(12))


def test_optimallhc_deprecated_arguments_warn_and_change_nothing():
    """optimallhc no longer caches plans or starts worker processes: generation, n_jobs and use_cache are ignored."""
    want = samplingplan(2, random_state=1).optimallhc(10)
    for kwargs in ({'n_jobs': 4}, {'use_cache': True}, {'generation': True}, {'use_cache': False, 'n_jobs': 1}):
        with pytest.warns(DeprecationWarning, match='ignored'):
            got = samplingplan(2, random_state=1).optimallhc(10, **kwargs)
        np.testing.assert_array_equal(got, want)
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # the defaults do not warn
        samplingplan(2, random_state=1).optimallhc(10)


def test_optimallhc_writes_no_files_and_prints_nothing(capsys, tmp_path, monkeypatch):
    """Earlier releases cached plans in the package directory (conftest.py checks that for every test) and printed
    progress; nothing may appear in the working directory either."""
    monkeypatch.chdir(tmp_path)
    samplingplan(2, random_state=0).optimallhc(12)
    assert capsys.readouterr().out == '' and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('population, iterations', [(1, 1), (2, 1), (2, 2), (5, 5)])
def test_optimallhc_with_little_search_effort(population, iterations):
    X = samplingplan(2, random_state=0).optimallhc(6, population=population, iterations=iterations)
    assert is_latin(X)


def test_optimallhc_degenerate_sizes():
    assert is_latin(samplingplan(1, random_state=0).optimallhc(7))  # one variable: any permutation is optimal
    assert samplingplan(3, random_state=0).optimallhc(1).shape == (1, 3)
    assert is_latin(samplingplan(2, random_state=0).optimallhc(12, p=2))  # ranked with Euclidean distances


def test_mmlhs_improves_its_start():
    """The book's evolutionary search (mmlhs.m) keeps the Latin property and does not worsen Phi_q."""
    sp = samplingplan(2, random_state=4)
    X0 = sp.rlh(12)
    X = sp.mmlhs(X0, population=10, iterations=10, q=2)
    assert is_latin(X) and sp.mmphi(X, 2) <= sp.mmphi(X0, 2)


def test_perturb_swaps_entries_within_columns():
    sp = samplingplan(3, random_state=0)
    X = sp.rlh(12)
    X0 = X.copy()
    Y = sp.perturb(X, 7)
    np.testing.assert_array_equal(X, X0)  # the input is not modified
    assert Y.shape == X.shape and is_latin(Y) and not np.array_equal(X, Y)
    assert all(set(X[:, j]) == set(Y[:, j]) for j in range(3))


# ------------------------------------------------------------------ full factorial and subsets
@pytest.mark.parametrize('k, ppd', [(2, 3), (3, 2), (1, 4)])
def test_fullfactorial_is_the_full_grid(k, ppd):
    for edges, levels in ((1, np.linspace(0, 1, ppd)), (0, (np.arange(ppd) + 0.5) / ppd)):
        X = samplingplan(k).fullfactorial(ppd, Edges=edges)
        want = np.array(np.meshgrid(*[levels] * k, indexing='ij')).reshape(k, -1).T
        assert X.shape == (ppd ** k, k)
        assert sorted(map(tuple, np.round(X, 12))) == sorted(map(tuple, np.round(want, 12)))


@pytest.mark.parametrize('n, ns, k', [(12, 4, 2), (16, 6, 2), (12, 4, 3)])
def test_subset_is_the_most_space_filling_subset(n, ns, k):
    """The exchange algorithm (Cook & Nachtsheim 1980) finds the subset with the smallest Phi_q of all."""
    X = samplingplan(k, random_state=1).optimallhc(n)
    Xs, Xr = samplingplan(k, random_state=2).subset(X, ns)
    assert Xs.shape == (ns, k) and Xr.shape == (n - ns, k)
    assert sorted(map(tuple, np.vstack([Xs, Xr]))) == sorted(map(tuple, X))  # a partition of X
    best = min(R.mmphi(X[list(c)], q=5) for c in itertools.combinations(range(n), ns))
    assert R.mmphi(Xs, q=5) == pytest.approx(best, rel=1e-12)


def test_subset_spreads_points_better_than_random_subsets():
    sp = samplingplan(2, random_state=0)
    X = sp.optimallhc(20)
    Xs, _ = sp.subset(X, 6, restarts=3)
    rng = np.random.default_rng(0)
    assert sp.mmphi(Xs, 5, 1) < np.median([sp.mmphi(X[rng.permutation(20)[:6]], 5, 1) for _ in range(50)])


def test_subset_rejects_bad_requests():
    sp = samplingplan(2, random_state=0)
    X = sp.rlh(8)
    for ns in (0, 9):
        with pytest.raises(ValueError):
            sp.subset(X, ns)
    with pytest.raises(ValueError, match='duplicate'):
        sp.subset(np.vstack([X, X[:1]]), 3)
