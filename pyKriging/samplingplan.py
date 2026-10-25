"""
Space-filling sampling plans: random and Morris-Mitchell optimal Latin hypercubes, full factorial grids and
space-filling subsets (Forrester, Sobester & Keane, Engineering Design via Surrogate Modelling, 2008, ch. 1;
Forrester, Sobester & Keane, Proc. R. Soc. A 463:3251, 2007, section 4).
"""
import warnings

import numpy as np
from scipy.spatial.distance import pdist

_QS = (1, 2, 5, 10, 20, 50, 100)  # Phi_q exponents searched by optimallhc (as in the book's bestlh.m)


def _dist(X, p=1):
    X = np.asarray(X, dtype=float)
    return pdist(X, 'cityblock') if p == 1 else pdist(X, 'euclidean') if p == 2 else pdist(X, 'minkowski', p=p)


def _rounded_dist(X, p=1):
    """Pair distances rounded to 32 significant bits (about 10 digits), so that mathematically equal distances
    compare equal, within a plan and between plans, whatever the units of X."""
    mantissa, exponent = np.frexp(_dist(X, p))
    return np.ldexp(np.round(mantissa * 2.0 ** 32) / 2.0 ** 32, exponent)


def _mmkey(X, p=1):
    """Morris-Mitchell definition 1.2B as a sort key (-d1, J1, -d2, J2, ...): smaller is more space-filling."""
    d, J = np.unique(_rounded_dist(X, p), return_counts=True)
    return tuple(np.column_stack([-d, J]).ravel().tolist())


def _phi(d, q):
    """(sum of d^-q)^(1/q) evaluated as (sum (d/d_min)^-q)^(1/q) / d_min, which cannot overflow."""
    dmin = d.min()
    return np.inf if dmin == 0 else np.sum((d / dmin) ** -float(q)) ** (1.0 / q) / dmin


def _swap_search(R0, qs, max_steps, patience, rng, J=50):
    """
    Minimise Phi_q by greedy pair swaps on the Latin hypercube's rank lattice, one chain per starting plan
    R0[c] (ranks, (C, n, k)) and exponent qs[c], all chains vectorised. Each step evaluates J random swaps per chain and applies the best one if it lowers
    Phi_q. Swapping column c of rows i1 and i2 changes only the distances d(i1, .) and d(i2, .), by an exact
    integer amount, so a candidate costs O(n) instead of O(n^2). Stops after max_steps, or once every chain has
    gone `patience` steps without an improvement.
    """
    C, n, k = R0.shape  # one starting plan per chain
    J = min(J, n * (n - 1) // 2)
    R = R0.astype(np.int32)
    D = np.zeros((C, n, n), dtype=np.int32)  # exact L1 distances in rank units
    for c in range(k):
        D += np.abs(R[:, :, None, c] - R[:, None, :, c])
    d0 = D[D > 0].min()
    with np.errstate(divide='ignore', over='ignore'):  # table of (d / d0)^-q; the scaling keeps q = 100 finite
        T = (np.arange(k * n) / d0) ** -np.array(qs, dtype=float)[:, None]
    T[:, 0] = 0.0
    T, off, ci = T.ravel(), (np.arange(C) * k * n)[:, None, None], np.arange(C)[:, None]
    idle, steps = np.zeros(C, dtype=int), 0
    while steps < max_steps and idle.min() < patience:
        steps += 1
        col, i1 = rng.integers(k, size=(C, J)), rng.integers(n, size=(C, J))
        i2 = (i1 + rng.integers(1, n, size=(C, J))) % n
        V = R.transpose(0, 2, 1)[ci, col]  # (C, J, n) the swapped column of each candidate
        a, b = np.take_along_axis(V, i1[..., None], 2), np.take_along_axis(V, i2[..., None], 2)
        dl = np.abs(b - V) - np.abs(a - V)  # change of d(i1, j); d(i2, j) changes by -dl
        np.put_along_axis(dl, i1[..., None], 0, 2)
        np.put_along_axis(dl, i2[..., None], 0, 2)
        D1, D2 = D[ci, i1] + off, D[ci, i2] + off
        dS = (T[D1 + dl] - T[D1] + T[D2 - dl] - T[D2]).sum(2)
        j = dS.argmin(1)
        accept = dS[ci[:, 0], j] < 0
        idle = np.where(accept, 0, idle + 1)
        for c in np.flatnonzero(accept):
            r1, r2, cc, d = i1[c, j[c]], i2[c, j[c]], col[c, j[c]], dl[c, j[c]]
            D[c, r1] += d
            D[c, :, r1] += d
            D[c, r2] -= d
            D[c, :, r2] -= d
            R[c, [r1, r2], cc] = R[c, [r2, r1], cc]
    return R


class samplingplan():
    _rng = None

    def __init__(self, k=2, random_state=None):
        """k: number of dimensions; random_state: seed or numpy Generator (None: follow np.random.seed)."""
        self.k = k
        if random_state is not None:
            self._rng = np.random.default_rng(random_state)

    def _gen(self):
        return self._rng if self._rng is not None else np.random.default_rng(np.random.randint(2 ** 31 - 1))

    def rlh(self, n, Edges=0):
        """Random Latin hypercube of n points in [0, 1]^k (Edges=1: the extreme bins are centred on the edges)."""
        perm = (self._rng or np.random).permutation
        X = np.column_stack([perm(n) for _ in range(self.k)]) + 1.0
        return (X - 1) / (n - 1) if Edges == 1 and n > 1 else (X - 0.5) / n

    def optimallhc(self, n, population=30, iterations=30, generation=None, n_jobs=None, use_cache=None, p=1):
        """
        Morris-Mitchell optimal Latin hypercube of n points (Forrester et al. 2008, bestlh.m): from a random
        Latin hypercube, Phi_q is minimised for each q in (1, 2, 5, 10, 20, 50, 100) and the best plan by
        definition 1.2B is returned. population * iterations sets the search effort. generation, n_jobs and
        use_cache are accepted for backward compatibility and ignored.
        """
        if generation is not None or n_jobs is not None or use_cache is not None:
            warnings.warn('optimallhc() no longer caches plans or uses worker processes: generation, n_jobs and '
                          'use_cache are ignored', DeprecationWarning, stacklevel=2)
        if n < 2 or self.k < 2:
            return self.rlh(n)
        rng = self._gen()
        starts = 32 if n <= 10 else 1  # small plans have many local optima under pair swaps and cost little to restart
        R0 = np.repeat([np.column_stack([rng.permutation(n) for _ in range(self.k)]) for _ in range(starts)], len(_QS), 0)
        qs = np.tile(_QS, starts)
        steps = int(population * iterations / 3 * max(1.0, np.sqrt(n / 50)))
        batch = max(1, int(5e7 // (n * n)))  # chains per batch, bounding the int32 (chains, n, n) arrays
        R = np.concatenate([_swap_search(R0[i:i + batch], qs[i:i + batch], steps, max(20, n // 5), rng)
                            for i in range(0, len(qs), batch)])
        X3D = np.stack(list((R + 0.5) / n), axis=2)
        return X3D[:, :, self.mmsort(X3D, p)[0]]

    def fullfactorial(self, ppd=5, Edges=1):
        """Full factorial grid with ppd points per dimension (Edges=0: bin centres instead of edge to edge)."""
        x = np.linspace(0, 1, ppd) if Edges == 1 else (np.arange(ppd) + 0.5) / ppd
        return np.stack(np.meshgrid(*[x] * self.k, indexing='ij'), -1).reshape(-1, self.k)

    def mmsort(self, X3D, p=1):
        """Indices of the plans X3D[:, :, i] ranked best first by the Morris-Mitchell criterion."""
        keys = [_mmkey(X3D[:, :, i], p) for i in range(X3D.shape[2])]
        return np.array(sorted(range(len(keys)), key=keys.__getitem__))

    def perturb(self, X, PertNum=1):
        """Swap PertNum random pairs of entries within random columns (preserves the Latin property)."""
        X = np.array(X)
        n, k = X.shape
        if n < 2:
            return X
        rand = self._rng.integers if self._rng is not None else np.random.randint
        cols, e1, shift = rand(0, k, size=PertNum), rand(0, n, size=PertNum), rand(1, n, size=PertNum)
        for c, a, b in zip(cols, e1, (e1 + shift) % n):
            X[a, c], X[b, c] = X[b, c], X[a, c]
        return X

    def mmlhs(self, X_start, population, iterations, q=2):
        """The book's evolutionary search (mmlhs.m) for the Latin hypercube minimising Phi_q."""
        X_best = np.array(X_start, dtype=float)
        n, phi_best, leveloff = len(X_best), self.mmphi(X_best, q), int(0.85 * iterations)
        for it in range(1, iterations + 1):
            mutations = round(1 + (0.5 * n - 1) * (leveloff - it) / (leveloff - 1)) if it < leveloff else 1
            offspring = [self.perturb(X_best, mutations) for _ in range(population)]
            phis = [self.mmphi(X, q) for X in offspring]
            if min(phis) < phi_best:
                X_best, phi_best = offspring[int(np.argmin(phis))], min(phis)
        return X_best

    def mmphi(self, X, q=2, p=1):
        """Morris-Mitchell criterion Phi_q = (sum over point pairs of d^-q)^(1/q); smaller is more space-filling."""
        return _phi(_dist(X, p), q)

    def jd(self, X, p=1):
        """Multiplicities J of the distinct pair distances d (ascending, rounded to 1e-10)."""
        d, J = np.unique(_rounded_dist(X, p), return_counts=True)
        return J, d

    def mm(self, X1, X2, p=1):
        """Which plan is more space-filling (definition 1.2B): 0 equal, 1 X1, 2 X2."""
        if np.array_equal(np.unique(X1, axis=0), np.unique(X2, axis=0)):
            return 0
        k1, k2 = _mmkey(X1, p), _mmkey(X2, p)
        m = min(len(k1), len(k2))
        return 0 if k1[:m] == k2[:m] else 1 if k1[:m] < k2[:m] else 2

    def subset(self, X, ns, q=5, p=1, restarts=20, sweeps=10):
        """
        Space-filling ns-point subset of plan X, e.g. the expensive points of a co-kriging plan (Forrester et al.
        2007, section 4): from each of `restarts` random subsets, every subset point is exchanged with the
        remaining point that most reduces Phi_q (Cook & Nachtsheim 1980), sweeping until no exchange helps.
        Returns (Xs, Xr): the subset and the remaining points.
        """
        X = np.asarray(X, dtype=float)
        n = len(X)
        if not 0 < ns <= n:
            raise ValueError('need 0 < ns <= len(X)')
        d = _dist(X, p)
        if d.min() == 0:
            raise ValueError('X contains duplicate points')
        P = np.zeros((n, n))
        P[np.triu_indices(n, 1)] = (d / d.min()) ** -float(q)  # scaled by d_min^q: same ranking, no overflow
        P += P.T
        rng, best = self._gen(), (np.inf, None)
        for _ in range(max(1, restarts)):
            sel = rng.permutation(n)[:ns]
            inS = np.isin(np.arange(n), sel)
            con = P[:, inS].sum(1)  # con[i] = sum over subset points l of d(i, l)^-q
            for _ in range(sweeps):
                changed = False
                for pos, j in enumerate(sel):
                    cand = np.where(inS, np.inf, con - P[:, j])
                    c = int(cand.argmin())
                    if cand[c] < con[j] * (1 - 1e-12):
                        inS[[j, c]] = False, True
                        con += P[:, c] - P[:, j]
                        sel[pos], changed = c, True
                if not changed:
                    break
            S = con[inS].sum()
            if S < best[0]:
                best = (S, inS.copy())
        return X[best[1]], X[~best[1]]
