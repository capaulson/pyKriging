"""
Speed and model quality of pyKriging's main operations.

    python benchmarks/bench.py                         # every scenario on the CPU (under a minute)
    python benchmarks/bench.py --quick                 # smaller problems, fewer repeats (seconds)
    python benchmarks/bench.py --only train,predict --out results.json
    python benchmarks/bench.py --device cuda           # CUDA likelihood for n >= 300 (PyTorch with CUDA)

It measures whichever pyKriging Python imports: install the checkout (pip install -e .) or point PYTHONPATH at
the tree to measure; the tree is recorded in the results. Scenarios: import, sampling, likelihood, predict,
train, infill, workflow, regression, cokriging, crossvalidation.

Times are medians over repeats, in seconds unless the key says otherwise (_ms, _us, _us_per_pt). Model quality
is measured independently of pyKriging: ref_nll is the concentrated negative log-likelihood of the trained theta
and p, computed here in float64 (lower is better), and rel_rmse is the RMSE against the true function on test
points divided by the function's range. Each scenario prints a summary line when it finishes; --out saves
everything as JSON (rewritten after every scenario).
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
import warnings

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import pdist

EPS = np.finfo(float).eps


# ----------------------------------------------------------------------------------------------- helpers
def lhs(n, k, seed):
    """Latin hypercube from NumPy alone, so the samples do not depend on the pyKriging being measured."""
    rng = np.random.default_rng(seed)
    return (np.argsort(rng.random((n, k)), axis=0) + rng.random((n, k))) / n


def f_sine(X):
    X = np.atleast_2d(X)
    return np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])


def f_branin(X):
    X = np.atleast_2d(X)
    x1, x2 = 15 * X[:, 0] - 5, 15 * X[:, 1]
    return (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10


def f_sum5(X):
    return np.sum(np.sin(np.pi * np.atleast_2d(X)), axis=1)


FUNCS = {'sine': f_sine, 'branin': f_branin, 'sum5': f_sum5}


def ref_nll(X, y, theta, p):
    """Concentrated -ln L of (theta, p) on min-max normalised data with an eps nugget (Forrester et al. 2007,
    eqs 2.3, 3.4-3.6), written independently of pyKriging."""
    Xn, yn = (X - X.min(0)) / np.ptp(X, axis=0), (y - y.min()) / np.ptp(y)
    n = len(yn)
    Psi = np.exp(-np.sum(theta * np.abs(Xn[:, None, :] - Xn[None, :, :]) ** p, axis=2)) + np.eye(n) * EPS
    try:
        L = np.linalg.cholesky(Psi)
    except np.linalg.LinAlgError:
        return float('inf')
    a, b = solve_triangular(L, np.ones(n), lower=True), solve_triangular(L, yn, lower=True)
    r = b - (a @ b) / (a @ a) * a
    return float(n / 2 * np.log(r @ r / n) + np.sum(np.log(np.diag(L))))


def rel_rmse(pred, truth):
    return float(np.sqrt(np.mean((np.asarray(pred) - truth) ** 2)) / np.ptp(truth))


def phi2(X):
    """Morris-Mitchell space-filling criterion Phi_2 with rectangular distances (lower is better)."""
    return float(np.sum(pdist(X, 'cityblock') ** -2.0) ** 0.5)


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return time.perf_counter() - t0, out


def median(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return statistics.median(values) if values else None


# ----------------------------------------------------------------------------------------------- scenarios
def bench_import(args):
    """Wall time of `import pyKriging` in a fresh interpreter (same tree as this run)."""
    import pyKriging
    root = os.path.dirname(os.path.dirname(os.path.abspath(pyKriging.__file__)))
    code = f'import sys, time; sys.path.insert(0, {root!r}); t = time.perf_counter(); import pyKriging; ' \
           'print(time.perf_counter() - t)'
    runs = [subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
            for _ in range(3 if args.quick else 5)]
    return {'import_s': median(float(r.stdout.split()[-1]) for r in runs)}


def bench_sampling(args):
    """Optimal Latin hypercubes (time and Phi_2), random Latin hypercubes and space-filling subsets."""
    from pyKriging import samplingplan
    res = {}
    for n in ([20] if args.quick else [20, 50]):
        runs = [timed(samplingplan(k=2, random_state=r).optimallhc, n) for r in range(3)]
        res[f'optimallhc_n{n}_k2'] = {'time_s': median(t for t, _ in runs), 'phi2': median(phi2(X) for _, X in runs)}
    sp = samplingplan(k=5, random_state=0)
    res['rlh_n1000_k5_us'] = median(timed(sp.rlh, 1000)[0] * 1e6 for _ in range(20))
    X = samplingplan(k=2, random_state=0).rlh(100)
    res['subset_100to20_s'] = median(timed(samplingplan(k=2, random_state=r).subset, X, 20)[0] for r in range(3))
    return res


def bench_likelihood(args):
    """One refit at fixed hyper-parameters (update) and one optimiser generation of 300 candidates."""
    from pyKriging import kriging
    res = {}
    candidates = np.column_stack([10 ** np.random.default_rng(1).uniform(-2, 2, (300, 2)),
                                  np.random.default_rng(2).uniform(1, 2, (300, 2))])
    for n in ([20, 100] if args.quick else [20, 100, 300, 500]):
        X = lhs(n, 2, 7)
        m = kriging(X, f_sine(X))
        reps = max(20, 4000 // n)
        t, _ = timed(lambda: [m.update([3.0 + 0.001 * i, 5.0, 1.9, 2.0]) for i in range(reps)])
        res[f'nll_eval_n{n}_us'] = t / reps * 1e6
        t, _ = timed(lambda: [m.fittingObjective(candidates) for _ in range(3)])
        res[f'generation300_n{n}_ms'] = t / 3 * 1e3
    return res


def bench_predict(args):
    """Per-point and batch prediction, standard error and expected improvement at fixed hyper-parameters."""
    from pyKriging import kriging
    res = {}
    P = np.random.default_rng(5).random((300, 2))
    batch = np.random.default_rng(6).random((10000, 2))
    for n in ([20, 100] if args.quick else [20, 100, 500]):
        X = lhs(n, 2, 3)
        m = kriging(X, f_sine(X))
        m.update([10.0, 10.0, 1.9, 1.9])
        m.predict(P[0]), m.predicterr(P[0])  # warm-up
        res[f'predict_n{n}_us_per_pt'] = timed(lambda: [m.predict(x) for x in P])[0] / len(P) * 1e6
        res[f'predicterr_n{n}_us_per_pt'] = timed(lambda: [m.predicterr(x) for x in P])[0] / len(P) * 1e6
        res[f'infill_ei_n{n}_us_per_pt'] = timed(lambda: [m.infill_ei(x) for x in P[:100]])[0] / 100 * 1e6
        res[f'predict_batch10k_n{n}_us_per_pt'] = timed(m.predict, batch)[0] / len(batch) * 1e6
        res[f'predicterr_batch10k_n{n}_us_per_pt'] = timed(m.predicterr, batch)[0] / len(batch) * 1e6
    return res


def bench_train(args):
    """Maximum-likelihood training: time, likelihood reached (ref_nll) and test error, over seeded repeats."""
    from pyKriging import kriging
    cases = [('sine', 20, 2, 'ga'), ('sine', 20, 2, 'pso'), ('branin', 50, 2, 'pso'), ('sine', 100, 2, 'pso'),
             ('sum5', 60, 5, 'pso')]
    res = {}
    for fname, n, k, optimizer in cases[:3] if args.quick else cases:
        f = FUNCS[fname]
        Xt = np.random.default_rng(99).random((2000, k))
        times, nlls, errors = [], [], []
        for r in range(3 if args.quick or n >= 100 else 5):
            X = lhs(n, k, 100 + r)
            y = f(X)
            m = kriging(X, y, seed=100 + r)
            times.append(timed(m.train, optimizer=optimizer)[0])
            nlls.append(ref_nll(X, y, m.theta, m.pl))
            errors.append(rel_rmse(m.predict(Xt), f(Xt)))
        res[f'train_{fname}_n{n}_k{k}_{optimizer}'] = {'time_s': median(times), 'ref_nll': median(nlls),
                                                       'rel_rmse': median(errors), 'times': times, 'nlls': nlls}
    return res


def bench_infill(args):
    """Choosing new sample points by prediction error and by expected improvement."""
    from pyKriging import kriging
    X = lhs(20, 2, 11)
    m = kriging(X, f_branin(X), seed=11)
    m.train(optimizer='ga')
    return {'infill_error_s': timed(m.infill, 1, method='error')[0],
            'infill_ei_s': timed(m.infill, 1, method='ei')[0],
            'infill_ei_5pts_s': timed(m.infill, 5, method='ei')[0]}


def bench_workflow(args):
    """examples/2D_simple_train.py without plots: optimal LHC -> train -> 3 x (infill 2, add, retrain)."""
    from pyKriging import kriging, samplingplan
    Xt = np.random.default_rng(99).random((1000, 2))
    times, errors = [], []
    for r in range(1 if args.quick else 3):
        t0 = time.perf_counter()
        X = samplingplan(k=2, random_state=200 + r).optimallhc(15)
        m = kriging(X, f_branin(X), seed=200 + r)
        m.train(optimizer='ga')
        for _ in range(3):
            for point in m.infill(2):
                m.addPoint(point, f_branin(point)[0])
            m.train()
        times.append(time.perf_counter() - t0)
        errors.append(rel_rmse(m.predict(Xt), f_branin(Xt)))
    return {'workflow_s': median(times), 'workflow_rel_rmse': median(errors)}


def bench_regression(args):
    """Regression kriging of the Branin function plus N(0, 15^2) noise: time and error against the noise-free
    function."""
    from pyKriging import regression_kriging
    Xt = np.random.default_rng(99).random((1000, 2))
    times, errors, lambdas = [], [], []
    for r in range(1 if args.quick else 3):
        X = lhs(30, 2, 300 + r)
        m = regression_kriging(X, f_branin(X) + np.random.default_rng(r).normal(0, 15, len(X)), seed=300 + r)
        times.append(timed(m.train, optimizer='pso')[0])
        errors.append(rel_rmse(m.predict(Xt), f_branin(Xt)))
        lambdas.append(m.Lambda)
    return {'regression_train_s': median(times), 'regression_rel_rmse_vs_truth': median(errors),
            'regression_Lambda': median(lambdas)}


def bench_cokriging(args):
    """The one-variable example of Forrester, Sobester & Keane (2007), section 3a: 11 cheap and 4 expensive
    samples. rho should be close to 1/A = 2; rel_rmse_kriging is ordinary kriging through the expensive data."""
    from pyKriging import kriging
    from pyKriging.coKriging import coKriging

    def fe(x):
        return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

    def fc(x):
        return 0.5 * fe(x) + 10 * (x - 0.5) + 5

    Xc, Xe, xt = np.linspace(0, 1, 11), np.array([0, 0.4, 0.6, 1]), np.linspace(0, 1, 201)
    times, errors, rhos = [], [], []
    for r in range(1 if args.quick else 3):
        ck = coKriging(Xc, fc(Xc), Xe, fe(Xe), seed=400 + r)
        times.append(timed(ck.train)[0])
        errors.append(rel_rmse(ck.predict(xt), fe(xt)))
        rhos.append(ck.rho)
    k = kriging(Xe, fe(Xe), seed=400)
    k.train()
    return {'cokriging_train_s': median(times), 'cokriging_rel_rmse': median(errors), 'cokriging_rho': median(rhos),
            'rel_rmse_kriging': rel_rmse(k.predict(xt), fe(xt))}


def bench_crossvalidation(args):
    """Closed-form leave-one-out residuals (SCVR) and 5-fold cross-validation, which retrains five times."""
    from pyKriging import kriging
    from pyKriging.CrossValidation import Cross_Validation
    X = lhs(30, 2, 500)
    m = kriging(X, f_sine(X), seed=500)
    m.train()
    t, (_, _, scvr) = timed(Cross_Validation(m).calculate_SCVR)
    np.random.seed(500)  # leave_n_out splits the samples with np.random
    return {'scvr_s': t, 'scvr_max_abs': float(np.max(np.abs(scvr))),
            'leave_n_out_q5_s': timed(Cross_Validation(m).leave_n_out, 5)[0]}


SCENARIOS = {'import': bench_import, 'sampling': bench_sampling, 'likelihood': bench_likelihood,
             'predict': bench_predict, 'train': bench_train, 'infill': bench_infill, 'workflow': bench_workflow,
             'regression': bench_regression, 'cokriging': bench_cokriging, 'crossvalidation': bench_crossvalidation}


# ----------------------------------------------------------------------------------------------- driver
def summary(result):
    """One-line rendering of a scenario's result: scalars only, 4 significant digits."""
    items = []
    for key, value in result.items():
        if isinstance(value, dict):
            items.append(f'{key}=({summary(value)})')
        elif isinstance(value, float):
            items.append(f'{key}={value:.4g}')
        elif not isinstance(value, list):
            items.append(f'{key}={value}')
    return ' '.join(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', help='save the results to this JSON file')
    parser.add_argument('--quick', action='store_true', help='smaller problems and fewer repeats')
    parser.add_argument('--only', default='', help='comma-separated scenarios to run (default: all)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'auto'],
                        help="likelihood backend for problems with n >= 300 (default: cpu; 'auto': CUDA if present)")
    args = parser.parse_args()
    names = [s for s in args.only.split(',') if s] or list(SCENARIOS)
    unknown = [s for s in names if s not in SCENARIOS]
    if unknown:
        parser.error(f'unknown scenario(s) {", ".join(unknown)}; choose from {", ".join(SCENARIOS)}')

    import scipy
    import pyKriging
    try:
        pyKriging.configure_gpu(args.device)
    except RuntimeError as err:
        parser.error(f'--device {args.device}: {err}')
    warnings.simplefilter('ignore')  # optimiser and plotting warnings would only clutter the table

    out = {'meta': {'tree': os.path.dirname(os.path.dirname(os.path.abspath(pyKriging.__file__))),
                    'pyKriging': pyKriging.__version__, 'device': pyKriging.get_device_info(), 'quick': args.quick,
                    'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                    'machine': platform.machine(), 'processor': platform.processor(), 'cpus': os.cpu_count(),
                    'date': time.strftime('%Y-%m-%d %H:%M:%S')}}
    print(f"pyKriging {pyKriging.__version__} from {out['meta']['tree']} on {out['meta']['device']['backend']}",
          flush=True)
    failed = False
    for name in names:
        t0 = time.perf_counter()
        try:
            out[name] = SCENARIOS[name](args)
            line = summary(out[name])
        except Exception as err:
            failed = True
            out[name] = {'error': f'{type(err).__name__}: {err}', 'traceback': traceback.format_exc()}
            line = 'ERROR ' + out[name]['error']
        print(f'{name:15s} {time.perf_counter() - t0:6.1f}s  {line}', flush=True)
        if args.out:
            with open(args.out, 'w') as fh:
                json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
