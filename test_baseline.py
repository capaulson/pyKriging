"""
Baseline testing script for pyKriging GPU acceleration
This script runs simple tests and saves results for comparison after GPU modifications
"""
from __future__ import print_function
import numpy as np
import json
import sys

# Test 1: Simple 2D Kriging with Branin function
def test_simple_kriging():
    print("\n" + "="*80)
    print("TEST 1: Simple 2D Kriging with Branin function")
    print("="*80)

    from pyKriging.krige import kriging
    from pyKriging.samplingplan import samplingplan
    import pyKriging

    # Create sampling plan
    np.random.seed(42)  # For reproducibility
    sp = samplingplan(2)
    X = sp.optimallhc(15)

    # Define test function
    testfun = pyKriging.testfunctions().branin
    y = testfun(X)

    print(f"Training points shape: {X.shape}")
    print(f"Response values shape: {y.shape}")

    # Create and train kriging model
    k = kriging(X, y, testfunction=testfun, name='baseline_test')
    k.train(optimizer='pso')

    # Make predictions at test points
    test_X = np.array([[0.5, 0.5], [0.25, 0.75], [0.75, 0.25]])
    predictions = np.array([k.predict(point) for point in test_X])
    variances = np.array([k.predict_var(point) for point in test_X])

    results = {
        'test_name': 'simple_kriging',
        'theta': k.theta.tolist(),
        'p': k.pl.tolist(),
        'mu': float(k.mu),
        'SigmaSqr': float(k.SigmaSqr),
        'LnDetPsi': float(k.LnDetPsi),
        'test_points': test_X.tolist(),
        'predictions': predictions.tolist(),
        'variances': variances.tolist(),
        'training_X_sample': X[:3].tolist(),
        'training_y_sample': y[:3].tolist()
    }

    print(f"\nTrained hyperparameters:")
    print(f"  theta: {k.theta}")
    print(f"  p: {k.pl}")
    print(f"  mu: {k.mu}")
    print(f"  SigmaSqr: {k.SigmaSqr}")
    print(f"\nPredictions at test points:")
    for i, (point, pred, var) in enumerate(zip(test_X, predictions, variances)):
        print(f"  Point {i+1} {point}: prediction={pred:.6f}, variance={var:.6f}")

    return results


# Test 2: Co-Kriging
def test_cokriging():
    print("\n" + "="*80)
    print("TEST 2: Co-Kriging with cheap and expensive functions")
    print("="*80)

    from pyKriging import coKriging

    def cheap(X):
        A=0.5
        B=10
        C=-5
        D=0
        return A*np.power(((X+D)*6-2), 2)*np.sin(((X+D)*6-2)*2)+((X+D)-0.5)*B+C

    def expensive(X):
        return np.power((X*6-2),2)*np.sin((X*6-2)*2)

    np.random.seed(42)
    Xe = np.array([0, 0.4, 0.6, 1])
    Xc = np.array([0.1,0.2,0.3,0.5,0.7,0.8,0.9,0,0.4,0.6,1])

    yc = cheap(Xc)
    ye = expensive(Xe)

    print(f"Expensive data points: {len(Xe)}")
    print(f"Cheap data points: {len(Xc)}")

    ck = coKriging.coKriging(Xc, yc, Xe, ye)
    ck.thetac = np.array([1.2073])
    ck.updateData()
    ck.updatePsi()
    nll = ck.neglnlikehood()

    # Make predictions
    test_points = np.array([0.15, 0.45, 0.85])
    predictions = np.array([ck.predict(point) for point in test_points])

    results = {
        'test_name': 'cokriging',
        'thetac': ck.thetac.tolist(),
        'rho': float(ck.rho),
        'neglnlikehood': float(nll),
        'test_points': test_points.tolist(),
        'predictions': predictions.tolist()
    }

    print(f"\nCo-Kriging parameters:")
    print(f"  thetac: {ck.thetac}")
    print(f"  rho: {ck.rho}")
    print(f"  Negative log likelihood: {nll}")
    print(f"\nPredictions at test points:")
    for i, (point, pred) in enumerate(zip(test_points, predictions)):
        print(f"  Point {i+1} [{point:.2f}]: prediction={pred:.6f}")

    return results


# Test 3: Matrix operations directly
def test_matrix_operations():
    print("\n" + "="*80)
    print("TEST 3: Core matrix operations")
    print("="*80)

    from pyKriging.matrixops import matrixops

    # Create a simple test case
    np.random.seed(42)
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)

    print(f"Test matrix size: {X.shape}")

    # Create matrixops instance
    mo = matrixops(X, y)
    mo.theta = np.array([1.0, 1.0])
    mo.pl = np.array([2.0, 2.0])
    mo.updateData()
    mo.updatePsi()
    nll = mo.neglikelihood()

    # Test prediction
    test_point = np.array([0.5, 0.5])
    mo.new_points(np.array([test_point]))
    prediction = mo.predict()
    pred_var = mo.predicterr()

    results = {
        'test_name': 'matrix_operations',
        'mu': float(mo.mu),
        'SigmaSqr': float(mo.SigmaSqr),
        'LnDetPsi': float(mo.LnDetPsi),
        'neglikelihood': float(nll),
        'test_point': test_point.tolist(),
        'prediction': float(prediction),
        'pred_variance': float(pred_var),
        'Psi_diagonal_sample': np.diag(mo.Psi)[:5].tolist()
    }

    print(f"\nMatrix operations results:")
    print(f"  mu: {mo.mu}")
    print(f"  SigmaSqr: {mo.SigmaSqr}")
    print(f"  LnDetPsi: {mo.LnDetPsi}")
    print(f"  Negative likelihood: {nll}")
    print(f"  Prediction at {test_point}: {prediction}")
    print(f"  Prediction variance: {pred_var}")

    return results


def main():
    print("="*80)
    print("BASELINE TESTING FOR PYKRIGING GPU ACCELERATION")
    print("="*80)
    print("Running tests to establish baseline results before GPU modifications...")

    all_results = {}

    try:
        results1 = test_simple_kriging()
        all_results['simple_kriging'] = results1
    except Exception as e:
        print(f"ERROR in test_simple_kriging: {e}")
        import traceback
        traceback.print_exc()
        all_results['simple_kriging'] = {'error': str(e)}

    try:
        results2 = test_cokriging()
        all_results['cokriging'] = results2
    except Exception as e:
        print(f"ERROR in test_cokriging: {e}")
        import traceback
        traceback.print_exc()
        all_results['cokriging'] = {'error': str(e)}

    try:
        results3 = test_matrix_operations()
        all_results['matrix_operations'] = results3
    except Exception as e:
        print(f"ERROR in test_matrix_operations: {e}")
        import traceback
        traceback.print_exc()
        all_results['matrix_operations'] = {'error': str(e)}

    # Save results to file
    output_file = 'baseline_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print(f"BASELINE TESTS COMPLETE - Results saved to {output_file}")
    print("="*80)

    return all_results


if __name__ == '__main__':
    main()
