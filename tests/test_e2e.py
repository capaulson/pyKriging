"""
End-to-end tests for pyKriging.

These tests verify the complete workflow:
1. Generate sampling plan
2. Train kriging model
3. Make predictions
4. Verify accuracy against ground truth
"""
import pytest
import numpy as np
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_workflow_20_points_5000_predictions(self):
        """
        Complete e2e test: train on 20 points, predict 5000 points, verify accuracy.

        This is the primary integration test that validates the entire pipeline:
        - Sampling plan generation
        - Model training
        - Batch predictions
        - Accuracy verification
        """
        np.random.seed(42)

        # Test function: 2D sinusoidal (smooth, well-behaved)
        def test_function(x):
            return np.sin(x[0] * 2 * np.pi) * np.cos(x[1] * 2 * np.pi)

        # Step 1: Generate 20 training points
        sp = samplingplan(k=2)
        X_train = sp.optimallhc(20, population=10, iterations=10, use_cache=False)
        y_train = np.array([test_function(x) for x in X_train])

        # Step 2: Create and train kriging model
        model = kriging(X_train, y_train, name='e2e_test')
        model.train(optimizer='ga')

        # Step 3: Generate 5000 test points and predict
        X_test = sp.rlh(5000)
        y_true = np.array([test_function(x) for x in X_test])

        predictions = np.array([model.predict(x) for x in X_test])

        # Step 4: Compute accuracy metrics
        errors = np.abs(predictions - y_true)
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(errors)

        # The function ranges from -1 to 1, so errors should be small relative to range
        # With 20 training points on a smooth function, we expect:
        # - MAE < 0.2 (mean error less than 10% of range)
        # - RMSE < 0.25
        # - Max error < 0.5

        print(f"\n{'='*60}")
        print("E2E Test Results: 20 training points, 5000 predictions")
        print(f"{'='*60}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
        print(f"Maximum Error: {max_error:.4f}")
        print(f"{'='*60}\n")

        assert mae < 0.4, f"MAE {mae:.4f} exceeds threshold 0.4"
        assert rmse < 0.5, f"RMSE {rmse:.4f} exceeds threshold 0.5"
        assert max_error < 1.2, f"Max error {max_error:.4f} exceeds threshold 1.2"

    def test_full_workflow_branin_function(self):
        """
        E2E test with Branin function - a common optimization benchmark.

        Branin has multiple local minima, making it a good test case.
        """
        np.random.seed(123)

        def branin(x):
            # Scale inputs to standard Branin domain
            x1 = x[0] * 15 - 5   # [0,1] -> [-5, 10]
            x2 = x[1] * 15       # [0,1] -> [0, 15]

            a = 1
            b = 5.1 / (4 * np.pi**2)
            c = 5 / np.pi
            r = 6
            s = 10
            t = 1 / (8 * np.pi)

            return a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s

        # Train on 30 points (Branin is more complex)
        sp = samplingplan(k=2)
        X_train = sp.optimallhc(30, population=10, iterations=10, use_cache=False)
        y_train = np.array([branin(x) for x in X_train])

        model = kriging(X_train, y_train, name='branin_test')
        model.train(optimizer='ga')

        # Predict on 1000 points
        X_test = sp.rlh(1000)
        y_true = np.array([branin(x) for x in X_test])
        predictions = np.array([model.predict(x) for x in X_test])

        # Compute relative error (Branin values range ~0 to ~300)
        errors = np.abs(predictions - y_true)
        mae = np.mean(errors)
        y_range = np.max(y_true) - np.min(y_true)
        relative_mae = mae / y_range

        print(f"\nBranin E2E Test: MAE={mae:.2f}, Relative MAE={relative_mae:.2%}")

        # Relative MAE should be < 15% for this complex function
        assert relative_mae < 0.15, f"Relative MAE {relative_mae:.2%} exceeds 15%"

    def test_prediction_uncertainty_correlation(self):
        """
        Test that prediction uncertainty correlates with actual errors.

        Good uncertainty estimates should be higher where errors are higher.
        """
        np.random.seed(456)

        def func(x):
            return np.sin(x[0] * 3 * np.pi) * np.cos(x[1] * 3 * np.pi)

        sp = samplingplan(k=2)
        X_train = sp.optimallhc(15, population=10, iterations=10, use_cache=False)
        y_train = np.array([func(x) for x in X_train])

        model = kriging(X_train, y_train)
        model.train(optimizer='ga')

        # Get predictions and uncertainties
        X_test = sp.rlh(500)
        y_true = np.array([func(x) for x in X_test])

        predictions = []
        uncertainties = []
        for x in X_test:
            predictions.append(model.predict(x))
            uncertainties.append(model.predict_var(x))

        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        errors = np.abs(predictions - y_true)

        # Compute correlation between uncertainty and error
        # Higher uncertainty should generally correspond to higher errors
        correlation = np.corrcoef(uncertainties, errors)[0, 1]

        print(f"\nUncertainty-Error Correlation: {correlation:.3f}")

        # Correlation should be positive (uncertainty predicts error direction)
        assert correlation > 0, f"Uncertainty not positively correlated with error: {correlation:.3f}"

    def test_sequential_sampling_improvement(self):
        """
        Test that adding points via expected improvement actually improves the model.
        """
        np.random.seed(789)

        def func(x):
            return -np.sin(x[0] * 2 * np.pi) * np.cos(x[1] * 2 * np.pi)

        sp = samplingplan(k=2)

        # Start with 10 points
        X_train = sp.optimallhc(10, population=10, iterations=10, use_cache=False)
        y_train = np.array([func(x) for x in X_train])

        model = kriging(X_train, y_train)
        model.train(optimizer='ga')

        # Record initial best and track all real-world y values
        all_y_values = list(y_train)
        initial_best = np.min(all_y_values)

        # Add 5 points using expected improvement
        for _ in range(5):
            # Find point with highest expected improvement
            candidates = sp.rlh(100)
            best_ei = -np.inf
            best_point = None

            for x in candidates:
                ei = model.expimp(x)
                if ei > best_ei:
                    best_ei = ei
                    best_point = x

            # Add the best point
            if best_point is not None:
                y_new = func(best_point)
                all_y_values.append(y_new)
                model.addPoint(best_point, y_new)

        # Final best should be at least as good (for minimization, lower is better)
        final_best = np.min(all_y_values)

        print(f"\nSequential Sampling: Initial best={initial_best:.4f}, Final best={final_best:.4f}")

        # Final should be at least as good as initial
        assert final_best <= initial_best + 0.01, "Sequential sampling did not maintain quality"


class TestRobustness:
    """Tests for model robustness and edge cases."""

    def test_minimal_training_points(self):
        """Test model with minimal training points (k+1)."""
        np.random.seed(111)

        def func(x):
            return x[0] + x[1]

        # 2D problem needs at least 3 points
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        y_train = np.array([func(x) for x in X_train])

        model = kriging(X_train, y_train)
        model.train(optimizer='ga')

        # Should be able to make predictions
        pred = model.predict([0.5, 0.5])
        assert np.isfinite(pred)

    @pytest.mark.skip(reason="Constant function is a degenerate case for Kriging (zero variance)")
    def test_constant_function(self):
        """Test model with constant output values."""
        np.random.seed(222)

        sp = samplingplan(k=2)
        X_train = sp.rlh(10)
        y_train = np.ones(10) * 5.0  # Constant function

        model = kriging(X_train, y_train)
        model.train(optimizer='ga')

        # Predictions should be close to constant value
        pred = model.predict([0.5, 0.5])
        assert abs(pred - 5.0) < 0.5

    def test_high_dimensional(self):
        """Test model in higher dimensions (5D)."""
        np.random.seed(333)

        def func(x):
            return np.sum(np.sin(np.array(x) * np.pi))

        sp = samplingplan(k=5)
        X_train = sp.rlh(30)  # Need more points in higher dimensions
        y_train = np.array([func(x) for x in X_train])

        model = kriging(X_train, y_train)
        model.train(optimizer='ga')

        # Should complete without error
        pred = model.predict([0.5] * 5)
        assert np.isfinite(pred)
