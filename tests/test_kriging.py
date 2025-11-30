"""
Unit tests for pyKriging kriging model.
"""
import pytest
import numpy as np
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan


class TestKrigingInitialization:
    """Tests for kriging model initialization."""

    def test_init_basic(self, seed, simple_2d_function):
        """Test basic kriging initialization."""
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        y = np.array([simple_2d_function(x) for x in X])

        k = kriging(X, y, name='test')

        assert k.n == 10
        assert k.k == 2
        assert k.name == 'test'

    def test_init_shapes(self, seed, simple_2d_function):
        """Test that internal arrays have correct shapes."""
        sp = samplingplan(k=2)
        X = sp.rlh(15)
        y = np.array([simple_2d_function(x) for x in X])

        k = kriging(X, y)

        # Check normalized data shapes
        assert k.X.shape[0] == 15
        assert k.X.shape[1] == 2
        assert len(k.y) == 15

    def test_init_hyperparameters(self, seed, simple_2d_function):
        """Test initial hyperparameter setup."""
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        y = np.array([simple_2d_function(x) for x in X])

        k = kriging(X, y)

        # Should have k theta values and k p values
        assert len(k.theta) == 2
        assert len(k.pl) == 2

    def test_init_normalization(self, seed):
        """Test that data is normalized to [0, 1]."""
        # Create data not in [0, 1]
        X = np.array([[0, 0], [10, 20], [5, 10]])
        y = np.array([100, 200, 150])

        k = kriging(X, y)

        # Internal X should be normalized
        X_cpu = k._backend.to_cpu(k.X)
        assert np.min(X_cpu) >= 0
        assert np.max(X_cpu) <= 1


class TestKrigingTraining:
    """Tests for kriging model training."""

    def test_train_completes(self, seed, simple_2d_function):
        """Test that training completes without error."""
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        y = np.array([simple_2d_function(x) for x in X])

        k = kriging(X, y)
        k.train(optimizer='ga')

        # Should have computed model parameters
        assert k.mu is not None
        assert k.SigmaSqr is not None

    def test_train_updates_hyperparameters(self, seed, simple_2d_function):
        """Test that training updates hyperparameters."""
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        y = np.array([simple_2d_function(x) for x in X])

        k = kriging(X, y)
        theta_before = k._backend.to_cpu(k.theta).copy()

        k.train(optimizer='ga')

        theta_after = k._backend.to_cpu(k.theta)
        # Hyperparameters should change during training
        assert not np.allclose(theta_before, theta_after)


class TestKrigingPrediction:
    """Tests for kriging model prediction."""

    def test_predict_returns_scalar(self, seed, simple_2d_function):
        """Test that predict returns a scalar."""
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        y = np.array([simple_2d_function(x) for x in X])

        k = kriging(X, y)
        k.train(optimizer='ga')

        pred = k.predict([0.5, 0.5])
        assert isinstance(pred, (int, float, np.floating))

    def test_predict_at_training_points(self, seed, simple_2d_function):
        """Test prediction accuracy at training points."""
        sp = samplingplan(k=2)
        X = sp.rlh(15)
        y = np.array([simple_2d_function(x) for x in X])

        k = kriging(X, y)
        k.train(optimizer='ga')

        # Predictions at training points should be close to actual values
        for i in range(len(X)):
            pred = k.predict(X[i])
            # Allow some tolerance due to normalization
            assert abs(pred - y[i]) < 0.5, f"Prediction {pred} far from actual {y[i]}"

    def test_predict_bounds(self, seed, simple_2d_function):
        """Test predictions are within reasonable bounds."""
        sp = samplingplan(k=2)
        X = sp.rlh(15)
        y = np.array([simple_2d_function(x) for x in X])

        k = kriging(X, y)
        k.train(optimizer='ga')

        # Generate test points
        test_X = sp.rlh(20)
        preds = [k.predict(x) for x in test_X]

        # Predictions should be within a reasonable range of training values
        y_range = np.max(y) - np.min(y)
        for pred in preds:
            assert pred >= np.min(y) - y_range
            assert pred <= np.max(y) + y_range


class TestKrigingUncertainty:
    """Tests for kriging uncertainty quantification."""

    def test_predicterr_returns_positive(self, seed, simple_2d_function):
        """Test that prediction error is positive."""
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        y = np.array([simple_2d_function(x) for x in X])

        k = kriging(X, y)
        k.train(optimizer='ga')

        err = k.predicterr([0.5, 0.5])
        assert err >= 0

    def test_predicterr_low_at_training_points(self, seed, simple_2d_function):
        """Test that uncertainty is low near training points."""
        sp = samplingplan(k=2)
        X = sp.rlh(15)
        y = np.array([simple_2d_function(x) for x in X])

        k = kriging(X, y)
        k.train(optimizer='ga')

        # Error at training points should be very low
        for i in range(min(5, len(X))):
            err = k.predicterr(X[i])
            assert err < 0.1, f"Uncertainty {err} too high at training point"


class TestKrigingAddPoint:
    """Tests for adding points to kriging model."""

    def test_addpoint_increases_n(self, seed, simple_2d_function):
        """Test that addPoint increases sample count."""
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        y = np.array([simple_2d_function(x) for x in X])

        k = kriging(X, y)
        k.train(optimizer='ga')

        n_before = k.n
        k.addPoint([0.5, 0.5], simple_2d_function([0.5, 0.5]))

        assert k.n == n_before + 1


class TestKrigingExpectedImprovement:
    """Tests for expected improvement calculation."""

    def test_infill_ei_returns_scalar(self, seed, simple_2d_function):
        """Test that infill_ei returns a scalar."""
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        y = np.array([simple_2d_function(x) for x in X])

        k = kriging(X, y)
        k.train(optimizer='ga')

        ei = k.infill_ei([0.5, 0.5])
        assert isinstance(ei, (int, float, np.floating))

    def test_infill_ei_non_negative(self, seed, simple_2d_function):
        """Test that expected improvement is non-negative."""
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        y = np.array([simple_2d_function(x) for x in X])

        k = kriging(X, y)
        k.train(optimizer='ga')

        # EI should be non-negative everywhere
        test_points = sp.rlh(20)
        for pt in test_points:
            ei = k.infill_ei(pt)
            assert ei >= -1e-10  # Allow small numerical errors


class TestKrigingNegLikelihood:
    """Tests for negative log-likelihood calculation."""

    def test_neglikelihood_finite(self, seed, simple_2d_function):
        """Test that negative log-likelihood is finite."""
        sp = samplingplan(k=2)
        X = sp.rlh(10)
        y = np.array([simple_2d_function(x) for x in X])

        k = kriging(X, y)
        k.updateModel()
        k.neglikelihood()

        nll = k._backend.to_cpu(k.NegLnLike)
        assert np.isfinite(nll)
