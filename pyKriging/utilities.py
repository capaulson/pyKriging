"""Model persistence, normalisation and cross-validation helpers."""
__author__ = 'chrispaulson'
import numpy as np


def norm(x):
    """Scale an array to the range [0, 1]."""
    x = x / (np.max(x) - np.min(x))
    return x - np.min(x)


def _by_reference(obj):
    """pyKriging's own classes are pickled by reference (pyKriging/__init__ rebinds the submodule names samplingplan
    and testfunctions to classes, which dill >= 0.3.6 cannot follow and would otherwise copy the classes by value)."""
    if isinstance(obj, type) and obj.__module__.startswith('pyKriging.') and '<' not in obj.__qualname__:
        return obj.__qualname__  # resolved by pickle's save_global through sys.modules
    return NotImplemented


def saveModel(model, filePath):
    """Save a model with dill. Functions and classes defined in the calling script (such as a test function) are
    stored by value with the globals they use, so the model also loads in another process."""
    import io
    import dill
    getattr(getattr(dill, '_dill', None), 'stack', {}).clear()  # dill < 0.3.5 keeps per-dump state in a global
    buffer = io.BytesIO()
    pickler = dill.Pickler(buffer, recurse=True)
    pickler.reducer_override = _by_reference
    pickler.dump(model)
    with open(filePath, 'wb') as fh:  # written only once pickling has succeeded
        fh.write(buffer.getvalue())


def loadModel(filePath):
    """Load a model saved with saveModel()."""
    import dill
    with open(filePath, 'rb') as fh:
        return dill.load(fh)


def splitArrays(krigeModel, q=5):
    """Yield (trainX, trainy, testX, testy) for q-fold cross-validation over the model's (normalised) data."""
    folds = np.array_split(np.random.permutation(krigeModel.n), q)
    for test in folds:
        yield (np.delete(krigeModel.X, test, axis=0), np.delete(krigeModel.y, test, axis=0),
               krigeModel.X[test], krigeModel.y[test])


def mse(actual, predicted):
    """Squared error (actual - predicted)^2."""
    return (actual - predicted) ** 2
