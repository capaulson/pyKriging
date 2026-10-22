"""
pyKriging - Kriging (Gaussian-process) surrogate modelling.

    from pyKriging import kriging, samplingplan
    X = samplingplan(k=2).optimallhc(20)
    model = kriging(X, y); model.train(); model.predict(X_new)

See https://github.com/capaulson/pyKriging
"""
__author__ = 'chrispaulson'
__version__ = '0.2.0'

from .krige import kriging
from .regressionkrige import regression_kriging
from .samplingplan import samplingplan
from .testfunctions import testfunctions
from .utilities import loadModel, mse, norm, saveModel, splitArrays
from .gpu_backend import configure_gpu, get_backend, get_device_info, is_gpu_available, to_cpu, to_gpu

__all__ = ['kriging', 'regression_kriging', 'samplingplan', 'testfunctions', 'saveModel', 'loadModel',
           'splitArrays', 'mse', 'norm', 'configure_gpu', 'get_device_info', 'is_gpu_available', 'get_backend',
           'to_cpu', 'to_gpu', 'coKriging', 'CrossValidation', 'GlobalSensitivity']  # the last three are submodules
