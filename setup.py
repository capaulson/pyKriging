from setuptools import setup, find_packages # Always prefer setuptools over distutils
from os import path, walk


here = path.abspath(path.dirname(__file__))
datadir = 'pyKriging/sampling_plans'
package_data = [ (d, [path.join(d, f) for f in files]) for d,folders,files in walk(datadir)]
data_files=[]
for i in package_data:
    for j in i[1]:
        data_files.append(j)
data_files = [path.relpath(file, datadir) for file in data_files]

setup(
    name='pyKriging',
    version='0.1.0',
    zip_safe = False,
    packages=find_packages(),
    package_data={'pyKriging': ['sampling_plans/*']},
    url='www.pykriging.com',
    license='',
    author='Chris Paulson',
    author_email='capaulson@gmail.com',
    description='A Kriging Toolbox for Python with GPU Acceleration',
    long_description='''
    pyKriging provides efficient Kriging (Gaussian Process) metamodeling with
    automatic GPU acceleration when available. Supports NVIDIA GPUs (via CuPy)
    and Apple Silicon (via PyTorch MPS).

    GPU acceleration provides 10-50x speedups for training on large datasets.
    Falls back to CPU (NumPy) automatically when GPU is unavailable.

    For more information visit: https://github.com/capaulson/pyKriging
    ''',
    install_requires=['scipy', 'numpy', 'dill', 'matplotlib', 'inspyred'],

    # Optional GPU dependencies
    extras_require={
        'gpu-nvidia': [
            'cupy>=12.0.0',  # CUDA GPU support for NVIDIA
        ],
        'gpu-metal': [
            'torch>=2.0.0',  # Metal GPU support for Apple Silicon (via MPS)
        ],
        'gpu-all': [
            # Install both for maximum compatibility
            'cupy>=12.0.0',
            'torch>=2.0.0',
        ],
    },

    # Metadata
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='kriging, gaussian process, metamodel, surrogate, gpu, cuda, metal',
)