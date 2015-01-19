from setuptools import setup, find_packages # Always prefer setuptools over distutils

setup(
    name='pyKriging',
    version='0.0.1',
    packages=find_packages(),
    url='www.pykriging.com',
    license='',
    author='Chris Paulson',
    author_email='capaulson@gmail.com',
    description='A Kriging Toolbox for Python',
    install_requires=['scipy', 'numpy', 'dill'],

)