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
    version='0.0.3',
    zip_safe = False,
    packages=find_packages(),
    package_data={'pyKriging': ['sampling_plans/*']},
    url='www.pykriging.com',
    license='',
    author='Chris Paulson',
    author_email='capaulson@gmail.com',
    description='A Kriging Toolbox for Python',
    install_requires=['scipy', 'numpy', 'dill', 'matplotlib'],


)