from __future__ import print_function
__author__ = 'cpaulson'
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan

# The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
sp = samplingplan(2)
X = sp.optimallhc(15)

# Next, we define the problem we would like to solve
testfun = pyKriging.testfunctions().paulson1
y = testfun(X)

# We can choose between a ga and a pso here
optimizer = 'ga'

# Now that we have our initial data, we can create an instance of a kriging model
print('Setting up the Kriging Model')
k = kriging(X, y, testfunction=testfun, name='simple_ei', testPoints=300)
k.train(optimizer=optimizer)
k.snapshot()


# Add 10 points based on model error reduction
for i in range(5):
    newpoints = k.infill(1, method='error')
    for point in newpoints:
        print('Adding point {}'.format(point))
        k.addPoint(point, testfun(point)[0])
    k.train(optimizer=optimizer)
    k.snapshot()

# Infill ten points based on the expected improvement criterion
for i in range(5):
    newpoints = k.infill(1, method='ei')
    for point in newpoints:
        print('Adding point {}'.format(point))
        k.addPoint(point, testfun(point)[0])
    k.train(optimizer=optimizer)
    k.snapshot()

# And plot the results
print('Now plotting final results...')
k.plot()



