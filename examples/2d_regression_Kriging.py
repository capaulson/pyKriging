from __future__ import print_function
__author__ = 'cpaulson'
import sys
sys.path.insert(0, '../')
import pyKriging
from pyKriging.regressionkrige import regression_kriging
from pyKriging.samplingplan import samplingplan


from pyKriging.krige import kriging
# The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
sp = samplingplan(2)
X = sp.optimallhc(30)

# Next, we define the problem we would like to solve
testfun = pyKriging.testfunctions().branin_noise

# We generate our observed values based on our sampling plan and the test function
y = testfun(X)
print(X, y)

testfun = pyKriging.testfunctions().branin


print('Setting up the Kriging Model')

# Now that we have our initial data, we can create an instance of a kriging model
k = regression_kriging(X, y, testfunction=testfun, name='simple', testPoints=250)
k.train(optimizer='pso')
k1 = kriging(X, y, testfunction=testfun, name='simple', testPoints=250)
k1.train(optimizer='pso')
print(k.Lambda)
k.snapshot()


for i in range(1):
    newpoints = k.infill(5)
    for point in newpoints:
        print('Adding point {}'.format(point))
        newValue = testfun(point)[0]
        k.addPoint(point, newValue)
        k1.addPoint(point, newValue)
    k.train()
    k1.train()
    # k.snapshot()
#
# # #And plot the model

print('Now plotting final results...')
print(k.Lambda)
k.plot(show=False)
k1.plot()


