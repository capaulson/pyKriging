__author__ = 'cpaulson'
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan

# The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
sp = samplingplan(2)
X = sp.optimallhc(15)

# Next, we define the problem we would like to solve
testfun = pyKriging.testfunctions().branin

# We generate our observed values based on our sampling plan and the test function
y = testfun(X)

print 'Setting up the Kriging Model'

# Now that we have our initial data, we can create an instance of a kriging model
k = kriging(X, y, testfunction=testfun, name='simple', testPoints=250)
k.train(optimizer='ga')
k.snapshot()

# Let's setup our infill to terminate once our prediction has 'converged'
# That is to say that two iterations of predictions have an rsquared value of something like 0.9999
while k.history['rsquared'][-1]<0.9999:
    newpoints = k.infill(2)
    for point in newpoints:
        print 'Adding point {}'.format( point )
        k.addPoint(point, testfun(point)[0])
    k.train()
    k.snapshot()
    print 'Current rsquared is: {}'.format(k.history['rsquared'][-1])

print 'The prediction has converged, with {} number of points in the model'.format(k.n)

# #And plot the model

print 'Now plotting final results...'
k.plot()


