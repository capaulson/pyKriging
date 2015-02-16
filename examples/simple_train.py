__author__ = 'cpaulson'
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan

# The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
sp = samplingplan(2)
X = sp.optimallhc(15)

# Next, we define the problem we would like to solve
testfun = pyKriging.testfunctions().branin

y = testfun(X)

print 'Setting up the Kriging Model'

# Now that we have our initial data, we can create an instance of a kriging model
k = kriging(X, y, testfunction=testfun, name='simple', testPoints=300)
k.train(optimizer='pso')
k.snapshot()


for i in range(5):
    newpoints = k.infill(2)
    for point in newpoints:
        print point
        k.addPoint(point, testfun(point)[0])
    k.train(optimizer='pso')
    k.snapshot()

# #And plot the results
print 'Now plotting final results...'
k.plot()


