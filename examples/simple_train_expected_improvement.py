__author__ = 'cpaulson'
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan

# The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
sp = samplingplan(2)
X = sp.optimallhc(10)

# Next, we define the problem we would like to solve
testfun = pyKriging.testfunctions().branin
y = testfun(X)

# Now that we have our initial data, we can create an instance of a kriging model
print 'Setting up the Kriging Model'
k = kriging(X, y, testfunction=testfun, name='simple_ei', testPoints=300)
k.train(optimizer='pso')
k.snapshot()

# Infill five points based on the expected improvement criterion
for i in range(5):
    newpoints = k.infill(1, method='ei')
    for point in newpoints:
        print point
        k.addPoint(point, testfun(point)[0])
    k.train(optimizer='pso')
    k.snapshot()

#Add a further 5 points based on model error reduction
newpoints = k.infill(5, method='error')
for point in newpoints:
    print point
    k.addPoint(point, testfun(point)[0])
k.train(optimizer='pso')
k.snapshot()

#And plot the results
print 'Now plotting final results...'
k.plot()



