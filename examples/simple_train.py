__author__ = 'cpaulson'
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan

# The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
sp = samplingplan(2)
X = sp.optimallhc(20)
# Next, we define the problem we would like to solve
testfun = pyKriging.testfunctions().branin
# testfun = pyKriging.testfunctions().paulson1
# testfun = pyKriging.testfunctions().squared
# testfun = pyKriging.testfunctions().stybtang_norm
# testfun = pyKriging.testfunctions().curretal88exp
# testfun = pyKriging.testfunctions().zhou98
y = testfun(X)

print 'Setting up the Kriging Model'
# Now that we have our initial data, we can create an instance of a kriging model
k = kriging(X, y, testfunction=testfun, name='simple')
k.train()

numberiter = 1
for i in range(numberiter):
    print 'Infill iteration {0} of {1}....'.format(i + 1, numberiter)
    newpoints = k.infill(5)

    for point in newpoints:
        k.addPoint(point, testfun(point)[0])
    k.train()
    # k.saveFigure(name='Infill_Iter_{0}.png'.format(i))

# And plot the results
print 'Now plotting results...'
k.plot()