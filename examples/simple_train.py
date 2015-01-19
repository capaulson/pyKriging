__author__ = 'cpaulson'

import sys
sys.path.insert(0, '..')
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan

# The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
sp = samplingplan(2)
X = sp.optimallhc(20)

# Next, we define the problem we would like to solve
# testfun = pyKrige.testfunctions2d().paulson1
# testfun = pyKrige.testfunctions2d().squared
testfun = pyKrige.testfunctions2d().branin
y = testfun(X)

print 'Setting up the Kriging Model'
# Now that we have our initial data, we can create an instance of a kriging model
k = kriging(X, y, testfunction=testfun, name='simple')
k.train()
print 'Done with training, moving on to infill'

# Now, you can add infill points
numberiter = 5
for i in range(numberiter):
    print 'Infill iteration {0} of {1}....'.format(i + 1, numberiter)
    newpoints = k.infill(1)
    for point in newpoints:
        k.addPoint(point, testfun(point)[0])
    k.train()

# And plot the results
k.plot()