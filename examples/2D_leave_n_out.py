__author__ = 'cpaulson'
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan
from pyKriging.CrossValidation import Cross_Validation
from pyKriging.utilities import saveModel

# The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
sp = samplingplan(2)
X = sp.optimallhc(5)

# Next, we define the problem we would like to solve
testfun = pyKriging.testfunctions().branin

# We generate our observed values based on our sampling plan and the test function
y = testfun(X)

print 'Setting up the Kriging Model'
cvMSE = []
# Now that we have our initial data, we can create an instance of a kriging model
k = kriging(X, y, testfunction=testfun, name='simple', testPoints=300)
k.train(optimizer='ga')
k.snapshot()
# cv = Cross_Validation(k)
# cvMSE.append( cv.leave_n_out(q=5)[0] )

k.plot()
for i in range(15):
    print i
    newpoints = k.infill(1)
    for point in newpoints:
        # print 'Adding point {}'.format(point)
        k.addPoint(point, testfun(point)[0])
    k.train(optimizer='pso')
    k.snapshot()
    # cv = Cross_Validation(k)
    # cvMSE.append( cv.leave_n_out(q=5)[0] )
k.plot()



# saveModel(k, 'crossValidation.plk')

# #And plot the model

print 'Now plotting final results...'
# k.plot()


print k.testPoints
print k.history['points']
print k.history['rsquared']
print k.history['avgMSE']
print cvMSE
from matplotlib import pylab as plt
plt.plot(range(len(k.history['rsquared'])), k.history['rsquared'])
plt.plot(range(len(cvMSE)), cvMSE)
plt.show()

