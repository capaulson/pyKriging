import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan
from pyKriging.testfunctions import testfunctions


## The Kriging model starts by defining a sampling plan, we use an optimal Lattin Hypercube here
sp = samplingplan(3)
X = sp.optimallhc(30)

## Next, we define the problem we would like to solve
testfun = testfunctions().squared
y = testfun(X)

## Now that we have our initial data, we can create an instance of a kriging model
k = kriging(X, y, testfunction=testfun, testPoints=300)

## The model is then trained
k.train()
k.snapshot()

# It's typically beneficial to add additional points based on the results of the initial training
# The infill method can be  used for this
# In this example, we will add nine points in three batches. The model gets trained after each stage
for i in range( 10 ):
    print k.history['rsquared'][-1]
    print 'Infill itteration {0}'.format( i+1 )
    infillPoints = k.infill(10)

    ## Evaluate the infill points and add them back to the Kriging model
    for point in infillPoints:
        print 'Adding point {}'.format( point )
        k.addPoint(point, testfun(point)[0])

    ## Retrain the model with the new points added in to the model
    k.train()
    k.snapshot()

## Once the training of the model is complete, we can plot the results
k.plot()

