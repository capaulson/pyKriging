import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan
import dill as pickle


## The Kriging model starts by defining a sampling plan, we use an optimal Lattin Hypercube here
sp = samplingplan(3)
X = sp.optimallhc(90)

## Next, we define the problem we would like to solve
testfun = pyKriging.testfunctions.squared
y = testfun(X)

## Now that we have our initial data, we can create an instance of a kriging model
k = kriging(X, y, testfunction=testfun)

## The model is then trained
k.train()

# It's typically beneficial to add additional points based on the results of the initial training
# The infill method can be  used for this
# In this example, we will add nine points in three batches. The model gets trained after each stage
print 'Finished Model Training'
for i in range(5):
    print 'Infill itteration {0}'.format((i+1))
    infillPoints = k.infill(20)

    ## Evaluate the infill points and add them back to the Kriging model
    for point in infillPoints:
        k.addPoint(point, testfun(point)[0])

    ## Retrain the model with the new points added in to the model
    k.train()

# Save the trained model
pickle.dump(k,  open('trainedModel.pkl', 'wb'))

## Once the training of the model is complete, we can plot the results
k.plot()

