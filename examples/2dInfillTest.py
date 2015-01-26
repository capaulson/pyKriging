import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan


## The Kriging model starts by defining a sampling plan, we use an optimal Lattin Hypercube here
sp = samplingplan(2)
X = sp.optimallhc(11)
## Next, we define the problem we would like to solve
testfun = pyKriging.testfunctions().branin
y = testfun(X)


tp = sp.fullfactorial(25)
import matplotlib.pyplot as plt
def plotTest(km):
    print km.theta
    print km.pl, '\n\n'
    y1 = testfun(tp)
    y2 = []
    for entry in tp:
        y2.append(km.predict(entry))
    fig = plt.figure()
    plt.plot(y1,y2,'.m')
    plt.plot(y1,y1,'r')
    plt.show()

## Now that we have our initial data, we can create an instance of a kriging model
k = kriging(X, y, testfunction=testfun)

## The model is then trained
k.train()
plotTest(k)

## It's typically beneficial to add additional points based on the results of the initial training
## The infill method can be  used for this
## In this example, we will add nine points in three batches. The model gets trained after each stage
for i in range(5):
    infillPoints = k.infill(5)

    ## Evaluate the infill points and add them back to the Kriging model
    for point in infillPoints:
        k.addPoint(point, testfun(point))

    ## Retrain the model with the new points added in to the model
    k.train()
    plotTest(k)
## Once the training of the model is complete, we can plot the results
k.plot()
plotTest(k)



#### This code allows data to be transferred to Matlab through a 'wormhole'

# print 'Establishing the wormhole now... '
# from Wormhole import Wormhole
# W = Wormhole()
# import time
# time.sleep(5)
# W.put("X",k.X)
# time.sleep(1)
# W.put("y",k.y)
# time.sleep(1)
#
# W.put("theta",k.theta)
# time.sleep(1)
#
# W.put("pl",k.pl)
# time.sleep(1)
# exit()

