__author__ = 'chrispaulson'
import numpy as np
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan
from pyKriging.testfunctions import testfunctions2d
from pyKriging.utilities import mse, splitArrays

sp = samplingplan()
X = sp.optimallhc(10)
t = testfunctions()
y = t.branin(X)
k = kriging(X,y, testfunction=t.branin)
k.train()

for l in range(5):
    msemeanArray = []
    msestdArray = []
    for i in splitArrays(k,6):
        testk = kriging( i[0], i[1] )
        testk.train()
        msemean, msestd = testk.calcuatemeanMSE()
        msemeanArray.append(msemean)
        msestdArray.append(msestd)
        del(testk)
    print msemeanArray
    print k.n, np.mean(msemeanArray), np.std(msestdArray)

    infillPoints = k.infill(5)
    ## Evaluate the infill points and add them back to the Kriging model
    for point in infillPoints:
        k.addPoint(point, t.branin(point))
    ## Retrain the model with the new points added in to the model
    k.train()

mseArray = []
for i in splitArrays(k,5):
    testk = kriging( i[0], i[1] )
    testk.train()
    msemean, msestd = testk.calcuatemeanMSE()
    msemeanArray.append(msemean)
    msestdArray.append(msestd)
    del(testk)

print k.n, np.mean(msemeanArray), np.std(msestdArray)
k.plot()