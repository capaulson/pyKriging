__author__ = 'chrispaulson'
import numpy as np
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan
from pyKriging.testfunctions import testfunctions2d
from pyKriging.utilities import mse, splitArrays

sp = samplingplan()
X = sp.optimallhc(16)
t = testfunctions2d()
y = t.branin(X)
k = kriging(X,y, optimizer='ga')
k.train(optimizer='ga')

for l in range(5):
    mseArray = []
    for i in splitArrays(k,5):
        testk = kriging( i[0], i[1] )
        testk.train()
        for j in range(len(i[2])):
            mseArray.append(mse(i[3][j], testk.predict( i[2][j] )))
        del(testk)

    print k.n, np.mean(mseArray), np.std(mseArray)

    infillPoints = k.infill(5)
    ## Evaluate the infill points and add them back to the Kriging model
    for point in infillPoints:
        k.addPoint(point, t.branin(point))
    ## Retrain the model with the new points added in to the model
    k.train(optimizer='ga')

mseArray = []
for i in splitArrays(k,5):
    testk = kriging( i[0], i[1] )
    testk.train()
    for j in range(len(i[2])):
        mseArray.append(mse(i[3][j], testk.predict( i[2][j] )))
    del(testk)

print k.n, np.mean(mseArray), np.std(mseArray)
k.plot()