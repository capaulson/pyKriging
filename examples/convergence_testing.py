__author__ = 'cpaulson'
import sys
sys.path.insert(0,'..')
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

cd = {}
cd['iter']=[]
cd['theta1']=[]
cd['theta2']=[]
cd['pl1']=[]
cd['pl2']=[]
cd['msemean']=[]
cd['msestd']=[]
cd['realerrmean']=[]
cd['realerrstd']=[]
cd['neglnlike']=[]


## The Kriging model starts by defining a sampling plan, we use an optimal Lattin Hypercube here
sp = samplingplan(2)
X = sp.optimallhc(10)
XTest = sp.rlh(500)

# X = (X * 10)-5
#Create our RH sampling plan for monitoring model value convergence
points2sample = 500
testPoints = sp.rlh(points2sample)
# testPoints = (testPoints * 10) -5
testPointresults = []
for i in range(points2sample):
    testPointresults.append([])

## Next, we define the problem we would like to solve
# testfun = pyKriging.testfunctions().paulson1
testfun = pyKriging.testfunctions().stybtang_norm
y = testfun(X)

## Now that we have our initial data, we can create an instance of a kriging model
k = kriging(X, y, testfunction=testfun)

## The model is then trained
k.train()
results = k.calcuatemeanMSE(points=testPoints)
cd['msemean'].append(results[0])
cd['msestd'].append(results[1])
tmp = []
for enu,i in enumerate(testPoints):
    tmp.append(np.abs(k.predict(i)- testfun(i)[0])/testfun(i)[0])
    # print enu, i, k.predict(i), testfun(i)
    testPointresults[enu].append(k.predict(i))

cd['realerrmean'].append(np.mean(tmp))
cd['realerrstd'].append(np.std(tmp))
cd['theta1'].append(k.theta[0])
cd['theta2'].append(k.theta[1])
cd['pl1'].append(k.pl[0])
cd['pl2'].append(k.pl[1])
cd['neglnlike'].append(k.NegLnLike)
cd['iter'].append(0)

## It's typically beneficial to add additional points based on the results of the initial training
## The infill method can be  used for this
## In this example, we will add nine points in three batches. The model gets trained after each stage
for i in range(50):
    infillPoints = k.infill(1)

    ## Evaluate the infill points and add them back to the Kriging model
    for point in infillPoints:
        print point
        k.addPoint(point, testfun(point))

    ## Retrain the model with the new points added in to the model
    # previousneglnlike = k.NegLnLike
    # while k.NegLnLike>=previousneglnlike:
    #     k.train()
    #     print 'loop'
    k.train()
    # k.saveFigure(name='converge_infill_iter_{0}.png'.format(i))

    results = k.calcuatemeanMSE(points=testPoints)
    cd['msemean'].append(results[0])
    cd['msestd'].append(results[1])

    tmp = []
    for enu,j in enumerate(testPoints):
        tmp.append(np.abs(k.predict(j)- testfun(j)[0])/testfun(j)[0])
        testPointresults[enu].append(k.predict(j))

    cd['realerrmean'].append(np.mean(tmp))
    cd['realerrstd'].append(np.std(tmp))
    cd['theta1'].append(k.theta[0])
    cd['theta2'].append(k.theta[1])
    cd['pl1'].append(k.pl[0])
    cd['pl2'].append(k.pl[1])
    cd['neglnlike'].append(k.NegLnLike)
    cd['iter'].append(i+1)
## Once the training of the model is complete, we can plot the results

k.plot()
#
#
# for i in range(points2sample):
#     # print np.mean(np.abs(testPointresults[i]- testfun(testPoints[i])[0])/testfun(testPoints[i])[0])
#     plt.plot(  range(len(testPointresults[i])), (testPointresults[i]- testfun(testPoints[i])[0])/testfun(testPoints[i])[0])
# plt.show()
# print cd
# print testPointresults
gradResults = []
for i in testPointresults:
    gradResults.append(np.gradient(i))
# print gradResults

grads = np.mean(gradResults,axis=0)
gradsstd = np.std(gradResults, axis=0)

f, (ax0,ax1, ax2, ax3,ax4,ax5 ) = plt.subplots(6, sharex=True, sharey=False)

ax0.plot(np.array(cd['iter']),np.array(cd['neglnlike']))
ax0.set_ylabel('neglnlike')

ax1.errorbar(cd['iter'], cd['realerrmean'], yerr=cd['realerrstd'])
ax1.set_ylabel('Actual Error')

ax2.errorbar(cd['iter'], cd['msemean'], yerr=cd['msestd'])
ax2.set_ylabel('Predicted MSE')

ax3.plot(cd['iter'], cd['theta1'])
ax3.plot(cd['iter'], cd['theta2'])
ax3.set_ylabel('Theta Values')

ax4.plot(cd['iter'], cd['pl1'])
ax4.plot(cd['iter'], cd['pl2'])
ax4.set_ylabel('Pl Values')

ax5.errorbar(cd['iter'], grads, yerr=gradsstd)
ax5.set_ylabel('Mean gradient')

plt.show()