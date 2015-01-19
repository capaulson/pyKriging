from __future__ import division
import numpy as np
from pyOpt import Optimization
from pyOpt import NSGA2
from krige import *
import cProfile


# theta = np.array([60.1, 10.5])
# pl = np.array([1.84, 1.32])
# x1 = np.array([0, 0])
# x2 = np.array([.1,.1])
# command = 'print np.exp(-1*np.sum(theta*np.abs(x1-x2)**pl))'
# cProfile.runctx( command, globals(), locals() )
# exit()


#Define the problem
def testfun(x,y):
    return np.sin(x*5) + np.cos(y*5)

#X here is a simple full factorial plan
ngrid = 3
X = np.mgrid[0:1:(ngrid*1j), 0:1:(ngrid*1j)].reshape(2, ngrid**2).T

#Create the training data
y = testfun(X[:, 0], X[:, 1])

#Start create an instance of a Surrogate Model
a = oneDSM(X=X, y=y)
a.update([4,4,2,2])
print 'new_method'
a.updatePhi1()
# print a.phi

print '\n\nold_method'
a.updatePhi()
# print a.phi

# command = 'a = oneDSM(X=X, y=y); a.update([4,3,1.329,1.33])'
# cProfile.runctx( command, globals(), locals() )