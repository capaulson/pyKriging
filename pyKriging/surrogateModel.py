__author__ = 'chrispaulson'

from pyGPs.Core import *
from pyGPs.Valid import valid
import numpy as np
import resultsViewer

import numpy as np
from pyOpt import Optimization
from pyOpt import NSGA2, ALPSO
import copy
# import mayavi

import krige
import matrixops

a = resultsViewer.pullResults()
x,y = a.loadData()

x = np.array(x)
print x.shape
print x
y = np.abs(np.array(y))

a = krige.oneDSM(X=x, y=y)
print x

# Configure the Optimization
opt_prob = Optimization('Surrogate Test', a.update)

for i in range(x.shape[1]):
    opt_prob.addVar('theta%d'%i,'c',lower=.05,upper=20,value=.2)
for i in range(x.shape[1]):
    opt_prob.addVar('pl%d'%i,'c',lower=1,upper=2,value=1.75)

opt_prob.addObj('f')
opt_prob.addCon('g1', 'i')

#print out the problem
print opt_prob

#Run the GA
# nsga = NSGA2(PopSize=300, maxGen=500, pMut_real=.1)
# nsga(opt_prob)
#
pso = ALPSO()
pso.setOption('SwarmSize',30)
pso.setOption('maxOuterIter',100)
pso.setOption('stopCriteria',1)
# pso.setOption('dt',1)
pso(opt_prob)

#print the best solution
print opt_prob.solution(0)

# Update the model variables to the best solution found by the optimizer
a.update([opt_prob.solution(0)._variables[0].value, opt_prob.solution(0)._variables[1].value,opt_prob.solution(0)._variables[2].value, opt_prob.solution(0)._variables[3].value,  opt_prob.solution(0)._variables[4].value, opt_prob.solution(0)._variables[5].value])


for enu,i in enumerate(x):
    print y[enu], a.predict(i)
