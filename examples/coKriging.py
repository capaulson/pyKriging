__author__ = 'cpaulson'

import sys
sys.path.insert(0,'.')
sys.path.insert(0,'..')
from pyKriging import coKriging

import numpy as np

def cheap(X):

    A=0.5
    B=10
    C=-5
    D=0

    print X
    print ((X+D)*6-2)
    return A*np.power( ((X+D)*6-2), 2 )*np.sin(((X+D)*6-2)*2)+((X+D)-0.5)*B+C

def expensive(X):
    return np.power((X*6-2),2)*np.sin((X*6-2)*2)


Xe = np.array([0, 0.4, 0.6, 1])
Xc = np.array([0.1,0.2,0.3,0.5,0.7,0.8,0.9,0,0.4,0.6,1])

yc = cheap(Xc)
ye = expensive(Xe)

ck = coKriging.coKriging(Xc, yc, Xe, ye)
ck.thetac = np.array([1.2073])
print ck.Xc
ck.updateData()
ck.updatePsi()
ck.neglnlikehood()

