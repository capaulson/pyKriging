__author__ = 'chrispaulson'
import dill as pickle
import numpy as np
from copy import deepcopy
def norm(x):
    # x = np.array(x, dtype=float)
    # return ((x-min(x))/(max(x)-min(x)))
    x =  ((x)/(max(x)-min(x)))
    return x-min(x)

def saveModel(model, filePath):
    pickle.dump(model, open(filePath, 'w'), byref=True)

def loadModel(filePath):
    return pickle.load(open(filePath,'r'))

def splitArrays(krigeModel, q=5):
    ind = np.arange(krigeModel.n)
    np.random.shuffle(ind)
    test = np.array_split(ind,q)
    for i in test:
        newX = deepcopy(krigeModel.X)
        newy = deepcopy(krigeModel.y)

        testX = newX[i]
        testy = newy[i]

        trainX = np.delete(newX,i,axis=0)
        trainy = np.delete(newy,i,axis=0)
        yield trainX, trainy, testX, testy

def mse(actual, predicted):
    return ((actual - predicted) ** 2)
