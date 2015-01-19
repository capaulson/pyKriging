__author__ = 'chrispaulson'
import numpy as np

class testfunctions2d():
    def linear(self, X):
        try:
            X.shape[1]
        except:
            X = np.array(X)

        if len(X.shape)<2:
            X = np.array([X])
        y = np.array([],dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, np.sum(X[i]))
        return y

    def squared(self, X, offset =.25):
        try:
            X.shape[1]
        except:
            X = np.array(X)

        if len(X.shape)<2:
            X = np.array([X])
        offset = np.ones(X.shape[1])*offset
        y = np.array([],dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, (np.sum((X[i]-offset)**2)**0.5))
        return y

    def cubed(self, X, offset=.25):
        try:
            X.shape[1]
        except:
            X = np.array(X)

        if len(X.shape)<2:
            X = np.array([X])
        offset = np.ones(X.shape[1])*offset
        y = np.array([],dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, (np.sum((X[i]-offset)**3)**(1/3.0)))
        return y

    def branin(self, X):
        try:
            X.shape[1]
        except:
            X = np.array([X])

        if X.shape[1] != 2:
            raise Exception
        x = X[:,0]
        y = X[:,1]
        X1 = 15*x-5
        X2 = 15*y
        a = 1
        b = 5.1/(4*np.pi**2)
        c = 5/np.pi
        d = 6
        e = 10
        ff = 1/(8*np.pi)
        return (a*( X2 - b*X1**2 + c*X1 - d )**2 + e*(1-ff)*np.cos(X1) + e)+5*x

    def branin_noise(self, X):
        try:
            X.shape[1]
        except:
            X = np.array([X])

        if X.shape[1] != 2:
            raise Exception
        x = X[:,0]
        y = X[:,1]
        X1 = 15*x-5
        X2 = 15*y
        a = 1
        b = 5.1/(4*np.pi**2)
        c = 5/np.pi
        d = 6
        e = 10
        ff = 1/(8*np.pi)
        noiseFree =  ((a*( X2 - b*X1**2 + c*X1 - d )**2 + e*(1-ff)*np.cos(X1) + e)+5*x)
        withNoise=[]
        for i in noiseFree:
            withNoise.append(i* ((np.random.standard_normal()/7.)+1))
        return np.array(withNoise)


    def paulson(self,X,hz=5):
        try:
            X.shape[1]
        except:
            X = np.array([X])
        if X.shape[1] != 2:
            raise Exception
        x = X[:,0]
        y = X[:,1]
        return .5*np.sin(x*hz) + .5*np.cos(y*hz)

    def paulson1(self,X,hz=10):
        try:
            X.shape[1]
        except:
            X = np.array([X])
        if X.shape[1] != 2:
            raise Exception
        x = X[:,0]
        y = X[:,1]
        return (np.sin(x*hz))/((x+.2)) + (np.cos(y*hz))/((y+.2))

if __name__=='__main__':
    a = testfunctions2d()
    print a.squared([1,1,1])
    print a.squared([[1,1,1],[2,2,2]])
    print a.cubed([[1,1,1],[2,2,2]])