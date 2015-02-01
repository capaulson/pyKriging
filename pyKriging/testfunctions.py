from __future__ import division
import numpy as np

class testfunctions():
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

    def stybtang(self,X):
        try:
            X.shape[1]
        except:
            X = np.array([X])
        d = X.shape[1]
        y = []
        for entry in X:
            sum = 0
            for i in range(d):
                xi = entry[i]
                new = np.power(xi,4) - 16*np.power(xi,2) + 5*xi
                sum = sum + new

            y.append(sum/2.)
        return  np.array(y)

    def stybtang_norm(self,X):
        try:
            X.shape[1]
        except:
            X = np.array([X])
        X = (X *10)-5
        d = X.shape[1]
        y = []
        for entry in X:
            sum = 0
            for i in range(d):
                xi = entry[i]
                new = np.power(xi,4) - 16*np.power(xi,2) + 5*xi
                sum = sum + new

            y.append(sum/2.)
        return  np.array(y)

    def curretal88exp(self,X):
        try:
            X.shape[1]
        except:
            X = np.array([X])
        x1 = X[:,0]
        x2 = X[:,1]

        fact1 = 1 - np.exp(-1/(2*x2))
        fact2 = 2300*np.power(x1,3) + 1900*np.power(x1,2) + 2092*x1 + 60
        fact3 = 100*np.power(x1,3) + 500*np.power(x1,2) + 4*x1 + 20

        return (fact1 * fact2/fact3)




if __name__=='__main__':
    a = testfunctions()
    print a.squared([1,1,1])
    print a.squared([[1,1,1],[2,2,2]])
    print a.cubed([[1,1,1],[2,2,2]])
    print a.stybtang([[1,1,1],[2,2,2]])
    print a.curretal88exp([[1,1,1],[2,2,2]])
    print a.zhou98([[1,1,1],[2,2,2]])