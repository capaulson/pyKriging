__author__ = 'cpaulson'

from sys import exit
import numpy as np
from numpy.matlib import rand,zeros,ones,empty,eye
from pyKriging import kriging

class coKriging():
    def __init__(self, Xc, yc, Xe, ye):

        # Create the data arrays
        self.Xc = np.atleast_2d(Xc).T
        self.yc = yc
        self.nc = self.Xc.shape[0]

        self.Xe = np.atleast_2d(Xe).T

        self.ye = ye
        self.ne = self.Xe.shape[0]

        # rho regression parameter
        self.rho = 1.9961
        self.reorder_data()
        # self.traincheap()

        self.k = self.Xc.shape[1]
        # if self.Xe.shape[1] != self.Xc.shape[1]:
        #     print 'Xc and Xe must have the same number of design variables. Fatal error -- Exiting...'
        #     exit()

        # Configure the hyperparameter arrays
        self.thetad = np.ones(self.k)
        self.thetac = None
        # self.thetac = self.kc.theta

        self.pd = np.ones(self.k) * 2.
        # self.pc = self.kc.pl
        self.pc = np.ones(self.k) * 2.

        # Matrix Operations
        self.one=ones([self.ne+self.nc,1])
        self.y=[self.yc, self.ye]

        print('here1')

    def reorder_data(self):
        xe = []
        ye = []
        xc = []
        yc = []

        Xd = []
        yd = []

        for enu,entry in enumerate(self.Xc):
            if entry in self.Xe:
                print('Found this value in XE!!')
                for enu1,test in enumerate(self.Xe):
                    # if entry[0] == test[0] and  entry[1] == test[1]:
                    if entry == test:
                        xe.append(test.tolist())
                        ye.append(self.ye[enu1].tolist())
                        xc.append(entry.tolist())
                        yc.append(self.yc[enu].tolist())
                        Xd.append(entry.tolist())
                        yd.append(self.ye[enu1].tolist()  - self.rho * self.yc[enu].tolist())
                        break

            else:
                xc.insert(0,entry.tolist())
                yc.insert(0,self.yc[enu].tolist())

        self.Xe = np.array(xe)
        self.ye = np.array(ye)
        self.Xc = np.array(xc)
        self.yc = np.array(yc)
        self.Xd = np.array(Xd)
        self.yd = np.atleast_2d(np.array(yd))


    def updateData(self):
        self.nc = self.Xc.shape[0]
        self.ne = self.Xe.shape[0]
        self.distanceXc()
        self.distanceXe()
        self.distanceXcXe()

    def traincheap(self):
        self.kc = kriging(self.Xc, self.yc)
        self.kc.train()
        print()


    def distanceXc(self):
        self.distanceXc = np.zeros((self.nc,self.nc, self.k))
        for i in range( self.nc ):
            for j in range(i+1,self.nc):
                self.distanceXc[i][j] = np.abs((self.Xc[i]-self.Xc[j]))

    def distanceXe(self):
        self.distanceXe = np.zeros((self.ne,self.ne, self.k))
        for i in range( self.ne ):
            for j in range(i+1,self.ne):
                self.distanceXe[i][j] = np.abs((self.Xe[i]-self.Xe[j]))

    def distanceXcXe(self):
        self.distanceXcXe = np.zeros((self.nc,self.ne, self.k))
        for i in range( self.nc ):
            for j in range(self.ne):
                self.distanceXcXe[i][j] = np.abs((self.Xc[i]-self.Xe[j]))


    def updatePsi(self):
        self.PsicXc = np.zeros((self.nc,self.nc), dtype=float)
        self.PsicXe = np.zeros((self.ne,self.ne), dtype=float)
        self.PsicXcXe = np.zeros((self.nc,self.ne), dtype=float)
        #
        # print self.thetac
        # print self.pc
        # print self.distanceXc
        newPsicXc = np.exp(-np.sum(self.thetac*np.power(self.distanceXc,self.pc), axis=2))
        print(newPsicXc[0])
        self.PsicXc = np.triu(newPsicXc,1)
        self.PsicXc = self.PsicXc + self.PsicXc.T + np.mat(eye(self.nc))+np.multiply(np.mat(eye(self.nc)),np.spacing(1))
        self.UPsicXc = np.linalg.cholesky(self.PsicXc)
        self.UPsicXc = self.UPsicXc.T
        print(self.PsicXc[0])
        print(self.UPsicXc)
        exit()

        newPsicXe = np.exp(-np.sum(self.thetac*np.power(self.distanceXe,self.pc), axis=2))
        self.PsicXe = np.triu(newPsicXe,1)
        self.PsiXe = self.PsicXe + self.PsicXe.T + np.mat(eye(self.ne))+np.multiply(np.mat(eye(self.ne)),np.spacing(1))
        self.UPsicXe = np.linalg.cholesky(self.PsicXe)
        self.UPsicXe = self.UPsicXe.T

        newPsiXeXc = np.exp(-np.sum(self.thetad*np.power(self.distanceXcXe,self.pd), axis=2))
        self.PsicXcXe = np.triu(newPsiXeXc,1)


    def neglnlikehood(self):
        a = np.linalg.solve(self.UPsicXc.T, np.matrix(self.yc).T)
        b = np.linalg.solve( self.UPsicXc, a )
        c = ones([self.nc,1]).T * b

        d = np.linalg.solve(self.UPsicXc.T, ones([self.nc,1]))
        e = np.linalg.solve(self.UPsicXc, d)
        f = ones([self.nc,1]).T * e

        self.muc = c/f
        # This only works if yc is transposed, then its a scalar under two layers of arrays. Correct? Not sure

        print('y',self.yd.T)
        a = np.linalg.solve(self.UPsicXe.T, self.yd)
        print('a',a)
        b = np.linalg.solve(self.UPsicXe, a)
        print('b', b)
        c = ones([self.ne,1]) * b
        print('c', c)

        d = np.linalg.solve(self.UPsicXe.T, ones([self.ne,1], dtype=float))
        print(d)

        e = np.linalg.solve(self.UPsicXe, d)
        print(e)

        f = ones([self.ne,1]).T * e
        print(f)

        self.mud= c/f


        a = np.linalg.solve(self.UPsicXc.T,(self.yc-ones([self.nc,1])*self.muc))/self.nc
        b = np.linalg.solve(self.UPsicXc, a)
        self.SigmaSqrc=(self.yc-ones([self.nc,1])*self.muc).T* b



        print(self.ne)
        print(self.mud)
        print(self.UPsicXe.T)
        a = np.linalg.solve(self.UPsicXe.T,(self.yd-ones([self.ne,1])*self.mud))/self.ne
        b = np.linalg.solve(self.UPsicXe, a)
        self.SigmaSqrd=(self.yd-ones([self.ne,1])*self.mud).T* b

        self.C=np.array([self.SigmaSqrc*self.PsicXc, self.rho*self.SigmaSqrc*self.PsicXcXe, self.rho*self.SigmaSqrc*self.PsicXeXc, np.power(self.rho,2)*self.SigmaSqrc*self.PsicXe+self.SigmaSqrd*self.PsidXe])
        np.reshape(c,[2,2])

        self.UC = np.linalg.cholesky(self.C)

        # self.mu=(self.one.T *(self.UC\(self.UC.T\y)))/(one'*(ModelInfo.UC\(ModelInfo.UC'\one)));



def fc(X):
    return np.power(X[:,0], 2) + X[:,0] + np.power(X[:,1], 2) + X[:,1]
def fe(X):
    return np.power(X[:,0], 2) + np.power(X[:,1], 2)

if __name__=='__main__':
    from . import samplingplan
    import random
    sp = samplingplan.samplingplan(2)
    X = sp.optimallhc(20)
    Xe = np.array( random.sample(X, 6) )

    yc = fc(X)
    ye = fe(Xe)

    ck = coKriging(X, yc, Xe, ye)
    ck.updateData()
    ck.updatePsi()
    ck.neglnlikehood()






