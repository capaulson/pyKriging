from __future__ import division, absolute_import
import numpy as np
from numpy.matlib import rand,zeros,ones,empty,eye
import scipy
class matrixops():

    def __init__(self):
        self.LnDetPsi = None
        self.Psi = np.zeros((self.n,self.n), dtype=np.float)
        self.psi = np.zeros((self.n,1))
        self.one = np.ones(self.n)
        self.mu = None
        self.U = None
        self.SigmaSqr = None
        self.Lambda = 1
        self.updateData()

    def updateData(self):
        self.distance = np.zeros((self.n,self.n, self.k))
        for i in xrange(self.n):
            for j in xrange(i+1,self.n):
                self.distance[i,j]= np.abs((self.X[i]-self.X[j]))

    def updatePsi(self):
        self.Psi = np.zeros((self.n,self.n), dtype=np.float)
        self.one = np.ones(self.n)
        self.psi = np.zeros((self.n,1))
        # self.updateData()
        newPsi = np.exp(-np.sum(self.theta*np.power(self.distance,self.pl), axis=2))
        self.Psi = np.triu(newPsi,1)
        # #
        # for i in xrange(self.n):
        #     for j in xrange(i+1,self.n):
        #         self.Psi[i,j]=np.exp(-np.sum(self.theta*np.power(np.abs((self.X[i]-self.X[j])),self.pl)))
        self.Psi = self.Psi + self.Psi.T + np.mat(eye(self.n))+np.multiply(np.mat(eye(self.n)),np.spacing(1))
        self.U = np.linalg.cholesky(self.Psi)
        # self.U = np.matrix(self.U.T)
        self.U = self.U.T

    ## The regression case, uses self.lambda instad of
    def regupdatePsi(self):
        self.Psi = np.zeros((self.n,self.n), dtype=np.float)
        self.one = np.ones(self.n)
        for i in xrange(self.n):
            for j in xrange(i+1,self.n):
                self.Psi[i,j]=np.exp(-np.sum(self.theta*np.power(np.abs((self.X[i]-self.X[j])),self.pl)))
        self.Psi = self.Psi + self.Psi.T + eye(self.n) + eye(self.n) * (self.Lambda)
        self.U = np.linalg.cholesky(self.Psi)
        self.U = np.matrix(self.U.T)

    def corr(self, x1, x2, theta, pl):
        return np.exp(-1*np.sum(theta*np.abs(x1-x2)**pl))

    def corr1(self, theta, pl):
        return np.exp(-np.sum(theta*self.distance**pl, axis=2))

    def neglikelihood(self):
        self.LnDetPsi=2*np.sum(np.log(np.abs(np.diag(self.U))))

        a = np.linalg.solve(self.U.T, self.one.T)
        b = np.linalg.solve(self.U, a)
        c = self.one.T.dot(b)
        d = np.linalg.solve(self.U.T, self.y)
        e = np.linalg.solve(self.U, d)
        self.mu=(self.one.T.dot(e))/c

        self.SigmaSqr = ((self.y-self.one.dot(self.mu)).T.dot(np.linalg.solve(self.U,np.linalg.solve(self.U.T,(self.y-self.one.dot(self.mu))))))/self.n
        self.NegLnLike=-1.*(-(self.n/2.)*np.log(self.SigmaSqr) - 0.5*self.LnDetPsi)

    def regneglikelihood(self):
        self.LnDetPsi=2*np.sum(np.log(np.abs(np.diag(self.U))))

        mu=(self.one.T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T, self.y))))/self.one.T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T, self.one)))
        self.mu=mu

        self.SigmaSqr = ((self.y-self.one.dot(self.mu)).T.dot(np.linalg.solve(self.U,np.linalg.solve(self.U.T,(self.y-self.one.dot(self.mu))))))/self.n

        self.NegLnLike=-1.*(-(self.n/2.)*np.log(self.SigmaSqr) - 0.5*self.LnDetPsi)

    def predict_normalized(self,x):
        for i in range(self.n):
            self.psi[i]=np.exp(-np.sum(self.theta*np.power((np.abs(self.X[i]-x)),self.pl)))
        z = self.y-self.one.dot(self.mu)
        a = np.linalg.solve(self.U.T, z)
        b=np.linalg.solve(self.U, a)
        c=self.psi.T.dot(b)

        f=self.mu + c
        return f[0]

    def predicterr_normalized(self,x):
        for i in range(self.n):
            try:
                self.psi[i]=np.exp(-np.sum(self.theta*np.power((np.abs(self.X[i]-x)),self.pl)))
            except Exception,e:
                print Exception,e
        try:
            SSqr=self.SigmaSqr*(1-self.psi.T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T,self.psi))))
        except Exception, e:
            print Exception,e
            pass

            # print 'X is: ', self.X
            # print 'x is: ', x
            # print 'size of u is:', np.size(self.U)
            # print 'u is: ', self.U
            # print 'psi is: ', self.psi
        # SSqr=SigmaSqr*(1-psi'*(U\(U'\psi)));
        SSqr = np.abs(SSqr[0])
        return np.power(SSqr,0.5)[0]