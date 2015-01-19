__author__ = 'chrispaulson'
import numpy as np
from matrixops import matrixops
import copy
from pyOpt import Optimization, ALPSO, NSGA2, SLSQP
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import copy

class regkriging(matrixops):
    def __init__(self, X, y, testfunction=None, name='', **kwargs):
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        self.testfunction = testfunction
        self.name = name
        self.n = self.X.shape[0]
        self.k = self.X.shape[1]
        self.theta = np.ones(self.k)
        self.pl = np.ones(self.k)
        self.sigma = 0
        self.normRange = []
        self.ynormRange = []
        self.normalizeData()
        self.Lambda = 1
        matrixops.__init__(self)

    def normX(self,X):
        for i in range(self.k):
            X[i] = (X[i] - self.normRange[i][0])/(self.normRange[i][1]-self.normRange[i][0])
        return X

    def inversenormX(self,X):
        for i in range(self.k):
            X[i] = (X[i] * (self.normRange[i][1] - self.normRange[i][0] )) + self.normRange[i][0]
        return X

    def normy(self, y):
        return (y - self.ynormRange[0])/(self.ynormRange[1]-self.ynormRange[0])

    def inversenormy(self, y):
        return (y * (self.ynormRange[1]-self.ynormRange[0])) + self.ynormRange[0]

    def normalizeData(self):
        for i in range(self.k):
            self.normRange.append( [ min(self.X[:,i]),  max(self.X[:,i])])

        for i in range(self.n):
            self.X[i] = self.normX(self.X[i])

        self.ynormRange.append(min(self.y))
        self.ynormRange.append(max(self.y))

        for i in range(self.n):
            self.y[i] = self.normy(self.y[i])


    def addPoint(self, newX,newy, norm=True):
        if norm:
            for i in range(self.k):
                newX = self.normX(newX)
            newy = self.normy(newy)

        self.X = np.append(self.X, [newX], axis=0)
        self.y = np.append(self.y,  newy)
        self.n = self.X.shape[0]
        try:
            self.regupdatePsi()
        except Exception, err:
            print Exception, err

    def fittingObjective(self, values):
        fail = 0
        f=10000
        for i in range(self.k):
            self.theta[i] = values[i]
        for i in range(self.k):
            self.pl[i] = values[i+(self.k)]
        self.Lambda = values[-1]
        try:
	        self.regupdatePsi()
        except Exception, err:
            print Exception, err
            print 'Failure in regupdatePsi, failing the entry'
            f = 1000
            fail = 1
        try:
            self.neglikelihood()
            f=self.NegLnLike
        except:
            print 'Failure in NegLNLike, failing the run'
            f = 1000
            fail = 1
        g = [-10.0]
        # print f,g,fail
        return f, g, fail

    def update(self, values):
        for i in range(self.k):
            self.theta[i] = values[i]
        for i in range(self.k):
            self.pl[i] = values[i+(self.k)]

        try:
	        self.regupdatePsi()
        except Exception, err:
            print Exception, err

    def updateModel(self):
        try:
	        self.regupdatePsi()
        except Exception, err:
            print Exception, err

    def predict(self, X):
        '''
        :param X: Design variable to evaluate
        :return: Returns the 'real world' predicted value
        '''
        X = self.normX(X)
        print X, self.predict_normalized(X), self.inversenormy(self.predict_normalized(X))
        return  self.inversenormy(self.predict_normalized(X))


    def errorObjective_normalized(self, values):
        fail = 0
        f = 1000
        try:
            f=-1*self.predicterr_normalized(values)
        except:
            print 'error prediction failed'
            fail = 1
        g = [-10.0]*3
        return f, g, fail

    def infill(self, points, method='error'):
        ## We'll be making non-permanent modifications to self.X and self.y here, so lets make a copy just in case
        initX = np.copy(self.X)
        inity = np.copy(self.y)

        ## This array will hold the new values we add
        returnValues = np.zeros([points,self.k],dtype=float)

        for i in range(points):
            opt_prob1 = Optimization('InFillPSO', self.errorObjective_normalized)
            for k in range(self.k):
                opt_prob1.addVar('{0}'.format(k),'c',lower=0,upper=1,value=.5)


            pso1 = ALPSO()
            pso1.setOption('SwarmSize',100)
            pso1.setOption('maxOuterIter',100)
            pso1.setOption('stopCriteria',1)
            pso1(opt_prob1)

            newpoint = np.zeros(self.k)

            for j in range(self.k):
                newpoint[j] = opt_prob1.solution(0)._variables[j].value
            returnValues[i][:] = self.inversenormX(newpoint)
            self.addPoint(returnValues[i], self.predict(returnValues[i]), norm=True)
            self.updateModel()
            del(opt_prob1)
            del(pso1)
        self.X = np.copy(initX)
        self.y = np.copy(inity)
        self.n = len(self.X)
        self.updateModel()
        return returnValues

    def train(self,optimizer='pso'):
        #Define the optimization problem for training the kriging model
        opt_prob = Optimization('Surrogate Test', self.fittingObjective)
        for i in range(self.k):
            opt_prob.addVar('theta%d'%i,'c',lower=1e-3,upper=1e2,value=.1)
        for i in range(self.k):
            opt_prob.addVar('pl%d'%i,'c',lower=1.5,upper=2,value=2)
        opt_prob.addVar('lambda','c',lower=1e-5,upper=1,value=1)
        opt_prob.addObj('f')
        opt_prob.addCon('g1', 'i')

        if optimizer=='pso':
            optimizer = ALPSO()
            optimizer.setOption('SwarmSize',150)
            optimizer.setOption('maxOuterIter',150)
            optimizer.setOption('stopCriteria',1)
            optimizer.setOption('filename', '{0}Results.log'.format(self.name))

        if optimizer=='ga':
            optimizer = NSGA2()
            optimizer.setOption('PopSize', (4*50))

        while True:
            try:
                self.trainingOptimizer(optimizer,opt_prob)
            except Exception as e:
                print e
                print 'Error traning Model, restarting the optimizer with a larger population'
                if optimizer=='ga':
                    optimizer.setOption('SwarmSize',200)
                    optimizer.setOption('maxOuterIter',100)
                    optimizer.setOption('stopCriteria',1)
                if optimizer=='ga':
                    optimizer.setOption('PopSize', 400)
            else:
                break


    def trainingOptimizer(self, optimizer, opt_prob):
        #Run the optimizer
        print 'running the global optimizer'
        optimizer(opt_prob)
        print 'done with global optimizer'
        print opt_prob.solution(0)
        for i in range(self.k):
            self.theta[i] = opt_prob.solution(0)._variables[i].value
        for i in range(self.k):
            self.pl[i] = opt_prob.solution(0)._variables[i+(self.k)].value
        print 'made it to lambda'
        self.Lambda = opt_prob.solution(0)._variables[2*self.k].value
        print self.Lambda
        print 'past lambda'
        del(optimizer)
        del(opt_prob)

    def plot(self):
        if self.k == 3:
            import mayavi.mlab as mlab
            predictFig = mlab.figure(figure='predict')
            errorFig = mlab.figure(figure='error')
            if self.testfunction:
                truthFig = mlab.figure(figure='test')
            dx = 1
            pts = 50j
            X,Y,Z = np.mgrid[0:dx:pts, 0:dx:pts, 0:dx:pts]
            scalars = np.zeros(X.shape)
            errscalars = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    for k1 in range(X.shape[2]):
                        errscalars[i][j][k1] = self.predicterr_normalized([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])
                        scalars[i][j][k1] = self.predict_normalized([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])

            if self.testfunction:
                tfscalars = np.zeros(X.shape)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        for k1 in range(X.shape[2]):
                            tfscalars[i][j][k1] = self.testfunction([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])
                mlab.contour3d(tfscalars, contours=15, transparent=True, figure=truthFig)


            # obj = mlab.contour3d(scalars, contours=10, transparent=True)
            mlab.contour3d(scalars, contours=15, transparent=True,figure=predictFig)
            mlab.contour3d(errscalars, contours=15, transparent=True,figure=errorFig)
            mlab.scatter()

            mlab.show()

        if self.k==2:
            samplePoints = zip(*self.X)
            # Create a set of data to plot
            plotgrid = 100
            x = np.linspace(0, 1, num=plotgrid)
            y = np.linspace(0, 1, num=plotgrid)
            X, Y = np.meshgrid(x, y)

            # Predict based on the optimized results

            zs = np.array([self.predict_normalized([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
            Z = zs.reshape(X.shape)
            Z = (Z*(self.ynormRange[1]-self.ynormRange[0]))+self.ynormRange[0]

            #Calculate errors
            zse = np.array([self.predicterr_normalized([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
            Ze = zse.reshape(X.shape)

            if self.testfunction:
                # Setup the truth function
                zt = self.testfunction( np.array(zip(np.ravel(X), np.ravel(Y))) )
                ZT = zt.reshape(X.shape)


            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(221)
            # contour_levels = np.linspace(min(zt), max(zt),50)
            contour_levels = 15
            CS = plt.contourf(X,Y,Z,contour_levels)
            plt.plot(samplePoints[0],samplePoints[1],'ow')

            if self.testfunction:
                CS = plt.contour(X,Y,ZT,contour_levels,colors='k')
            plt.plot(samplePoints[0],samplePoints[1], 'ow')
            plt.colorbar()

            ax = fig.add_subplot(222)
            CS = plt.contourf(X,Y,Ze,contour_levels)
            plt.colorbar()
            plt.plot(samplePoints[0],samplePoints[1],'ow')

            ax = fig.add_subplot(212, projection='3d')
            # fig = plt.gcf()
            #ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z, rstride=5, cstride=5, alpha=0.6)
            if self.testfunction:
                ax.plot_wireframe(X, Y, ZT, rstride=5, cstride=5)

            plt.show()

    def saveFigure(self, name=None):
                if self.k == 3:
                    import mayavi.mlab as mlab
                    mlab.options.offscreen = True
                    predictFig = mlab.figure(figure='predict')
                    mlab.clf(figure='predict')
                    errorFig = mlab.figure(figure='error')
                    mlab.clf(figure='error')
                    if self.testfunction:
                        truthFig = mlab.figure(figure='test')
                        mlab.clf(figure='test')
                    dx = 1
                    pts = 50j
                    X,Y,Z = np.mgrid[0:dx:pts, 0:dx:pts, 0:dx:pts]
                    scalars = np.zeros(X.shape)
                    errscalars = np.zeros(X.shape)
                    for i in range(X.shape[0]):
                        for j in range(X.shape[1]):
                            for k1 in range(X.shape[2]):
                                errscalars[i][j][k1] = self.predicterr_normalized([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])
                                scalars[i][j][k1] = self.predict_normalized([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])

                    if self.testfunction:
                        tfscalars = np.zeros(X.shape)
                        for i in range(X.shape[0]):
                            for j in range(X.shape[1]):
                                for k1 in range(X.shape[2]):
                                    tfscalars[i][j][k1] = self.testfunction([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])
                        mlab.contour3d(tfscalars, contours=15, transparent=True, figure=truthFig)


                    # obj = mlab.contour3d(scalars, contours=10, transparent=True)
                    mlab.contour3d(scalars, contours=15, transparent=True,figure=predictFig)
                    mlab.contour3d(errscalars, contours=15, transparent=True,figure=errorFig)
                    mlab.savefig('%s_prediction.wrl'%name, figure=predictFig)
                    mlab.savefig('%s_error.wrl'%name, figure=errorFig)
                    if self.testfunction:
                        mlab.savefig('%s_actual.wrl'%name, figure=truthFig)
                    mlab.close(all=True)
