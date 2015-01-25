from __future__ import division
__author__ = 'chrispaulson'
import numpy as np
from matrixops import matrixops
import copy
# from pyOpt import Optimization, ALPSO, NSGA2, SLSQP, MIDACO
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from pyKriging import samplingplan
import inspyred
from random import Random
from time import time
from inspyred import ec
from inspyred.ec import terminators

class kriging(matrixops):
    def __init__(self, X, y, testfunction=None, name='', **kwargs):
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        self.testfunction = testfunction
        self.name = name
        self.n = self.X.shape[0]
        self.k = self.X.shape[1]
        self.theta = np.ones(self.k)
        self.pl = np.ones(self.k)*2.
        self.sigma = 0
        self.normRange = []
        self.ynormRange = []
        self.normalizeData()
        self.sp = samplingplan.samplingplan()
        self.updateData()
        self.updateModel()

        self.thetamin = 1e-4
        self.thetamax = 100
        self.pmin = 1
        self.pmax = 2

        matrixops.__init__(self)
    
    def normX(self,X):
        X = copy.deepcopy(X)
        for i in range(self.k):
            X[i] = (X[i] - self.normRange[i][0])/float(self.normRange[i][1]-self.normRange[i][0])
        return X

    def inversenormX(self,X):
        X= copy.deepcopy(X)
        for i in range(self.k):
            X[i] = (X[i] * float(self.normRange[i][1] - self.normRange[i][0] )) + self.normRange[i][0]
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
            newX = self.normX(newX)
            newy = self.normy(newy)

        self.X = np.append(self.X, [newX], axis=0)
        self.y = np.append(self.y,  newy)
        self.n = self.X.shape[0]
        self.updateData()
        self.updateModel()

    # def fittingObjective(self, values):
        #fittingObjective for pyOpt verison
    #     fail = 0
    #     f=10000
    #     for i in range(self.k):
    #         self.theta[i] = values[i]
    #     for i in range(self.k):
    #         self.pl[i] = values[i+(self.k)]
    #     try:
	 #        self.updatePsi()
    #     except Exception, err:
    #         # print Exception, err
    #         #print 'Failure in updatePsi, failing the entry'
    #         f = 1000
    #         fail = 1
    #     try:
    #         self.neglikelihood()
    #         f=self.NegLnLike
    #     except Exception,e:
    #         print 'Failure in NegLNLike, failing the run'
    #         print Exception,e
    #         f = 1000
    #         fail = 1
    #     g = [-10.0]
    #     return f, g, fail

    def fittingObjective(self,candidates, args):
        fitness = []
        for entry in candidates:
            f=10000
            for i in range(self.k):
                self.theta[i] = entry[i]
            for i in range(self.k):
                self.pl[i] = entry[i+(self.k)]
            try:
                self.updatePsi()
            except Exception, err:
                # print Exception, err
                # print 'Failure in updatePsi, failing the entry'
                f = 1000
            try:
                self.neglikelihood()
                f=self.NegLnLike
            except Exception,e:
                print 'Failure in NegLNLike, failing the run'
                print Exception,e
                f = 1000
            fitness.append(f)
        return fitness

    def generate_population(self,random, args):
        size = args.get('num_inputs', None)
        bounder = args["_ec"].bounder
        chromosome = []
        for lo, hi in zip(bounder.lower_bound, bounder.upper_bound):
            chromosome.append(random.uniform(lo, hi))
        return chromosome

    def update(self, values):
        for i in range(self.k):
            self.theta[i] = values[i]
        for i in range(self.k):
            self.pl[i] = values[i+(self.k)]
        self.updateModel()

    def updateModel(self):
        try:
	        self.updatePsi()
        except Exception, err:
            print Exception, err

    def predict(self, X):
        '''
        :param X: Design variable to evaluate
        :return: Returns the 'real world' predicted value
        '''
        X = copy.deepcopy(X)
        X = self.normX(X)
        return  self.inversenormy(self.predict_normalized(X))

    def predict_var(self, X):
        '''
        :param x: new design variable to evaluate 
        :return: Returns the posterior variance
        '''
        X = copy.deepcopy(X)
        X = self.normX(X)
        # print X, self.predict_normalized(X), self.inversenormy(self.predict_normalized(X))
        return self.predicterr_normalized(X)

    def infill_objective(self,candidates, args):
        fitness = []
        for entry in candidates:
            fitness.append(-1*self.predicterr_normalized(entry))
        return fitness

    def infill(self, points, method='error'):
        ## We'll be making non-permanent modifications to self.X and self.y here, so lets make a copy just in case
        initX = np.copy(self.X)
        inity = np.copy(self.y)

        ## This array will hold the new values we add
        returnValues = np.zeros([points,self.k],dtype=float)
        for i in range(points):
            rand = Random()
            rand.seed(int(time()))
            ea = inspyred.swarm.PSO(Random())
            ea.terminator = terminators.evaluation_termination
            ea.topology = inspyred.swarm.topologies.ring_topology
            final_pop = ea.evolve(generator=self.generate_population,
                                  evaluator=self.infill_objective,
                                  pop_size=100,
                                  maximize=False,
                                  bounder=ec.Bounder([0]*self.k,[1]*self.k),
                                  max_evaluations=3000,
                                  neighborhood_size=10,
                                  num_inputs=self.k)
            final_pop.sort(reverse=True)
            newpoint = final_pop[0].candidate
            returnValues[i][:] = self.inversenormX(newpoint)
            self.addPoint(returnValues[i], self.predict(returnValues[i]), norm=True)

        self.X = np.copy(initX)
        self.y = np.copy(inity)
        self.n = len(self.X)
        self.updateData()
        self.updateModel()
        return returnValues

    def train(self):
        self.updateData()
        self.updateModel()
        lowerBound = [self.thetamin]*self.k + [self.pmin]*self.k
        upperBound = [self.thetamax]*self.k + [self.pmax]*self.k
        rand = Random()
        rand.seed(int(time()))
        ea = inspyred.swarm.PSO(Random())
        ea.terminator = terminators.average_fitness_termination
        ea.topology = inspyred.swarm.topologies.star_topology
        ea.observer = inspyred.ec.observers.stats_observer
        final_pop = ea.evolve(generator=self.generate_population,
                              evaluator=self.fittingObjective,
                              pop_size=50,
                              maximize=False,
                              bounder=ec.Bounder(lowerBound, upperBound),
                              max_evaluations=20000,
                              neighborhood_size=25,
                              num_inputs=self.k,
                              tolerance=0.01,
                              min_diversity=.01)
        # Sort and print the best individual, who will be at index 0.
        final_pop.sort(reverse=True)
        newValues = final_pop[0].candidate
        for i in range(self.k):
            self.theta[i] = newValues[i]
        for i in range(self.k):
            self.pl[i] = newValues[i+(self.k)]
        self.updateModel()

    def plot(self, labels=False):
        if self.k == 3:
            import mayavi.mlab as mlab
            predictFig = mlab.figure(figure='predict')
            errorFig = mlab.figure(figure='error')
            if self.testfunction:
                truthFig = mlab.figure(figure='test')
            dx = 1
            pts = 25j
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
                plot = mlab.contour3d(tfscalars, contours=15, transparent=True, figure=truthFig)
                plot.compute_normals = False


            # obj = mlab.contour3d(scalars, contours=10, transparent=True)
            plot = mlab.contour3d(scalars, contours=15, transparent=True,figure=predictFig)
            plot.compute_normals = False
            errplt = mlab.contour3d(errscalars, contours=15, transparent=True,figure=errorFig)
            errplt.compute_normals = False
            mlab.show()

        if self.k==2:
            samplePoints = zip(*self.X)
            # Create a set of data to plot
            plotgrid = 61
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
                zt = self.testfunction( np.array(zip(np.ravel((X * (self.normRange[0][1] - self.normRange[0][0])) + self.normRange[0][0]), np.ravel((Y * (self.normRange[1][1] - self.normRange[1][0])) + self.normRange[1][0]))) )
                ZT = zt.reshape(X.shape)

            #Plot real world values
            X = (X * (self.normRange[0][1] - self.normRange[0][0])) + self.normRange[0][0]
            Y = (Y * (self.normRange[1][1] - self.normRange[1][0])) + self.normRange[1][0]
            spx = (self.X[:,0] * (self.normRange[0][1] - self.normRange[0][0])) + self.normRange[0][0]
            spy = (self.X[:,1] * (self.normRange[1][1] - self.normRange[1][0])) + self.normRange[1][0]
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(221)
            # contour_levels = np.linspace(min(zt), max(zt),50)
            contour_levels = 15
            CS = plt.contourf(X,Y,Z,contour_levels)
            plt.plot(spx, spy,'ow')
            plt.colorbar()

            if self.testfunction:
                CS = plt.contour(X,Y,ZT,contour_levels,colors='k')
            plt.plot(spx, spy,'ow')

            ax = fig.add_subplot(222)
            CS = plt.contourf(X,Y,Ze, contour_levels)
            plt.colorbar()
            plt.plot(spx, spy,'ow')

            ax = fig.add_subplot(212, projection='3d')
            # fig = plt.gcf()
            #ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.4)
            if self.testfunction:
                ax.plot_wireframe(X, Y, ZT, rstride=3, cstride=3)

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
                    pts = 75j
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
                        mlab.contour3d(tfscalars, contours=15, transparent=True, figure=truthFig,compute_normals=False)


                    # obj = mlab.contour3d(scalars, contours=10, transparent=True)
                    pred = mlab.contour3d(scalars, contours=15, transparent=True,figure=predictFig)
                    pred.compute_normals = False
                    errpred = mlab.contour3d(errscalars, contours=15, transparent=True,figure=errorFig)
                    errpred.compute_normals = False
                    mlab.savefig('%s_prediction.wrl'%name, figure=predictFig)
                    mlab.savefig('%s_error.wrl'%name, figure=errorFig)
                    if self.testfunction:
                        mlab.savefig('%s_actual.wrl'%name, figure=truthFig)
                    mlab.close(all=True)

    def calcuatemeanMSE(self, p2s=200, points=None):
        if points==None:
            points = self.sp.rlh(p2s)
        values = np.zeros(len(points))
        for enu,point in enumerate(points):
            values[enu] = self.predict_var(point)
        return np.mean(values), np.std(values)


