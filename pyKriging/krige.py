from __future__ import division

__author__ = 'chrispaulson'
import numpy as np
import scipy
from scipy.optimize import minimize
from matrixops import matrixops
import copy
from matplotlib import pyplot as plt
import pylab

from mpl_toolkits.mplot3d import axes3d
from pyKriging import samplingplan
import inspyred
from random import Random
from time import time
from inspyred import ec
import math as m


class kriging(matrixops):
    def __init__(self, X, y, testfunction=None, name='', testPoints=None, **kwargs):
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        self.testfunction = testfunction
        self.name = name
        self.n = self.X.shape[0]
        self.k = self.X.shape[1]
        self.theta = np.ones(self.k)
        self.pl = np.ones(self.k) * 2.
        self.sigma = 0
        self.normRange = []
        self.ynormRange = []
        self.normalizeData()
        self.sp = samplingplan.samplingplan(self.k)
        self.updateData()
        self.updateModel()

        self.thetamin = 1e-4
        self.thetamax = 100
        self.pmin = 1
        self.pmax = 2

        # Setup functions for tracking history
        self.history = {}
        self.history['points'] = []
        self.history['neglnlike'] = []
        self.history['theta'] = []
        self.history['p'] = []
        self.history['rsquared'] = [0]
        self.history['adjrsquared'] = [0]
        self.history['chisquared'] = [1000]
        self.history['lastPredictedPoints'] = []
        if testPoints:
            self.history['pointData'] = []
            self.testPoints = self.sp.rlh(testPoints)

            for point in self.testPoints:
                testPrimitive = {}
                testPrimitive['point'] = point
                if self.testfunction:
                    testPrimitive['actual'] = self.testfunction(point)[0]
                else:
                    testPrimitive['actual'] = None
                testPrimitive['predicted'] = []
                testPrimitive['mse'] = []
                testPrimitive['gradient'] = []
                self.history['pointData'].append(testPrimitive)

        else:
            self.history['pointData'] = None


        matrixops.__init__(self)

    def normX(self, X):
        X = copy.deepcopy(X)
        for i in range(self.k):
            X[i] = (X[i] - self.normRange[i][0]) / float(self.normRange[i][1] - self.normRange[i][0])
        return X

    def inversenormX(self, X):
        X = copy.deepcopy(X)
        for i in range(self.k):
            X[i] = (X[i] * float(self.normRange[i][1] - self.normRange[i][0] )) + self.normRange[i][0]
        return X

    def normy(self, y):
        return (y - self.ynormRange[0]) / (self.ynormRange[1] - self.ynormRange[0])

    def inversenormy(self, y):
        return (y * (self.ynormRange[1] - self.ynormRange[0])) + self.ynormRange[0]

    def normalizeData(self):
        for i in range(self.k):
            self.normRange.append([min(self.X[:, i]), max(self.X[:, i])])

        for i in range(self.n):
            self.X[i] = self.normX(self.X[i])

        self.ynormRange.append(min(self.y))
        self.ynormRange.append(max(self.y))

        for i in range(self.n):
            self.y[i] = self.normy(self.y[i])

    def addPoint(self, newX, newy, norm=True):
        if norm:
            newX = self.normX(newX)
            newy = self.normy(newy)

        self.X = np.append(self.X, [newX], axis=0)
        self.y = np.append(self.y, newy)
        self.n = self.X.shape[0]
        self.updateData()
        while True:
            try:
                self.updateModel()
            except:
                self.train()
            else:
                break

    def generate_population(self, random, args):
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
            self.pl[i] = values[i + self.k]
        self.updateModel()

    def updateModel(self):
        try:
            self.updatePsi()
        except Exception, err:
            # pass
            # print Exception, err
            raise Exception("bad params")

    def predict(self, X):
        '''
        :param X: Design variable to evaluate
        :return: Returns the 'real world' predicted value
        '''
        X = copy.deepcopy(X)
        X = self.normX(X)
        return self.inversenormy(self.predict_normalized(X))

    def predict_var(self, X):
        '''
        :param X: new design variable to evaluate
        :return: Returns the posterior variance
        '''
        X = copy.deepcopy(X)
        X = self.normX(X)
        # print X, self.predict_normalized(X), self.inversenormy(self.predict_normalized(X))
        return self.predicterr_normalized(X)

    def expimp(self, x):
        S = self.predicterr_normalized(x)
        y_min = np.min(self.y)
        if S <= 0.:
            EI = 0.
        elif S > 0.:
            EI_one = ((y_min - self.predict_normalized(x)) * (0.5 + 0.5*m.erf((
                      1./np.sqrt(2.))*((y_min - self.predict_normalized(x)) /
                                       S))))
            EI_two = ((S * (1. / np.sqrt(2. * np.pi))) * (np.exp(-(1./2.) *
                      ((y_min - self.predict_normalized(x))**2. / S**2.))))
            EI = EI_one + EI_two
        return EI

    def weightedexpimp(self, x, w):
        """weighted expected improvement (Sobester et al. 2005)"""
        S = self.predicterr_normalized(x)
        y_min = np.min(self.y)
        if S <= 0.:
            EI = 0.
        elif S > 0.:
            EI_one = w*((y_min - self.predict_normalized(x)) * (0.5 +
                        0.5*m.erf((1./np.sqrt(2.))*((y_min -
                                  self.predict_normalized(x)) / S))))
            EI_two = ((1. - w)*(S * (1. / np.sqrt(2. * np.pi))) *
                      (np.exp(-(1./2.) * ((y_min -
                       self.predict_normalized(x))**2. / S**2.))))
            EI = EI_one + EI_two
        return EI

    def infill_objective_mse(self,candidates, args):
        fitness = []
        for entry in candidates:
            fitness.append(-1 * self.predicterr_normalized(entry))
        return fitness

    def infill_objective_ei(self,candidates, args):
        fitness = []
        for entry in candidates:
            fitness.append(-1 * self.expimp(entry))
        return fitness

    def infill(self, points, method='error'):
        # We'll be making non-permanent modifications to self.X and self.y here, so lets make a copy just in case
        initX = np.copy(self.X)
        inity = np.copy(self.y)

        # This array will hold the new values we add
        returnValues = np.zeros([points, self.k], dtype=float)
        for i in range(points):
            rand = Random()
            rand.seed(int(time()))
            ea = inspyred.swarm.PSO(Random())
            ea.terminator = self.no_improvement_termination
            ea.topology = inspyred.swarm.topologies.ring_topology
            if method=='ei':
                evaluator = self.infill_objective_ei
            else:
                evaluator = self.infill_objective_mse

            final_pop = ea.evolve(generator=self.generate_population,
                                  evaluator=evaluator,
                                  pop_size=155,
                                  maximize=False,
                                  bounder=ec.Bounder([0] * self.k, [1] * self.k),
                                  max_evaluations=20000,
                                  neighborhood_size=30,
                                  num_inputs=self.k)
            final_pop.sort(reverse=True)
            newpoint = final_pop[0].candidate
            returnValues[i][:] = newpoint
            self.addPoint(returnValues[i], self.predict(returnValues[i]), norm=True)

        self.X = np.copy(initX)
        self.y = np.copy(inity)
        self.n = len(self.X)
        self.updateData()
        while True:
            try:
                self.updateModel()
            except:
                self.train()
            else:
                break
        return returnValues

    def no_improvement_termination(self, population, num_generations, num_evaluations, args):
        """Return True if the best fitness does not change for a number of generations of if the max number
        of evaluations is exceeded.

        This function keeps track of the current best fitness and compares it to
        the best fitness in previous generations. Whenever those values are the
        same, it begins a generation count. If that count exceeds a specified
        number, the terminator returns True.

        .. Arguments:
           population -- the population of Individuals
           num_generations -- the number of elapsed generations
           num_evaluations -- the number of candidate solution evaluations
           args -- a dictionary of keyword arguments

        Optional keyword arguments in args:

        - *max_generations* -- the number of generations allowed for no change in fitness (default 10)

        """
        max_generations = args.setdefault('max_generations', 10)
        previous_best = args.setdefault('previous_best', None)
        max_evaluations = args.setdefault('max_evaluations', 30000)
        current_best = np.around(max(population).fitness, decimals=4)
        if previous_best is None or previous_best != current_best:
            args['previous_best'] = current_best
            args['generation_count'] = 0
            return False or (num_evaluations >= max_evaluations)
        else:
            if args['generation_count'] >= max_generations:
                return True
            else:
                args['generation_count'] += 1
                return False or (num_evaluations >= max_evaluations)

    def train(self, optimizer='pso'):
        self.updateData()
        # self.updateModel()
        lowerBound = [self.thetamin] * self.k + [self.pmin] * self.k
        upperBound = [self.thetamax] * self.k + [self.pmax] * self.k
        rand = Random()
        rand.seed(int(time()))
        if optimizer is 'pso':
            ea = inspyred.swarm.PSO(Random())
            ea.terminator = self.no_improvement_termination
            ea.topology = inspyred.swarm.topologies.ring_topology
            # ea.observer = inspyred.ec.observers.stats_observer
            final_pop = ea.evolve(generator=self.generate_population,
                                  evaluator=self.fittingObjective,
                                  pop_size=300,
                                  maximize=False,
                                  bounder=ec.Bounder(lowerBound, upperBound),
                                  max_evaluations=30000,
                                  neighborhood_size=20,
                                  num_inputs=self.k)
            # Sort and print the best individual, who will be at index 0.
            final_pop.sort(reverse=True)
        elif optimizer is 'ga':
            ea = inspyred.ec.GA(Random())
            ea.terminator = self.no_improvement_termination
            final_pop = ea.evolve(generator=self.generate_population,
                                  evaluator=self.fittingObjective,
                                  pop_size=300,
                                  maximize=False,
                                  bounder=ec.Bounder(lowerBound, upperBound),
                                  max_evaluations=30000,
                                  num_elites=10,
                                  mutation_rate=.05)
        for entry in final_pop:
            newValues = entry.candidate
            preLOP = copy.deepcopy(newValues)
            locOP_bounds = []
            for i in range(self.k):
                locOP_bounds.append( [self.thetamin, self.thetamax] )

            for i in range(self.k):
                locOP_bounds.append( [self.pmin, self.pmax] )

            lopResults = minimize(self.fittingObjective_local, newValues, method='SLSQP', bounds=locOP_bounds, options={'disp': False})

            newValues = lopResults['x']

            for i in range(self.k):
                self.theta[i] = newValues[i]
            for i in range(self.k):
                self.pl[i] = newValues[i + self.k]
            try:
                self.updateModel()
            except:
                pass
            else:
                break

    def fittingObjective(self,candidates, args):
        fitness = []
        for entry in candidates:
            f=10000
            for i in range(self.k):
                self.theta[i] = entry[i]
            for i in range(self.k):
                self.pl[i] = entry[i + self.k]
            try:
                self.updateModel()
                self.neglikelihood()
                f = self.NegLnLike
            except Exception, e:
                # print 'Failure in NegLNLike, failing the run'
                # print Exception, e
                f = 10000
            fitness.append(f)
        return fitness

    def fittingObjective_local(self,entry):
        f=10000
        for i in range(self.k):
            self.theta[i] = entry[i]
        for i in range(self.k):
            self.pl[i] = entry[i + self.k]
        try:
            self.updateModel()
            self.neglikelihood()
            f = self.NegLnLike
        except Exception, e:
            # print 'Failure in NegLNLike, failing the run'
            # print Exception, e
            f = 10000
        return f

    def plot(self, labels=False, show=True):
        if self.k == 3:
            import mayavi.mlab as mlab

            predictFig = mlab.figure(figure='predict')
            errorFig = mlab.figure(figure='error')
            if self.testfunction:
                truthFig = mlab.figure(figure='test')
            dx = 1
            pts = 25j
            X, Y, Z = np.mgrid[0:dx:pts, 0:dx:pts, 0:dx:pts]
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
                            tfplot = tfscalars[i][j][k1] = self.testfunction([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])
                plot = mlab.contour3d(tfscalars, contours=15, transparent=True, figure=truthFig)
                plot.compute_normals = False

            # obj = mlab.contour3d(scalars, contours=10, transparent=True)
            plot = mlab.contour3d(scalars, contours=15, transparent=True, figure=predictFig)
            plot.compute_normals = False
            errplt = mlab.contour3d(errscalars, contours=15, transparent=True, figure=errorFig)
            errplt.compute_normals = False
            if show:
                mlab.show()

        if self.k==2:
            fig = pylab.figure(figsize=(8,6))
            samplePoints = zip(*self.X)
            # Create a set of data to plot
            plotgrid = 61
            x = np.linspace(self.normRange[0][0], self.normRange[0][1], num=plotgrid)
            y = np.linspace(self.normRange[1][0], self.normRange[1][1], num=plotgrid)

            x = np.linspace(0, 1, num=plotgrid)
            y = np.linspace(0, 1, num=plotgrid)
            X, Y = np.meshgrid(x, y)

            # Predict based on the optimized results

            zs = np.array([self.predict([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
            Z = zs.reshape(X.shape)
            # Z = (Z*(self.ynormRange[1]-self.ynormRange[0]))+self.ynormRange[0]

            #Calculate errors
            zse = np.array([self.predict_var([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
            Ze = zse.reshape(X.shape)

            spx = (self.X[:,0] * (self.normRange[0][1] - self.normRange[0][0])) + self.normRange[0][0]
            spy = (self.X[:,1] * (self.normRange[1][1] - self.normRange[1][0])) + self.normRange[1][0]
            contour_levels = 25

            ax = fig.add_subplot(222)
            CS = pylab.contourf(X,Y,Ze, contour_levels)
            pylab.colorbar()
            pylab.plot(spx, spy,'ow')

            ax = fig.add_subplot(221)
            if self.testfunction:
                # Setup the truth function
                zt = self.testfunction( np.array(zip(np.ravel(X), np.ravel(Y))) )
                ZT = zt.reshape(X.shape)
                CS = pylab.contour(X,Y,ZT,contour_levels ,colors='k',zorder=2)


            # contour_levels = np.linspace(min(zt), max(zt),50)
            if self.testfunction:
                contour_levels = CS.levels
                delta = np.abs(contour_levels[0]-contour_levels[1])
                contour_levels = np.insert(contour_levels, 0, contour_levels[0]-delta)
                contour_levels = np.append(contour_levels, contour_levels[-1]+delta)

            CS = plt.contourf(X,Y,Z,contour_levels,zorder=1)
            pylab.plot(spx, spy,'ow', zorder=3)
            pylab.colorbar()

            ax = fig.add_subplot(212, projection='3d')
            # fig = plt.gcf()
            #ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.4)
            if self.testfunction:
                ax.plot_wireframe(X, Y, ZT, rstride=3, cstride=3)
            if show:
                pylab.show()

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
            X, Y, Z = np.mgrid[0:dx:pts, 0:dx:pts, 0:dx:pts]
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
                mlab.contour3d(tfscalars, contours=15, transparent=True, figure=truthFig, compute_normals=False)

            # obj = mlab.contour3d(scalars, contours=10, transparent=True)
            pred = mlab.contour3d(scalars, contours=15, transparent=True, figure=predictFig)
            pred.compute_normals = False
            errpred = mlab.contour3d(errscalars, contours=15, transparent=True, figure=errorFig)
            errpred.compute_normals = False
            mlab.savefig('%s_prediction.wrl' % name, figure=predictFig)
            mlab.savefig('%s_error.wrl' % name, figure=errorFig)
            if self.testfunction:
                mlab.savefig('%s_actual.wrl' % name, figure=truthFig)
            mlab.close(all=True)
        if self.k == 2:
            samplePoints = zip(*self.X)
            # Create a set of data to plot
            plotgrid = 61
            x = np.linspace(0, 1, num=plotgrid)
            y = np.linspace(0, 1, num=plotgrid)
            X, Y = np.meshgrid(x, y)

            # Predict based on the optimized results

            zs = np.array([self.predict_normalized([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
            Z = zs.reshape(X.shape)
            Z = (Z * (self.ynormRange[1] - self.ynormRange[0])) + self.ynormRange[0]

            # Calculate errors
            zse = np.array([self.predicterr_normalized([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
            Ze = zse.reshape(X.shape)

            if self.testfunction:
                # Setup the truth function
                zt = self.testfunction(np.array(
                    zip(np.ravel((X * (self.normRange[0][1] - self.normRange[0][0])) + self.normRange[0][0]),
                        np.ravel((Y * (self.normRange[1][1] - self.normRange[1][0])) + self.normRange[1][0]))))
                ZT = zt.reshape(X.shape)

            # Plot real world values
            X = (X * (self.normRange[0][1] - self.normRange[0][0])) + self.normRange[0][0]
            Y = (Y * (self.normRange[1][1] - self.normRange[1][0])) + self.normRange[1][0]
            spx = (self.X[:, 0] * (self.normRange[0][1] - self.normRange[0][0])) + self.normRange[0][0]
            spy = (self.X[:, 1] * (self.normRange[1][1] - self.normRange[1][0])) + self.normRange[1][0]
            fig = plt.figure(figsize=(8, 6))
            # contour_levels = np.linspace(min(zt), max(zt),50)
            contour_levels = 15
            plt.plot(spx, spy, 'ow')
            cs = plt.colorbar()

            if self.testfunction:
                pass
            plt.plot(spx, spy, 'ow')

            cs = plt.colorbar()
            plt.plot(spx, spy, 'ow')

            ax = fig.add_subplot(212, projection='3d')
            ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.4)

            if self.testfunction:
                ax.plot_wireframe(X, Y, ZT, rstride=3, cstride=3)
        if name:
            plt.savefig(name)
        else:
            plt.savefig('pyKrigingResult.png')

    def calcuatemeanMSE(self, p2s=200, points=None):
        if points is None:
            points = self.sp.rlh(p2s)
        values = np.zeros(len(points))
        for enu, point in enumerate(points):
            values[enu] = self.predict_var(point)
        return np.mean(values), np.std(values)

    def snapshot(self):
        self.history['points'].append(self.n)
        self.history['neglnlike'].append(self.NegLnLike)
        self.history['theta'].append(copy.deepcopy(self.theta))
        self.history['p'].append(copy.deepcopy(self.pl))

        currentPredictions = []
        if self.history['pointData']!=None:
            for pointprim in self.history['pointData']:
                predictedPoint = self.predict(pointprim['point'])
                currentPredictions.append(copy.deepcopy( predictedPoint) )

                pointprim['predicted'].append( predictedPoint )
                pointprim['mse'].append( self.predict_var(pointprim['point']) )
                try:
                    pointprim['gradient'] = np.gradient( pointprim['predicted'] )
                except:
                    pass
        if self.history['lastPredictedPoints'] != []:
            self.history['chisquared'].append( self.chisquared(  self.history['lastPredictedPoints'], currentPredictions  ) )
            self.history['rsquared'].append( self.rsquared( self.history['lastPredictedPoints'], currentPredictions ) )
            self.history['adjrsquared'].append( self.adjrsquares( self.history['rsquared'][-1], len( self.history['pointData'] )  ) )
        self.history[ 'lastPredictedPoints' ] = copy.deepcopy(currentPredictions)

    def rsquared(self,actual, observed):
        return np.corrcoef(observed, actual)[0,1] ** 2

    def adjrsquares(self, rsquared, obs):
        return 1-(1-rsquared)*((obs-1)/(obs-self.k))   # adjusted R-square


    def chisquared(self, actual, observed):
        actual = np.array(actual)
        observed = np.array(observed)
        return np.sum( np.abs( np.power( (observed-actual)  ,2)/actual ) )