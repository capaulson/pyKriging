"""
@author: Giorgos
"""
import numpy as np
from matplotlib import pyplot as plt
import pyKriging
from pyKriging.krige import kriging
from pyKriging.utilities import *
import random
import scipy.stats as stats


class Cross_Validation():

    def __init__(self, model, name=None):
        """
        X- sampling plane
        y- Objective function evaluations
        name- the name of the model
        """
        self.model = model
        self.X = self.model.X
        self.y = self.model.y
        self.n, self.k = np.shape(self.X)
        self.predict_list, self.predict_varr, self.scvr = [], [], []
        self.name = name

    def calculate_RMSE_Rsquared(self, optimiser, nt):
        """
        this function calculates the root mean squared error of the interpola-
        ted model for a sample of nt test data
        Input:
            optimiser- optimiser to be used
            nt- the size of the sample test data
        Output:
            RMSE- the root mean squared error of nt sampling points
            Rsquared- the correlation coefficient
        """
        yi_p, yi, yi_dif, yiyi_p, yiyi, yi_pyi_p = [], [], [], [], [], []
        Sample = random.sample([i for i in range(len(self.X))], nt)
        Model = kriging(self.X, self.y, name='%s' % self.name)
        Model.train(optimiser)
        for i, j in enumerate(Sample):
            yi_p.append(Model.predict(self.X[j]))
            yi.append(self.y[j])
            yi_dif.append(yi[i] - yi_p[i])
            yiyi_p.append(yi[i]*yi_p[i])
            yiyi.append(yi[i]*yi[i])
            yi_pyi_p.append(yi_p[i]*yi_p[i])
        RMSE = np.sqrt((sum(yi_dif)**2.) / float(nt))
        Rsquared = ((float(nt)*sum(yiyi_p) - sum(yi)*sum(yi_p)) /
                    (np.sqrt((float(nt)*sum(yiyi) - sum(yi)**2.) *
                     (float(nt)*sum(yi_pyi_p) - sum(yi_p)**2.))))**2.
        return ['RMSE = %f' % RMSE, 'Rsquared = %f' % Rsquared]

    def calculate_SCVR(self, optimiser='pso', plot=0):
        """
        this function calculates the standardised cross-validated residual
        (SCVR)
        value for each sampling point.
        Return an nx1 array with the SCVR value of each sampling point. If plot
        is 1, then plot scvr vs doe and y_pred vs y.
        Input:
            optimiser- optimiser to be used
            plot- if 1 plots scvr vs doe and y_pred vs y
        Output:
            predict_list- list with different interpolated kriging models
            excluding
                            each time one point of the sampling plan
            predict_varr- list with the square root of the posterior variance
            scvr- the scvr as proposed by Jones et al. (Journal of global
            optimisation, 13: 455-492, 1998)
        """
        y_normalised = (self.y - np.min(self.y)) / (np.max(self.y) -
                                                    np.min(self.y))
        y_ = np.copy(self.y)
        Kriging_models_i, list_arrays, list_ys, train_list = [], [], [], []

        for i in range(self.n):
            exclude_value = [i]
            idx = list(set(range(self.n)) - set(exclude_value))
            list_arrays.append(self.X[idx])
            list_ys.append(y_[idx])
            Kriging_models_i.append(kriging(list_arrays[i], list_ys[i],
                                            name='%s' % self.name))
            train_list.append(Kriging_models_i[i].train(optimizer=optimiser))
            self.predict_list.append(Kriging_models_i[i].predict(self.X[i]))
            self.predict_varr.append(Kriging_models_i[i].predict_var(
                                     self.X[i]))
            self.scvr.append((y_normalised[i] - Kriging_models_i[i].normy(
                             self.predict_list[i])) /
                             self.predict_varr[i][0, 0])
        if plot == 0:
            return self.predict_list, self.predict_varr, self.scvr
        elif plot == 1:
            fig = plt.figure(figsize=(12, 8), facecolor='w', edgecolor='k',
                             linewidth= 2.0, frameon=True)
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.scatter([i for i in range(1, self.n+1)], self.scvr, alpha=0.5,
                        edgecolor='black', facecolor='b', linewidth=2.)
            ax1.plot([i for i in range(0, self.n+3)], [3]*(self.n+3), 'r')
            ax1.plot([i for i in range(0, self.n+3)], [-3]*(self.n+3), 'r')
            ax1.set_xlim(0, self.n+2)
            ax1.set_ylim(-4, 4)
            ax1.set_xlabel('DoE individual')
            ax1.set_ylabel('SCVR')
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.scatter(self.predict_list, self.y, alpha=0.5,
                        edgecolor='black', facecolor='b', linewidth=2.)
            if np.max(self.y) > 0:
                ax2.set_ylim(0, np.max(self.y) + 0.00001)
                ax2.set_xlim(0, max(self.predict_list) + 0.00001)
            else:
                ax2.set_ylim(0, np.min(self.y) - 0.00001)
                ax2.set_xlim(0, min(self.predict_list) - 0.00001)
            ax2.plot(ax2.get_xlim(), ax2.get_ylim(), ls="-", c=".3")
            ax2.set_xlabel('predicted y')
            ax2.set_ylabel('y')

            plt.show()
            return self.predict_list, self.predict_varr, self.scvr
        else:
            raise ValueError('value for plot should be either 0 or 1')

    def calculate_transformed_SCVR(self, transformation, optimiser='pso',
                                   plot=0):
        """
        this function calculates the transformed standardised cross-validated
        residual (SCVR) value for each sampling point. This helps to improve
        the model.
        Return an nx1 array with the SCVR value of each sampling point. If plot
        is 1, then plot scvr vs doe and y_pred vs y.
        Input:
            optimiser- optimiser to be used
            plot- if 1 plots scvr vs doe and y_pred vs y
            transformation- the tranformation of the objective function
            (logarithmic or inverse)
        Output:
            predict_list- list with different interpolated kriging models
            excluding
                            each time one point of the sampling plan
            predict_varr- list with the square root of the posterior variance
            scvr- the scvr as proposed by Jones et al. (Journal of global
            optimisation, 13: 455-492, 1998)
        """
        y_ = np.copy(self.y)
        if transformation == 'logarithmic':
            y_ = np.log(y_)
        elif transformation == 'inverse':
            y_ = -(1.0/y_)
        y_normalised = (y_ - np.min(y_)) / (np.max(y_) -
                                            np.min(y_))
        Kriging_models_i, list_arrays, list_ys, train_list = [], [], [], []

        for i in range(self.n):
            exclude_value = [i]
            idx = list(set(range(self.n)) - set(exclude_value))
            list_arrays.append(self.X[idx])
            list_ys.append(y_[idx])
            Kriging_models_i.append(kriging(list_arrays[i], list_ys[i],
                                            name='%s' % self.name))
            train_list.append(Kriging_models_i[i].train(optimizer=optimiser))
            self.predict_list.append(Kriging_models_i[i].predict(self.X[i]))
            self.predict_varr.append(Kriging_models_i[i].predict_var(
                                     self.X[i]))
            self.scvr.append((y_normalised[i] - Kriging_models_i[i].normy(
                             self.predict_list[i])) /
                             self.predict_varr[i][0, 0])
        if plot == 0:
            return self.predict_list, self.predict_varr, self.scvr
        elif plot == 1:
            fig = plt.figure(figsize=(12, 8), facecolor='w', edgecolor='k',
                             linewidth= 2.0, frameon=True)
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.scatter([i for i in range(1, self.n+1)], self.scvr, alpha=0.5,
                        edgecolor='black', facecolor='b', linewidth=2.)
            ax1.plot([i for i in range(0, self.n+3)], [3]*(self.n+3), 'r')
            ax1.plot([i for i in range(0, self.n+3)], [-3]*(self.n+3), 'r')
            ax1.set_xlim(0, self.n+2)
            ax1.set_ylim(-4, 4)
            ax1.set_xlabel('DoE individual')
            ax1.set_ylabel('SCVR')
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.scatter(self.predict_list, y_, alpha=0.5,
                        edgecolor='black', facecolor='b', linewidth=2.)
            if np.max(y_) > 0:
                ax2.set_ylim(0, np.max(y_) + 0.00001)
                ax2.set_xlim(0, max(self.predict_list) + 0.00001)
            else:
                ax2.set_ylim(0, np.min(y_) - 0.00001)
                ax2.set_xlim(0, min(self.predict_list) - 0.00001)
            ax2.plot(ax2.get_xlim(), ax2.get_ylim(), ls="-", c=".3")
            ax2.set_xlabel('predicted %s' % 'ln(y)' if transformation ==
                           'logarithmic' else '-1/y')
            ax2.set_ylabel('predicted %s' % 'ln(y)' if transformation ==
                           'logarithmic' else '-1/y')

            plt.show()
            return self.predict_list, self.predict_varr, self.scvr
        else:
            raise ValueError('value for plot should be either 0 or 1')

    def QQ_plot(self):
        """
        returns the QQ-plot with normal distribution

        """
        plt.figure(figsize=(12, 8), facecolor='w', edgecolor='k',
                   linewidth= 2.0, frameon=True)
        stats.probplot(self.scvr, dist="norm", plot=plt)
        plt.xlabel('SCVR')
        plt.ylabel('Standard quantile')
        plt.show()

    def leave_n_out(self, q=5):
        '''
        :param q: the numer of groups to split the model data inot
        :return:
        '''
        mseArray = []
        for i in splitArrays(self.model,5):
            testk = kriging( i[0], i[1] )
            testk.train()
            for j in range(len(i[2])):
                mseArray.append(mse(i[3][j], testk.predict( i[2][j] )))
            del(testk)
        return np.average(mseArray), np.std(mseArray)


## Example Use Case:
