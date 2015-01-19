# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 09:16:40 2014

@author: -UOS-204351
"""
import numpy as np
from matplotlib import pyplot as plt
from pyKriging.krige import kriging


def calculate_SCVR(X, y, name, optimiser='pso', plot=0):
    """
    this function calculates the standardised cross-validated residual (SCVR)
    value for each sampling point.
    Return an nx1 array with the SCVR value of each sampling point. If plot
    is 1, then plot scvr vs doe and y_pred vs y.
    Input:
        X- Sampling plan
        y- Objective function evaluations
        name- the name of the model
        optimiser- optimiser to be used
        plot- if 1 plots scvr vs doe and y_pred vs y
    Output:
        predict_list- list with different interpolated kriging models excluding
                        each time one point of the sampling plan
        predict_varr- list with the square root of the posterior variance 
        scvr- the scvr as proposed by Jones et al. (Journal of global optimisation, 13: 455-492, 1998)
    """
    y_normalised = (y - np.min(y))/(np.max(y)-np.min(y))
    y_ = np.copy(y)
    n, k = np.shape(X)
    Kriging_models_i, list_arrays, list_ys, train_list, predict_list, predict_varr, scvr = [], [], [], [], [], [], []

    for i in range(n):
        exclude_value = [i]
        idx = list(set(range(n)) - set(exclude_value))
        list_arrays.append(X[idx])
        list_ys.append(y_[idx])
        Kriging_models_i.append(kriging(list_arrays[i], list_ys[i], name='%s' % name))
        train_list.append(Kriging_models_i[i].train(optimizer=optimiser))
        predict_list.append(Kriging_models_i[i].predict(X[i]))
        predict_varr.append(Kriging_models_i[i].predict_var(X[i]))
        scvr.append((y_normalised[i] - Kriging_models_i[i].normy(predict_list[i]))/predict_varr[i][0, 0])
    if plot == 0:
        return predict_list, predict_varr, scvr
    elif plot == 1:
        fig = plt.figure(figsize=(12, 8), facecolor='w', edgecolor='k', linewidth= 2.0, frameon=True)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter([i for i in range(1,n+1)], scvr, alpha=0.5, edgecolor='black', facecolor='b', linewidth=2.)
        ax1.plot([i for i in range(0, n+2)], [3]*(n+2), 'r')
        ax1.plot([i for i in range(0, n+2)], [-3]*(n+2), 'r')
        ax1.set_xlim(0, n+2)
        ax1.set_ylim(-4, 4)
        ax1.set_xlabel('DoE individual')
        ax1.set_ylabel('SCVR')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(predict_list, y, alpha=0.5, edgecolor='black', facecolor='b', linewidth=2.)
        if np.max(y) > 0:
            ax2.set_ylim(0, np.max(y)+0.05)
            ax2.set_xlim(0, max(predict_list)+0.05)
        else:
            ax2.set_ylim(0, np.min(y)-0.05)
            ax2.set_xlim(0, min(predict_list)-0.05)
        ax2.plot(ax2.get_xlim(), ax2.get_ylim(), ls="-", c=".3")        
        ax2.set_xlabel('predicted y')
        ax2.set_ylabel('y')
        
        plt.show()
        return predict_list, predict_varr, scvr
    else:
        raise ValueError('value for plot should be either 0 or 1')