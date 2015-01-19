# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 18:20:24 2014

@author: Giorgos
"""
import numpy as np
import matplotlib.pyplot as plt
from pylab import rc
#from fullfactorial import mmphi

rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})


def pareto_frontier_multi(myArray):
    """
    Method to take two equally-sized arrays and return just the elements which lie
    on the Pareto frontier, sorted into order.
    
    """
    #sort on first dimension
    myArray = myArray[myArray[:,0].argsort()]
    #add first row to pareto frontier
    pareto_frontier = myArray[0:1,:]
    #text next row against the last row in pareto frontier
    for row in myArray[1:,:]:
        if sum([row[x] >= pareto_frontier[-1][x] for x in range(len(row))]) == len(row):
            #if it is better on all features add the row to pareto frontier
            pareto_frontier = np.concatenate((pareto_frontier, [row]))
    return pareto_frontier


def pareto_frontier(Xs, Ys, maxX = False, maxY = False):
    """
    Method to take two equally-sized lists and return just the elements which lie
    on the Pareto frontier, sorted into order.
    Default behaviour is to find the maximum for both X and Y, but the option is
    available to specify maxX = False or maxY = False to find the minimum for either
    or both of the parameters.
    """
	# Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
	# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]    
	# Loop through the sorted list
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]: 
                p_front.append(pair) 
        else:
            if pair[1] <= p_front[-1][1]:
                p_front.append(pair) 
	# Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY




    

