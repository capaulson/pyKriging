"""
Created on Tue Dec 23 09:18:20 2014

@author: Giorgos
"""
import numpy as np
from matplotlib import pyplot as plt


# at the moment this class is working only for a two dimensional proplem. I will
# improve it very soon to work for more variables.
class SobolSensitivity():
    """
    class which calculates the Global sensitivity indices as proposed by
    I.M. Sobol ("Global sensitivity indices for nonlinear mathematical models
    and their Monte Carlo estimates" In Mathematics and Computers in Simulation
    55(2001): 271-280)
    """

    def __init__(self, X, y):
        """
        X- sampling plane
        y- Objective function evaluations
        gamma_i- the list with the gamma_i main effect of the ith variable
        """
        self.X = X
        self.y = y
        self.n, self.k = np.shape(self.X)
        self.dx_i = np.ones((self.n, self.k))
        self.gamma_i = []
        self.denom_sum, self.Sobol = [], []

    def sensitivity_gamma(self, V):
        """
        Returns the gamma_i main effect of the ith variable
        Input:
            V- is the hypervolume created by all variables but x_i
        Output:
            gamma_i- the list with the gamma_i main effect of the ith variable
        """
        dx_ii = np.ones((self.k, self.n, self.k))
        for i in range(self.k):
            idx = list(set(range(self.k)) - set([i]))
            for j, m in enumerate(idx):
                dx_ii[i, :, j] = np.gradient(self.X[:, m])
                self.gamma_i.append((1./V)*(np.sum(self.y*dx_ii[i, :, j])))
        return self.gamma_i

    def sensitivity_Sobol(self, Model, plot=0):
        """
        Returns the Sobol sensitivity metrics for each variable
        Input:
            Model- the name of the model given in a string 
        Output:
            Sobol- the Sobol sensitivity metrics for each variable
            plot- if 1 then a pie chart is plotted illustrating the effect of each
            variable
        """
        colors = ['lightskyblue', 'gold', 'yellowgreen', 'lightcoral']
        for i in range(self.k):
            self.dx_i[:, i] = np.gradient(self.X[:, i])
            self.denom_sum.append(np.sum((self.gamma_i[i]**2)*self.dx_i[:, i]))
        for j in range(self.k):
            self.Sobol.append(np.sum((self.gamma_i[j]**2)*self.dx_i[:, j])/sum(self.denom_sum))
        if plot == 1:
            plt.figure(figsize=(12, 8), facecolor='w', edgecolor='k', linewidth= 2.0, frameon=True)
            labels = ["S_%d" % i for i in range(1, self.k+1)]
            sizes = [i for i in self.Sobol]
            colors = [i for i in colors[0:len(labels)]]
            explode = [0.01 + 0.02 * (i % 3) for i in range(len(sizes))]
            plt.pie(sizes, labels=labels, colors=colors, explode=explode,
                    autopct='%1.1f%%', shadow=True, startangle=90)
            # Set aspect ratio to be equal so that pie is drawn as a circle.
            plt.axis('equal')
            plt.title(Model)
            plt.show()
        return self.Sobol


if __name__== '__main__':
    dataFile = 'C:\gitRepositories\pykrige\examples\OpLHC_DOE_Obj_Fun.txt'
    
    data = np.genfromtxt(dataFile, delimiter=' ', invalid_raise=False)
    X = data[:,[3,4]]
    VAS = data[:, [5]][:,0]
    print(VAS)
    AASM = data[:, [6]][:,0]
    print(AASM)
    VAD = data[:, [7]][:,0]
    print(VAD)