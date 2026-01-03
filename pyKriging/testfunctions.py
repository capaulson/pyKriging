"""
Test Functions Module for Kriging Benchmark and Validation

This module provides a collection of standard benchmark functions commonly used
to test and validate surrogate modeling and optimization algorithms. These functions
have known analytical forms, making them ideal for comparing predicted vs. actual values.

Functions included:
- Linear, squared, cubed: Simple polynomial functions
- Branin: Classic 2D optimization benchmark with multiple local minima
- Paulson: Sinusoidal test functions
- Runge: Tests interpolation behavior near boundaries
- Styblinski-Tang: Multimodal function with many local minima
- Currin et al. (1988): Exponential test function
- Rastrigin: Highly multimodal function
- Rosenbrock: "Banana function" - classic optimization test

Author: chrispaulson
"""

import numpy as np


class testfunctions():
    """
    Collection of benchmark test functions for surrogate model validation.

    All functions accept either single points or arrays of points,
    automatically handling input dimensionality.
    """
    def linear(self, X):
        """
        Linear function: f(X) = sum(X)

        A simple linear function that sums all input dimensions.
        Useful for testing basic surrogate model behavior.

        Args:
            X: Input point(s) of any dimension

        Returns:
            Sum of all coordinates for each input point
        """
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
        """
        Squared distance function: f(X) = sqrt(sum((X - offset)^2))

        Euclidean distance from a point to an offset location.
        Creates a bowl-shaped function centered at the offset.

        Args:
            X: Input point(s) of any dimension
            offset: Center point offset (default 0.25 in all dimensions)

        Returns:
            Euclidean distance from offset for each input point
        """
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
        """
        Cubed distance function: f(X) = (sum((X - offset)^3))^(1/3)

        Cube root of sum of cubed deviations from offset.
        Similar to squared but with different curvature properties.

        Args:
            X: Input point(s) of any dimension
            offset: Center point offset (default 0.25 in all dimensions)

        Returns:
            Cubed distance metric for each input point
        """
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
        """
        Branin (Branin-Hoo) function - classic 2D optimization benchmark.

        This is a widely used test function with three global minima.
        Input is normalized to [0,1]^2 and internally scaled to the
        standard Branin domain [-5,10] x [0,15].

        Global minima: f(x*) ≈ 0.397887 at:
            x* = (-π, 12.275), (π, 2.275), (9.42478, 2.475)

        Args:
            X: 2D input points (n x 2 array)

        Returns:
            Branin function values (modified with +5*x term)

        Raises:
            Exception: If input is not 2-dimensional
        """
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
        """
        Branin function with additive Gaussian noise.

        Same as branin() but with N(0, 15) noise added to each output.
        Useful for testing regression kriging and noise-robust models.

        Args:
            X: 2D input points (n x 2 array)

        Returns:
            Noisy Branin function values
        """
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
            withNoise.append(i + np.random.standard_normal()*15)
        return np.array(withNoise)


    def paulson(self,X,hz=5):
        """
        Paulson sinusoidal test function.

        A 2D oscillating function: f(x,y) = 0.5*sin(x*hz) + 0.5*cos(y*hz)
        The frequency parameter controls the number of oscillations.

        Args:
            X: 2D input points (n x 2 array)
            hz: Frequency multiplier (default 5)

        Returns:
            Sinusoidal function values in range [-1, 1]
        """
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
        """
        Paulson variant with amplitude modulation.

        Similar to paulson() but with 1/(x+0.2) amplitude modulation,
        creating stronger oscillations near the origin.

        Args:
            X: 2D input points (n x 2 array)
            hz: Frequency multiplier (default 10)

        Returns:
            Amplitude-modulated sinusoidal values
        """
        try:
            X.shape[1]
        except:
            X = np.array([X])
        if X.shape[1] != 2:
            raise Exception
        x = X[:,0]
        y = X[:,1]
        return (np.sin(x*hz))/((x+.2)) + (np.cos(y*hz))/((y+.2))

    def runge(self, X, offset=0.0):
        """
        Runge function: f(X) = 1 / (1 + sum((X - offset)^2))

        Classic function demonstrating Runge's phenomenon in polynomial
        interpolation. Has a peak at the offset and decays toward boundaries.
        Good for testing surrogate model behavior at domain edges.

        Args:
            X: Input points of any dimension
            offset: Location of peak (default 0.0 in all dimensions)

        Returns:
            Runge function values in range (0, 1]
        """
        try:
            X.shape[1]
        except:
            X = np.array(X)

        if len(X.shape)<2:
            X = np.array([X])
        offset = np.ones(X.shape[1])*offset
        y = np.array([],dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, ( 1 / (1 + np.sum((X[i]-offset)**2))))
        return y

    def stybtang(self,X):
        """
        Styblinski-Tang function - multimodal optimization benchmark.

        f(X) = 0.5 * sum(xi^4 - 16*xi^2 + 5*xi)

        A d-dimensional function with many local minima.
        Global minimum: f(x*) ≈ -39.16599*d at x* = (-2.903534, ..., -2.903534)

        Args:
            X: Input points of any dimension (in range [-5, 5]^d)

        Returns:
            Styblinski-Tang function values
        """
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
        """
        Normalized Styblinski-Tang function for [0,1]^d input.

        Same as stybtang() but accepts normalized input [0,1]^d
        which is internally scaled to [-5,5]^d.

        Args:
            X: Input points in normalized range [0,1]^d

        Returns:
            Styblinski-Tang function values
        """
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
        """
        Currin et al. (1988) exponential function.

        A 2D test function with exponential and polynomial terms.
        Commonly used in computer experiment literature.

        Reference: Currin, C., Mitchell, T., Morris, M., & Ylvisaker, D. (1988).

        Args:
            X: 2D input points (n x 2 array)

        Returns:
            Function values
        """
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

    def cosine(self, X):
        """
        Cosine function: f(X) = cos(sum(X))

        Simple periodic function for testing surrogate model
        behavior with oscillating outputs.

        Args:
            X: Input points of any dimension

        Returns:
            Cosine of sum of coordinates, in range [-1, 1]
        """
        try:
            X.shape[1]
        except:
            X = np.array(X)

        if len(X.shape)<2:
            X = np.array([X])
        y = np.array([],dtype=float)
        for i in range(X.shape[0]):
            y = np.append(y, np.cos(np.sum(X[i])))
        return y

    def rastrigin(self, x):
        """
        2D Rastrigin function:
            with global minima: 0 at x = [0, 0]
        :param x:
        :return:
        """
        y = [0.0] * 1  # Initialize array for objectives F(X)

        y[0] = 20 + x[0] ** 2 + x[1] ** 2 - 10 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))
        return y

    def rosenbrock(self, x):
        '''
        Rosenbrock function(Any order, usually 2D and 10D, sometimes larger dimension is tested)
        with global minima: 0 at x = [1] * dimension
        :param x:
        :return:
        '''
        y = [0.0] * 1
        function_sum = 0
        for i in np.arange(0, len(x)-1):
            function_sum += (1 - x[i]) ** 2 + 100 * ((x[i + 1] - x[i] ** 2) ** 2)
        y[0] = function_sum
        return y



if __name__=='__main__':
    a = testfunctions()
    print(a.rastrigin([0, 0]))
    print(a.rosenbrock([1] * 10))
    print(a.squared([1,1,1]))
    print(a.squared([[1,1,1],[2,2,2]]))
    print(a.cubed([[1,1,1],[2,2,2]]))
    print(a.stybtang([[1,1,1],[2,2,2]]))
    print(a.curretal88exp([[1,1,1],[2,2,2]]))
    print(a.cosine([[1,1,1],[2,2,2]]))
    print(a.runge([[1,1,1],[2,2,2]]))

