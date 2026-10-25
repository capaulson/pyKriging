"""
Track how a model improves during infill with q-fold ("leave n out") cross-validation: after every new point the
samples are split into 5 groups, and models trained without each group predict it.
"""
import numpy as np
import matplotlib.pyplot as plt

from pyKriging import kriging, samplingplan, testfunctions
from pyKriging.CrossValidation import Cross_Validation


def main():
    np.random.seed(4)  # cross-validation splits the samples at random

    # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
    X = samplingplan(k=2, random_state=4).optimallhc(10)

    # Next, we define the problem we would like to solve
    testfun = testfunctions().branin

    # We generate our observed values based on our sampling plan and the test function
    y = testfun(X)

    print('Setting up the Kriging Model')
    k = kriging(X, y, testfunction=testfun, name='simple', testPoints=300, seed=4)
    k.train(optimizer='ga')
    k.snapshot()
    cvMSE = [Cross_Validation(k).leave_n_out(q=5)[0]]

    for i in range(10):
        for point in k.infill(1):
            k.addPoint(point, testfun(point)[0])
        k.train(optimizer='pso')
        k.snapshot()
        cvMSE.append(Cross_Validation(k).leave_n_out(q=5)[0])
        print('{} points: cross-validated MSE {:.2e} (normalised units)'.format(k.n, cvMSE[-1]))

    print('Now plotting final results...')
    k.plot(show=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(k.history['points'][1:], k.history['rsquared'][1:], 'o-')
    ax1.set(xlabel='number of samples', ylabel='$r^2$ between successive models')
    ax2.semilogy(k.history['points'], cvMSE, 'o-')
    ax2.set(xlabel='number of samples', ylabel='5-fold cross-validated MSE')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
