"""
Noisy data: regression kriging filters the noise (its regression constant Lambda is tuned with the other
hyper-parameters) where ordinary kriging interpolates it. Both models get the same infill points.
"""
import numpy as np

from pyKriging import kriging, regression_kriging, samplingplan, testfunctions


def main():
    np.random.seed(2)  # branin_noise draws its noise from np.random

    # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
    X = samplingplan(k=2, random_state=2).optimallhc(30)

    # Next, we define the problem we would like to solve: noisy observations of the Branin function
    noisy = testfunctions().branin_noise
    testfun = testfunctions().branin
    y = noisy(X)

    print('Setting up the Kriging Models')

    # Now that we have our initial data, we can create a regression kriging model and an ordinary one
    k = regression_kriging(X, y, testfunction=testfun, name='regression', testPoints=250, seed=2)
    k.train(optimizer='pso')
    k1 = kriging(X, y, testfunction=testfun, name='interpolating', testPoints=250, seed=2)
    k1.train(optimizer='pso')
    print('Lambda = {:.3g}'.format(k.Lambda))
    k.snapshot()

    # Five infill points from the regression model, observed with noise and added to both models
    for point in k.infill(5):
        print('Adding point {}'.format(point))
        newValue = noisy(point)[0]
        k.addPoint(point, newValue)
        k1.addPoint(point, newValue)
    k.train()
    k1.train()

    Xt = np.random.rand(1000, 2)
    for name, model in (('regression kriging', k), ('ordinary kriging', k1)):
        rmse = np.sqrt(np.mean((model.predict(Xt) - testfun(Xt)) ** 2))
        print('{}: RMSE against the noise-free function {:.2f}'.format(name, rmse))
    print('Lambda = {:.3g}'.format(k.Lambda))

    # And plot the models
    print('Now plotting final results...')
    k.plot(show=False)
    k1.plot()


if __name__ == '__main__':
    main()
