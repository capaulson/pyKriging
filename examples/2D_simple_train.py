"""
Model the Branin function: train a kriging model on an optimal Latin hypercube, improve it with five rounds of
error-based infill (two new points per round) and plot the result.
"""
from pyKriging import kriging, samplingplan, testfunctions


def main():
    # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
    X = samplingplan(k=2, random_state=1).optimallhc(15)

    # Next, we define the problem we would like to solve
    testfun = testfunctions().branin

    # We generate our observed values based on our sampling plan and the test function
    y = testfun(X)

    print('Setting up the Kriging Model')

    # Now that we have our initial data, we can create an instance of a kriging model (seeded: reproducible)
    k = kriging(X, y, testfunction=testfun, name='simple', testPoints=250, seed=1)
    k.train(optimizer='ga')
    k.snapshot()

    # Add points where the model is least certain, retraining after each round
    for i in range(5):
        for point in k.infill(2):
            print('Adding point {}'.format(point))
            k.addPoint(point, testfun(point)[0])
        k.train()
        k.snapshot()

    x = [0.5, 0.5]
    print('Prediction at {}: {:.2f} +/- {:.2f} (true value {:.2f})'.format(x, k.predict(x), k.predicterr(x),
                                                                          testfun(x)[0]))

    # And plot the model
    print('Now plotting final results...')
    k.plot()


if __name__ == '__main__':
    main()
