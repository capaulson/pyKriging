"""
Infill until the model has converged: until two successive models predict the same values (r^2 >= 0.9999 at
250 test points), with at most 15 rounds of two new points each.
"""
from pyKriging import kriging, samplingplan, testfunctions

MAX_ROUNDS = 15


def main():
    # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
    X = samplingplan(k=2, random_state=3).optimallhc(15)

    # Next, we define the problem we would like to solve
    testfun = testfunctions().branin

    # We generate our observed values based on our sampling plan and the test function
    y = testfun(X)

    print('Setting up the Kriging Model')

    # Now that we have our initial data, we can create an instance of a kriging model
    k = kriging(X, y, testfunction=testfun, name='simple', testPoints=250, seed=3)
    k.train(optimizer='ga')
    k.snapshot()

    # Let's setup our infill to terminate once our prediction has 'converged'
    # That is to say that two iterations of predictions have an rsquared value of something like 0.9999
    for i in range(MAX_ROUNDS):
        if k.history['rsquared'][-1] >= 0.9999:
            print('The prediction has converged, with {} points in the model'.format(k.n))
            break
        for point in k.infill(2):
            print('Adding point {}'.format(point))
            k.addPoint(point, testfun(point)[0])
        k.train()
        k.snapshot()
        print('Current rsquared is: {}'.format(k.history['rsquared'][-1]))
    else:
        print('Stopped after {} rounds ({} points), rsquared {}'.format(MAX_ROUNDS, k.n, k.history['rsquared'][-1]))

    # And plot the model
    print('Now plotting final results...')
    k.plot()


if __name__ == '__main__':
    main()
