"""
Model a wavy 2-D function: five points chosen to reduce the model error, then five chosen by expected
improvement, which balances exploring uncertain regions against exploiting the predicted minimum.
"""
from pyKriging import kriging, samplingplan, testfunctions


def main():
    # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
    X = samplingplan(k=2, random_state=2).optimallhc(15)

    # Next, we define the problem we would like to solve
    testfun = testfunctions().paulson1
    y = testfun(X)

    # We can choose between a ga and a pso here
    optimizer = 'ga'

    # Now that we have our initial data, we can create an instance of a kriging model
    print('Setting up the Kriging Model')
    k = kriging(X, y, testfunction=testfun, name='simple_ei', testPoints=300, seed=2)
    k.train(optimizer=optimizer)
    k.snapshot()

    # Add 5 points based on model error reduction
    for i in range(5):
        for point in k.infill(1, method='error'):
            print('Adding point {} (error {:.3f})'.format(point, k.predicterr(point)))
            k.addPoint(point, testfun(point)[0])
        k.train(optimizer=optimizer)
        k.snapshot()

    # Infill 5 points based on the expected improvement criterion
    for i in range(5):
        for point in k.infill(1, method='ei'):
            print('Adding point {} (expected improvement {:.3g})'.format(point, k.infill_ei(point)))
            k.addPoint(point, testfun(point)[0])
        k.train(optimizer=optimizer)
        k.snapshot()

    print('Best sample: y = {:.4f} after {} evaluations'.format(min(k.inversenormy(k.y)), k.n))

    # And plot the results
    print('Now plotting final results...')
    k.plot()


if __name__ == '__main__':
    main()
