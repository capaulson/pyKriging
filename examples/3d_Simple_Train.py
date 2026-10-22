"""
A three-variable model: 30 initial samples plus ten rounds of ten error-based infill points. The 3-D plot draws
iso-surfaces with mayavi, which is optional (pip install mayavi); without it the plot is skipped.
"""
import importlib.util

from pyKriging import kriging, samplingplan, testfunctions


def main():
    # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
    X = samplingplan(k=3, random_state=6).optimallhc(30)

    # Next, we define the problem we would like to solve
    testfun = testfunctions().squared
    y = testfun(X)

    # Now that we have our initial data, we can create an instance of a kriging model
    k = kriging(X, y, testfunction=testfun, testPoints=300, seed=6)

    # The model is then trained
    k.train()
    k.snapshot()

    # It's typically beneficial to add additional points based on the results of the initial training
    # The infill method can be used for this
    # In this example, we will add ten points in ten batches. The model gets trained after each stage
    for i in range(10):
        print('Infill iteration {0}'.format(i + 1))
        infillPoints = k.infill(10)

        # Evaluate the infill points and add them back to the Kriging model
        for point in infillPoints:
            k.addPoint(point, testfun(point)[0])

        # Retrain the model with the new points added in to the model
        k.train()
        k.snapshot()
        print('{} points, rsquared between successive models {:.6f}'.format(k.n, k.history['rsquared'][-1]))

    x = [0.5, 0.5, 0.5]
    print('Prediction at {}: {:.4f} +/- {:.4f} (true value {:.4f})'.format(x, k.predict(x), k.predicterr(x),
                                                                          testfun(x)[0]))

    # Once the training of the model is complete, we can plot the results
    if importlib.util.find_spec('mayavi') is None:
        print('mayavi is not installed: skipping the 3-D plot (pip install mayavi to see it)')
    else:
        k.plot()


if __name__ == '__main__':
    main()
