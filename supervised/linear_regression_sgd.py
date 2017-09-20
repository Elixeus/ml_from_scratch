"""Linear Regression with Stochastic Gradient Descent"""
import numpy as np


class LinearRegressionSGD(MLClass):
    """The LinearRegressionSGD class for doing linear regression using the Stochastic Gradient Descent approach."""
    def __init__(self, with_intercept=False, alpha=0.00001, tolerance=1e-10, n_epoch=1000):
        """The initialization requires severy parameters.
        -----------------------
        parameters:
        -----------------------
        with_intercept:
        type Boolean
        False if no intercept, else set to True
    
        alpha:
        type float
        the learning rate. typically a value between 0.0001 and 0.001
    
        tolerance:
        type float
        the lower threshold for performance improvement. If the performance of one iteration is smaller than this threshold, we consider the learning to have converged.
    
        n_epoch:
        type int
        the number of iterations."""
        self._coef = None
        self._mse = None
        self._intercept = None
        self._with_intercept = with_intercept
        self._alpha = alpha
        self._tolerance = tolerance
        self._n_epoch = n_epoch

    def fit(self, X, y):
        """train the data using the gradient descent approach"""
        coef = np.random.randn(X.shape[1])
        iterations = 0
        mse = np.sum(np.power(X.dot(coef) - y, 2)) / X.shape[1]
        while iterations < self._n_epoch:
            coef_new = coef - (self._alpha * (2 * X.transpose().dot(X.dot(coef) - y))/ X.shape[1])
            mse_new = np.sum((X.dot(coef_new) - y) ** 2) / X.shape[1]
            if abs(mse_new - mse) < self._tolerance:
#                 self._coef = coef
#                 self._mse = mse
                print 'Converged'
                break

            else:
                if iterations % 100 == 0:
                    print "Iteration: %d - MSE: %.4f" %(iterations, mse_new)

                coef = coef_new
                mse = mse_new
                iterations += 1
        self._coef = coef
        self._mse = mse
        print 'iteration exhausted'

    def predict(self, X):
        return X.dot(self._coef)

    """
    TODO:
    1. Create gradient descent as a function in the utility and update the fit function
    2. Create a linear regression class that includes _coef and _mse as private variables; also create getter/@property for these variables
    """ 