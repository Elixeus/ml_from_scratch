"""Linear Regression with Stochastic Gradient Descent"""
from __future__ import division
import numpy as np
from utils.metaclass.linear_regression_super import LinearRegressionSuper
from utils.util import r_squared


class LinearRegressionSGD(LinearRegressionSuper):
    """The LinearRegressionSGD class for doing linear regression using the Stochastic Gradient Descent approach."""
    def __str__(self):
        return 'Gradient Descent Linear Regression'

    def __init__(self, with_intercept=False, alpha=0.00001, tolerance=1e-10, n_epoch=1000, seed=20):
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
        the number of epoch."""
        super(LinearRegressionSGD, self).__init__(with_intercept)
        self._alpha = alpha
        self._tolerance = tolerance
        self._n_epoch = n_epoch
        self._seed = seed

    def fit(self, X, y):
        """train the data using the gradient descent approach"""
        # TODO: check dimensions of X and y
        np.random.seed(self._seed)
        # deal with intercepts
        if not self._with_intercept:
            X_cons = X
        else:
            X_cons = np.insert(X, 0, values=1, axis=1)
        
        coef = np.random.randn(X_cons.shape[1])
        epoch = 0
        mse = np.sum(np.power(X_cons.dot(coef) - y, 2)) / X_cons.shape[1]
        while epoch < self._n_epoch:
            coef_new = coef - (self._alpha * (2 * X_cons.transpose().dot(X_cons.dot(coef) - y))/ X.shape[1])
            mse_new = np.sum((X_cons.dot(coef_new) - y) ** 2) / X_cons.shape[1]
            # deal with learning rate of too large values
            if (mse_new > mse) & ((mse_new/mse) > 1.05):
                raise ValueError('The mse is increasing instead. Please check the learning rate.')
            # exhausted all the iterations
            if epoch == self._n_epoch - 1:
                print 'Iteration exhausted'
                break
            # converged
            if np.abs(mse_new - mse) < self._tolerance:
                print 'Converged'
                break
            # print every 1000 iterations
            if epoch % 1000 == 0:
                print "Iteration: {0} - MSE: {1:.4f}".format(epoch, mse_new)

            coef = coef_new
            mse = mse_new
            epoch += 1

        self._coef = coef
        self._mse = mse
        self._residual = mse * X_cons.shape[1]# np.sum((X_cons.dot(self.coef) - y) ** 2)
        self._r_squared = r_squared(y, self.residual)
    """
    TODO:
    1. Create gradient descent as a function in the utility and update the fit function
    2. Create a linear regression class that includes _coef and _mse as private variables; also create getter/@property for these variables
    """ 