#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The OLS class is built on top of the MLClass abstract class"""
from __future__ import division
import numpy as np
from utils.metaclass.linear_regression_super import LinearRegressionSuper
from utils.utils import r_squared


class OLS(LinearRegressionSGD):
    """The Ordinary Least Squared (OLS) class can fit the data and make
a prediction based on the training result"""
    def __str__(self):
        return 'Ordinary Least Squared'
    def fit(self, X, y):
        """train the model"""
        # TODO: check dimension of y
        if len(y) == X.shape[0]:
            if not self._with_intercept:
                X_cons = X
            else:
                X_cons = np.insert(X, 0, values=1, axis=1)
            w_hat = np.linalg.inv(X_cons.transpose().dot(X_cons)).dot(X_cons.transpose().dot(y))
            self._mse = np.sum((X_cons.dot(w_hat) - y) ** 2) / len(y)
            self._intercept = w_hat[0]
            self._coef = w_hat
            self._r_squared = r_squared(y, self.mse)
            print 'Model trained.'
            print 'The coeficients are {}'.format(self.coef)
            print 'The mean squared error is {}'.format(self.mse)

        else:
            self.reset_params()
            print 'The number of rows {mat} in in the matrix'\
                  'does not match the size of the'\
                  'vector {vec}'.format(mat=X.shape[0], vec=len(y))
