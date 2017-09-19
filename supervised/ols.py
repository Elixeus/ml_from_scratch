#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The OLS class is built on top of the MLClass abstract class"""
from __future__ import division
import numpy as np
from ml_from_scratch.utils.utils import MLClass


class OLS(MLClass):
    """The Ordinary Least Squared (OLS) class can fit the data and make
a prediction based on the training result"""
    def __init__(self, with_intercept=False):
        """This is the ordinary least squared (OLS) class. It has 2
variables: _coef and _mse. _coef stores the coeficients and _mse stores
the mean squared error."""
        self._coef = None
        self._mse = None
        self._intercept = None
        self._with_intercept = with_intercept

    @property
    def coef(self):
        """getter for the _coef variable"""
        return self._coef

    @property
    def mse(self):
        """getter for the _mse variable"""
        return self._mse
    @property
    def intercept(self):
        """getter for the _intercept variable"""
        return self._intercept

    def fit(self, X, y):
        """train the model"""
        if len(y) == X.shape[0]:
            if not self._with_intercept:
                w_hat = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(y))
                mse = np.sum((X.dot(w_hat) - y) ** 2) / len(y)
                self._coef = w_hat
                self._mse = mse
            else:
                X_cons = np.insert(X, X.shape[1], values=1, axis=1)
                w_hat = np.linalg.inv(X_cons.transpose().dot(X_cons)).dot(X_cons.transpose().dot(y))
                self._coef = w_hat
                self._mse = np.sum((X_cons.dot(w_hat) - y) ** 2) / len(y)
                self._intercept = w_hat[-1]
                
            print 'Model trained.'
            print 'The coeficients are {}'.format(self.coef)
            print 'The mean squared error is {}'.format(self.mse)

        else:
            self._coef = None
            self._mse = None
            print 'The number of rows {mat} in in the matrix'\
                  'does not match the size of the'\
                  'vector {vec}'.format(mat=X.shape[0], vec=len(y))

    def predict(self, X):
        """use the weights to predict new values"""
        if self._coef is None:
            raise ValueError('ols coeficients are not properly initialized')
        else:
            if not self._with_intercept:
                return X.dot(self._coef)
            else:
                X_cons = np.insert(X, X.shape[1], values=1, axis=1)
                return X_cons.dot(self.coef)
