#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The OLS class is built on top of the MLClass abstract class"""
from __future__ import division
import numpy as np
from utils.utils import MLClass


class OLS(MLClass):
    """The Ordinary Least Squared (OLS) class can fit the data and make
a prediction based on the training result"""
    def __init__(self):
        """This is the ordinary least squared (OLS) class. It has 2
variables: _coef and _mse. _coef stores the coeficients and _mse stores
the mean squared error."""
        self._coef = None
        self._mse = None
    @property
    def coef(self):
        """getter for the _coef variable"""
        return self._coef

    @property
    def mse(self):
        """getter for the _mse variable"""
        return self._mse

    def fit(self, X, y):
        if y.size == X.shape[0]:
            w_hat = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(y))
            mse = np.sum((X.dot(w_hat) - y) ** 2) / len(y)
            self._coef = w_hat
            self._mse = mse
            print 'The coeficients are {}'.format(w_hat)
            print 'The mean squared error is {}'.format(mse)
        else:
            self._coef = None
            self._mse = None
            print 'The number of rows {mat} in in the matrix'\
                  'does not match the size of the'\
                  'vector {vec}'.format(mat=X.shape[0], vec=len(y))

    def predict(self, X):
        if self._coef is None:
            raise ValueError('ols coeficients are not properly initialized')
        else:
            return X.dot(self._coef)
