from __future__ import division
import numpy as np
from utils.metaclass.mlclass import MLClass


class PCA(MLClass):
    def __init__(self, method='svd'):
        self._components = None
        self._eigenvals = None
        self._method = method

    @property
    def components(self):
        return self._components

    @property
    def eigenvals(self):
        return self._eigenvals

    def fit(self, X):
        if self._method in ('svd' ,):
            results = self.__pca_svd(X)
        elif self._method in ('cov', ):
            results = self.__pca_cov(X)
        self._components = results[0]
        self._eigenvals = results[1]

    def __pca_svd(self, X):
        """perform singular value decomposition on the original matrix to get
        the eigenvalues and eigenvectors"""
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean
        return (np.linalg.svd(X_centered)[2], np.linalg.svd(X_centered)[1])
        
    def __pca_cov(self, X):
        """perform eigenvalue decomposition on the covariance matrix of X to 
        get the eigenvalues and eigenvectors of the covariance matrix"""
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean
        X_cov = np.cov(X_centered.transpose().dot(X_centered))
        return (np.linalg.eig(X_cov)[1], np.linalg.eig(X_cov)[0])