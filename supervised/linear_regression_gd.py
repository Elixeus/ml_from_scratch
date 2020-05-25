import numpy as np
from utils.metaclass.linear_regression_super import LinearRegressionSuper

class LinearRegressionOLS(LinearRegressionSuper):
    def __str__(self):
        return 'ordinary least squared for single and multivariate linear regression'
    def __init__(self, with_intercept=True):
        super(LinearRegressionOLS, self).__init__(with_intercept)
    def fit(self, X, y):
        mean = None
        X_cons = None
        results = None
        # check if X is matrix or vector
        if X.shape[1] == 1: # if it's a vector
            mean = X.mean()
            self._coef = (sum((X - mean).T.dot(y)) /(sum(np.power(X, 2)) - 1 / X.shape[0] * (sum(X)) ** 2))[0, 0]
            if self._with_intercept:
                self._intercept = 1 / X.shape[0] * sum(y - self.coef*X)
            else:
                self._intercept = 0
        # generic matrix solution
        else:
            if not self._with_intercept:
                X_cons = X
            else:
                X_cons = np.insert(X, 0, values=1, axis=1)
            results = np.linalg.inv(X_cons.T.dot(X_cons)).dot(X_cons.T).dot(y)
            self._coef = results[1:]
            self._intercept = results[0]

class LinearRegressionGD(LinearRegressionSuper):
    def __str__(self):
        return 'linear regression with gradient descent method'
    def __init__(self, with_intercept=True):
        super(LinearRegressionGD, self).__init__(with_intercept)
        self._theta = None
        self._alpha = None
        self._epsilon = None
        self._epoch = 0
        self._grad = None
        self._loss = None
    def fit(self, X, y, theta=None, alpha=1, epsilon=0.0001, epoch=2000):
        self._theta = theta
        self._alpha = alpha
        self._epsilon = epsilon
        self._epoch = epoch
        self._grad = 0
        X_cons = np.insert(X, 0, values=1, axis=1)
        # consider the case when theta is not initialized
        if self._theta is None:
            self._theta = np.zeros([X_cons.shape[1], 1])
        self._grad = 1/(X_cons.shape[0])*X_cons.T.dot(X_cons.dot(self._theta) - y)
        print(self._grad)
        n = 0
        while not all(np.abs(self._grad - self._alpha*self._grad)) <= self._epsilon and n <= self._epoch:
            print('iteration: {}. loss: {}'.format(n, self._loss))
            self._theta = self._theta - self._alpha*self._grad
            self._grad = 1/(X_cons.shape[0])*X_cons.T.dot(X_cons.dot(self._theta) - y)
            self._loss = 1/(2*(X_cons.shape[0]))*(X_cons.dot(self._theta) - y).T.dot(X_cons.dot(self._theta) - y)
            n += 1
        self._coef = self._theta

