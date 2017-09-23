import numpy as np
from utils.metaclass.linear_regression_super import LinearRegressionSuper
form utils.utils import r_squared

class Ridge(LinearRegressionSuper):
    def __init__(self, with_intercept=False, alpha=0.05):
        super(Ridge, self).__init__(with_intercept)
        self._alpha = alpha

    def fit(self, X, y):
        if len(y) == X.shape[0]:
            if not self._with_intercept:
                w_hat = np.linalg.inv(X.transpose().dot(X) + self._alpha*np.eye(X.shape[1])).dot(X.transpose()).dot(y)
                self._coef = w_hat
                self._mse = np.sum((X.dot(w_hat) - y) ** 2) / len(y)
                self._r_squared = r_squared(y, self.mse)
            else:
                X_cons = np.insert(X, 0, values=1, axis=1)
                eye = np.eye(X_cons.shape[1])
                eye[0][0] = 0
                w_hat = np.linalg.inv(X_cons.transpose().dot(X_cons) + self._alpha*eye).dot(X_cons.transpose()).dot(y)
                self._coef = w_hat
                self._mse = np.sum((X_cons.dot(w_hat) - y) ** 2) / len(y)
                self._intercept = w_hat[0]
                self._r_squared = r_squared(y, self.mse)
        else:
            self.reset_params()
            print 'The number of rows {mat} in the matrix'\
                  'does not match the size of the vector {vec}'.format(mat=X.shape[0],
                                                                       vec=len(y))