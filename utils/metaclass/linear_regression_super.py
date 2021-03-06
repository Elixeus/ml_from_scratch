from utils.metaclass.mlclass import MLClass


class LinearRegressionSuper(MLClass):
    def __str__(self):
        return 'Linear Regression Superclass'
    def __init__(self, with_intercept=False):
        self._coef = None
        self._mse = None
        self._intercept = None
        self._with_intercept = with_intercept
        self._r_squared = None
        self._residual = None
    
    @property
    def coef(self):
        return self._coef
    @property
    def mse(self):
        return self._mse
    @property
    def intercept(self):
        return self._intercept
    @property
    def r_squared(self):
        return self._r_squared
    @property
    def residual(self):
        return self._residual
        
    def reset_params(self):
        self._coef = None
        self._mse = None
        self._intercept = False
        self._r_squared = None
    
    def predict(self, X):
        """use the weights to predict new values"""
        if self.coef is None:
            raise ValueError('The {} model is not initialized'.format(self))
        else:
            if not self._with_intercept:
                return X.dot(self.coef)
            else:
                X_cons = np.insert(X, 0, values=1, axis=1)
                return X_cons.dot(self.coef)