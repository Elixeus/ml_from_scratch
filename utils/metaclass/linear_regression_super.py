from utils.metaclass.mlclass import MLClass


class LinearRegressionSuper(MLClass):
    def __init__(self, with_intercept=False):
        self._coef = None
        self._mse = None
        self._intercept = None
        self._with_intercept = with_intercept
    
    @property
    def coef(self):
        return self._coef
    @property
    def mse(self):
        return self._mse
    @property
    def intercept(self):
        return self._intercept