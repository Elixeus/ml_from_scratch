import numpy as np


def r_squared(y, residual):
    total = np.sum((y - np.mean(y))**2)
    return 1.0 - (residual / total)