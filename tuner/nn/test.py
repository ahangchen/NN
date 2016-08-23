from scipy.optimize import leastsq
import numpy as np


def residuals(p, x, y):
    return p[0] * x + p[1] - y


p0 = [-1.0, 1.0]

plsq = leastsq(residuals, p0, args=(np.array([i for i in range(20)]), np.array([float(i) * 12 for i in range(20)])))
print(plsq[0])