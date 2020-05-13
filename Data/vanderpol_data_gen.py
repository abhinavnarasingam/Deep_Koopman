# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import random
from scipy import integrate
import matplotlib.pyplot as plt


def vanderpol_data(x1range, x2range, num_IC, t, seed):
    def vanderpol(t, x, u):
        return np.array([x[1], (1 - x[0] ** 2) * x[1] - x[0] + u])

    np.random.seed(seed)
    # t0, tf, dt = 0, 10, 1e-02                # start and end
    # t = np.arange(t0, tf, dt)               # the points of evaluation of solution
    for j in range(1, num_IC + 1):
        x0 = [np.random.rand() * (x1range[1] - x1range[0]) + x1range[0],
              np.random.rand() * (x2range[1] - x2range[0]) + x2range[0]]  # initial value
        x = np.zeros((len(t), len(x0)))  # array for solution
        x[0, :] = x0
        r = integrate.ode(vanderpol).set_integrator("dopri5")  # choice of method
        r.set_initial_value(x0, t[0]).set_f_params(0.0)  # initial values
        # u = np.random.randn(1,t.size)*2
        for i in range(1, t.size):
            # r.set_f_params(u[0][i])
            x[i, :] = r.integrate(t[i])  # get one more value, add it to the array
            if not r.successful():
                raise RuntimeError("Could not integrate")
        if j == 1:
            X = x
        else:
            X = np.concatenate((X, x), axis=0)

    assert (X.shape == (t.size * num_IC, len(x0)))
    return X

num_IC = 1000
filenamePrefix = 'VanderPol'

x1range = [-2.0,2.0]
x2range = [-2.0, 2.0]
tSpan = np.arange(0,10,0.01)



seed = 1;
X_test = X_train = vanderpol_data(x1range, x2range, round(0.1*num_IC), tSpan, seed)
np.savetxt(filenamePrefix + "_test" + str(1) + ".csv", X_train, delimiter=",")

seed = 2;
X_val = vanderpol_data(x1range, x2range, round(.2*num_IC), tSpan, seed);
np.savetxt(filenamePrefix + "_val" + str(1) + ".csv", X_train, delimiter=",")

for j in range(1,6):
    seed = 2+j
    X_train = vanderpol_data(x1range, x2range, round(.7*num_IC), tSpan, seed)
    np.savetxt(filenamePrefix + "_train" + str(j) + ".csv", X_train, delimiter=",")



