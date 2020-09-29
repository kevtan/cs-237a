#!/usr/bin/env python

#This script will introduce us to Scipy, a library useful for scientific computation
#Adapted from https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html

#Integration

#Using known function
print("Integration:")
from scipy.integrate import quad
def integrand(x, a, b):
    return a*x**2 + b

a = 2
b = 1
I = quad(integrand, 0, 1, args=(a,b))
print("Integral of 2x^2 + 1 from x=0..1: {}".format(I[0]))

#Using arbitrarily spaced samples
import numpy as np
def f1(x):
   return x**2

x = np.array([1,3,4])
y1 = f1(x)
from scipy.integrate import simps
I1 = simps(y1, x)
print("Integral of x^2 evaluated at {}: {}".format(x.tolist(), I1))

#Optimization
print("\nOptimization:")
import numpy as np
from scipy.optimize import minimize
def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
print("Nelder-Mead simplex method:")
res = minimize(rosen, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

print("Broyden-Fletcher-Goldfarb-Shanno method:")
res1 = minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'disp': True})

#Linear Algebra

print("\nLinear algebra:")

#inverse:
import numpy as np
A = np.array([[1,3,5],[2,5,1],[2,3,8]])
print("A:\n{}\n".format(A))
print("A^(-1):\n{}\n".format(np.linalg.inv(A)))

#Systems solution
A = np.array([[1, 2], [3, 4]])
b = np.array([[5], [6]])
print("A:\n{}\n".format(A))
print("b:\n{}\n".format(b))
print("A^(-1) b:\n{}\n".format(np.linalg.solve(A, b)))
print("det(A): {}".format(np.linalg.det(A)))
print("||A||_inf: {}".format(np.linalg.norm(A, np.inf)))
