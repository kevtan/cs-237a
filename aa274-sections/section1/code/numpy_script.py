#!/usr/bin/env python

#This script will introduce us to Numpy, a library useful for working with n-dimensional data.
import numpy as np

print("Empty:")
empty = np.empty((2,2))
print(empty)
print("shape: {}".format(empty.shape))

print("\nFilled:")
filled = np.array([[1,2],[3,4]])
print(filled)

print("\nZeros:")
zeros = np.zeros((2,2))
print(zeros)

print("\nEye:")
identity = np.eye(2)
print(identity)

print("\nOnes:")
ones = np.ones((2,2))
print(ones)

print("\nConstants:")
constants = np.full((2,2),4)
print(constants)

print("\nIndexing:")
print("ones[:,0] = {}".format(ones[:,0]))
print("filled[0,-1] = {}".format(filled[0,-1]))
print("constants > 5 =\n{}".format(constants > 5))

print("\nOperations:")
print("filled + ones =\n{}".format(filled + ones))
print("filled - ones =\n{}".format(filled - ones))
print("filled * filled =\n{}".format(filled * filled))
print("filled / filled =\n{}".format(filled / filled))
print("filled.dot(ones) =\n{}".format(filled.dot(ones)))
print("np.sqrt(filled) =\n{}".format(np.sqrt(filled)))


np.reshape(ones,(-1))
np.tile(ones,4)
np.stack((ones,constants),axis=0)
