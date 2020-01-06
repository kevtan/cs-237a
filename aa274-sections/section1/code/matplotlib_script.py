#!/usr/bin/env python

#This script will introduce us to Matplotlib, a library used for visualization
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(100) / 5
y = np.sin(x)

plt.plot(x,y)
plt.title("Line plot")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()

plt.scatter(x,y)
plt.title("Scatter plot")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()

#Adapted from https://matplotlib.org/tutorials/introductory/sample_plots.html
data = np.random.randn(2, 100)
plt.hist(data[0])
plt.title("Histogram")
plt.xlabel("data[0]")
plt.ylabel("Count")
plt.show()

plt.hist2d(data[0], data[1])
plt.title("2D Histogram")
plt.xlabel("data[0]")
plt.ylabel("data[1]")
plt.show()
