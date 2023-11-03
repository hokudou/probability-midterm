import numpy
from wiener_process import brownian
from statistics import mean, stdev
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import csv

# The Wiener process parameter.
standard_deviation = 1
# Total time.
T = 10
# Number of steps.
# Time step size
dt = 1
# Number of realizations to generate.
m = 1
# Create an empty array to store the realizations.
x = numpy.empty((m, T + 1))
# Initial values of x.
x[:, 0] = 0

# # Answer of (1)
brownian(x[:, 0], T, dt, standard_deviation, out=x[:, 1:])
t = numpy.linspace(0.0, T * dt, T + 1)
# Commentout under here if you want to see the other question's answers
for k in range(m):
    plt.plot(t, x[k], linewidth=2)
plt.savefig("example1")
plt.show()