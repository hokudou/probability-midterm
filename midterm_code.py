import numpy
from wiener_process import brownian
from statistics import mean, stdev
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import csv

# The Wiener process parameter.
standard_deviation = 1.2
# Total time.
T = 100
# Number of steps.
# Time step size
dt = 1
# Number of realizations to generate.
m = 100
# Create an empty array to store the realizations.
x = numpy.empty((m, T + 1))
# Initial values of x.
x[:, 0] = -11.0

# # Answer of (1)
# brownian(x[:, 0], T, dt, standard_deviation, out=x[:, 1:])
# Commentout upper here if you want to use the existing distribution
# Commentout under here if you want to generate new distribution
# I recorded the result to CSV file, so that I can use the result later.
# For the poster, I would like to use the data I wrote in 'brownian.csv'
# numpy.savetxt("brownian.csv", x, delimiter=", ", fmt="% s")

x = numpy.genfromtxt("brownian.csv", delimiter=",")

# Answer of (2)
average = numpy.empty((m + 1, T + 1))
t = numpy.linspace(0.0, T * dt, T + 1)
# Commentout under here if you want to see the other question's answers
for k in range(m):
    plt.plot(t, x[k], linewidth=0.5)
    average[k] = mean(x[:, k])
average[T]=mean(x[:, T])
plt.plot(t, average[:, 0], linewidth=2)
plt.xlabel("t", fontsize=16)
plt.ylabel("x", fontsize=16)
plt.savefig("2nd_pic")
plt.show()

# Answer of (3)
sample_A = x[:, 5]
sample_B = x[:, 60]
sample_A_mean = mean(sample_A)
sample_B_mean = mean(sample_B)
sample_A_stdev = stdev(sample_A)
sample_B_stdev = stdev(sample_B)
# Commentout under here if you want to see the other question's answers
plt.hist(sample_A, label="t=5", alpha=0.5)
plt.hist(sample_B, label="t=60", alpha=0.5)
plt.legend(loc="best")
plt.text(
    x[:, 0][0],
    10,
    "$𝜇$(t=5)={0}\n $𝜇$(t=60)={1}\n $𝜎$(t=5)={2}\n $𝜎$(t=60)={3}".format(
        sample_A_mean, sample_B_mean, sample_A_stdev, sample_B_stdev
    ),
)
plt.savefig("3rd_pic")
plt.show()


# Answer of (4)
# Commentout under here if you want to see the other question's answers
test_statistic_AB = (sample_A_mean-sample_B_mean) / \
    (sample_A_stdev/(len(sample_A)**(1/2)))
alphas = [0.01, 0.05, 0.10]
for alpha in alphas:
    z_value = scipy.stats.norm.ppf(1-alpha/2)
    if (abs(test_statistic_AB) <= z_value):
        print('With significance', 1-alpha, 'They do have equal means')
    else:
        print('With significance', 1-alpha, 'They do not have equal means')

# Answer of (5)
# Commentout under here if you want to see the other question's answers
modifier = numpy.random.uniform(0, 5, 100)
sample_C = sample_A + modifier
sample_C_mean = mean(sample_C)
sample_C_stdev = stdev(sample_C)
test_statistic_AC = (sample_A_mean-sample_C_mean) / \
    (sample_A_stdev/(len(sample_A)**(1/2)))
alphas = [0.01, 0.05, 0.10]
for alpha in alphas:
    z_value = scipy.stats.norm.ppf(1-alpha/2)
    if (abs(test_statistic_AC) <= z_value):
        print('With significance', 1-alpha, 'They do have equal means')
    else:
        print('With significance', 1-alpha, 'They do not have equal means')
s = scipy.stats.norm.rvs(sample_C_mean, sample_C_stdev, T)
norm = sorted(s, key=None, reverse=False)
sample_C_sorted = sorted(sample_C)
plt.axline((0, 0), slope=1, color='#8e8e8e', lw=2)
plt.scatter(sample_C_sorted, norm, c='blue', s=10)
plt.xlabel('Sample C')
plt.ylabel('Theoretical normal distribution')
plt.savefig("4th_pic")
plt.show()
# #As we can see it from the graph, sample C is following normal distribution
# around the range -12 to -6, but overall the correlation is not strong positive.

# Answer of (6)
sample_1 = x[0, :]
sample_2 = x[1, :]
sample_3 = x[2, :]
sample_4 = x[3, :]
samples = [sample_1, sample_2, sample_3, sample_4]
plt.boxplot(samples, vert=False)
plt.savefig("5th_pic")
plt.show()
