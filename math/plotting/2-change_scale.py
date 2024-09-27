#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

#plot the the line graph

plt.plot(x, y)
# set the x and y axis labeles and the title of the graph line
plt.xlabel('time (years)')
plt.ylabel('Fraction Remaining')
plt.title('exponential Decay of C-14')
#set the scale of the y-axis equal to log
plt.yscale('log')
#set the range of x-axis
plt.xlim(0,28650)
#show the line graph plooted
plt.show()
