#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

#plotting the two graphs 

plt.plot(x,y1, color='red' , linestyle='dashed' , label='C-14')
plt.plot(x,y2 , color='green' ,label='Ra-226')
#setting the title and the labels 
plt.title('Exponential Decay of Radioactive Elements')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
#setting the range
plt.xlim(0,20000)
plt.ylim(0,1)
#the legend to show the legend labeling in the garph
plt.legend()
#show the graphs
plt.show()
