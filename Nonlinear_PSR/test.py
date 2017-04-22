import numpy as np
dataX = np.loadtxt("hw1x.txt")
temp = np.exp((-(dataX-10)**2)/(2*1))
print np.count_nonzero(temp)