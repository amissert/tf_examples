


import numpy as np
import numpy.random as randy
import matplotlib.pyplot as plt
# import tensorflow as tf


# make fake data -------------------
ndata = 200 # number of points for each class
fake_sigma = 0.7
fake_mean = 1
fake_sg = np.array([(np.append( fake_sigma*randy.randn(ndata) + fake_mean, 
                                fake_sigma*randy.randn(ndata) - fake_mean)),
                    (np.append( fake_sigma*randy.randn(ndata) + fake_mean, 
                                fake_sigma*randy.randn(ndata) - fake_mean))])
fake_bg = np.array([(np.append( fake_sigma*randy.randn(ndata) + fake_mean, 
                                fake_sigma*randy.randn(ndata) - fake_mean)),
                    (np.append( fake_sigma*randy.randn(ndata) - fake_mean, 
                                fake_sigma*randy.randn(ndata) + fake_mean))])
# plot it
plt.scatter(fake_bg[0,:],fake_bg[1,:])
plt.scatter(fake_sg[0,:],fake_sg[1,:])
plt.show()



