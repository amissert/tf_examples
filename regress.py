import numpy as np
import numpy.random as randy
import matplotlib.pyplot as plt
import tensorflow as tf

# make fake data -------------------
ndata = 25
noise = randy.randn(ndata)
fake_x = np.linspace(0, 10, ndata)
fake_y = np.linspace(0, 10, ndata) + (noise)
fake_x.reshape((ndata, 1))
fake_y.reshape((ndata, 1))

# make tensorflow model ------------
x_in = tf.placeholder(tf.float32) 
y_in = tf.placeholder(tf.float32)

# liner model
rand_init = tf.contrib.layers.xavier_initializer_conv2d(
            uniform=True, seed=None, dtype=tf.float32)

# use 'get_variable' to make tensorflow variables
W = tf.get_variable(name="w", dtype=tf.float32,  initializer=rand_init, 
                     shape=(1))
b = tf.get_variable(name="b", dtype=tf.float32,  initializer=rand_init, 
                     shape=(1))

# define relationship
out = x_in*W + b

# loss
loss = tf.reduce_mean(tf.square(out - y_in))

# minimization operation
learn_rate = 0.001
fit_line = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

# run
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
sess.run(out, {x_in: fake_x, y_in: fake_y})

# fit
for i in range(1000):
    sess.run(fit_line, {x_in: fake_x, y_in: fake_y})

# print results
pred = sess.run(out,{x_in: fake_x, y_in: fake_y})
plt.plot(fake_x,pred,fake_x,fake_y)
plt.show()





