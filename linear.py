

import tensorflow as tf

# variables are constructed with a initial value and a type
W = tf.Variable([3.], dtype=tf.float32)
b = tf.Variable([-3.], dtype=tf.float32)

# placeholder for data
x = tf.placeholder(tf.float32)

# establish relationship
lmodel = W*x + b #< simple linear regression

# variables are not initialized until an init routine is called in a session
ss = tf.Session()
init = tf.global_variables_initializer()
ss.run(init)  # all variables now contain initial values

# run the model
print(ss.run(lmodel, {x: [1,2,3,4]}))

# define loss (function to be minimized)
y = tf.placeholder(tf.float32)  # for target data
delta = tf.square(lmodel - y)   # array of squared diff
loss = tf.reduce_sum(delta)  # minimize total squared diff

# show the loss
print("Total loss:", ss.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# try fixing some weights
fixW = tf.assign(W,[-1])
fixb = tf.assign(b,[1])

# these actions must be performed in a session
ss.run([fixW,fixb]) #< weights are now fixed
print("New loss:",ss.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# we can minimize loss by hand, but it's better to use grad descent
optimus = tf.train.GradientDescentOptimizer(0.01) #< 0.01 is training weight
train = optimus.minimize(loss)

# train through grad descent
ss.run(init)
for i in range(1000):
    ss.run(train, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]})

#print weight values
print(ss.run([W,b]))

# print loss
show_loss = tf.Print(loss, [loss], "this is the fit loss")
ss.run(show_loss,{x:[1,2,3,4], y:[0,-1,-2,-3]})



