import tensorflow as tf


# add constants
c1 = tf.constant(3.0, dtype=tf.float32)
c2 = tf.constant(3.0, dtype=tf.float32)
csum = tf.add(c1, c2)


# add placeholders 
# how is this different? this a container that can be fed values later
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
tsum = a + b


# to actWually compute things, we must create and run a session
ss = tf.Session()


print(ss.run(tsum, {a:[2,4], b:[4,5]}))




