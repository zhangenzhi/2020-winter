import tensorflow as tf

v = tf.Variable(tf.zeros([1,2,3]))
v.assign_add(tf.ones([1,2,3]))
print(v)