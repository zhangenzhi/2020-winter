import tensorflow as tf

a = [[],[]]
b = [[3,3],[4,4]]

c = tf.concat(b,0)

print(c)