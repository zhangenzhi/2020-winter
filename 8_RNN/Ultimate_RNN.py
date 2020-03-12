import tensorflow as tf

class Cell(object):

    def __init__(self):
        
        self.state = tf.Tensor(shape=(10,))
        
        self.build()

    def __call__(self,inputs):
        return self.call(inputs)

    def call(self,inputs):
        new_state = tf.nn.tanh(tf.matmul(self.U,inputs) + tf.matmul(self.W*self.state) + self.b)
        self.state = new_state
        predict = tf.nn.tanh(tf.matmul(self.V,new_state) + self.c)
        return predict

    def build(self):
        #U,V,W
        self.W = tf.Variable(shape = (64,)) 
        self.V = tf.Variable(shape = (64,))
        self.U = tf.Variable(shape = (64,))

        self.b = tf.Variable(shape = (64,))
        self.c = tf.Variable(shape = (64,))

    def train(self):
        pass