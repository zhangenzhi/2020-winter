import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers

class Linear(layers.Layer):

    def __init__(self,units):
        super(Linear,self).__init__() # inherit: get varibale and functions from base class or parents
        self.units = units

    def build(self,input_shape):
        #overide base::build
        self.w = self.add_weight(shape = (input_shape[-1],self.units),
                                 initializer = 'random_uniform',
                                 trainable = True)

        self.b = self.add_weight(shape = (self.units,),
                                 initializer = 'random_uniform',
                                 trainable = True)

    def call(self,inputs): # overload
        return tf.matmul(inputs,self.w) + self.b

class Dense_layer(layers.Layer):

    def __init__(self,units = 3):
        super(Dense_layer,self).__init__()
        
        self.units = units
        self.blocks = []
        for i in range(units-1):
            self.blocks.append(Linear(32))

        self.Output = Linear(10)

    def call(self,inputs):
        
        x = inputs
        for i in range(self.units-1):
            x = self.blocks[i](x)
            x = tf.nn.relu(x)
        x = self.Output(x)

        return x

class Multi_perceptron_machine(keras.Model):

    def __init__(self):
        super(Multi_perceptron_machine,self).__init__()

        self.block_1 = Dense_layer(4)
        self.block_2 = Dense_layer(2)
        self.Output = Dense_layer(1)

    def call(self,inputs):
        x = self.block_1(inputs)
        x = tf.nn.relu(x)
        x = self.block_2(x)
        x = tf.nn.relu(x)
        x = self.Output(x)
        return x

if __name__ == "__main__":
    
    # my_layer = Linear(1)
    # my_dense = Dense_layer()
    my_mpm = Multi_perceptron_machine()
    # my_mpm.summary()

    x = tf.ones((2,2))
    print(x)

    y = my_mpm(x)
    print(y)
    my_mpm.summary()
    

    