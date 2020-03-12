import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers

def get_dataset():
    (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

    dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
    tf.cast(mnist_labels,tf.int64)))
    dataset = dataset.shuffle(1000).batch(32) 
    return dataset 

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

class Block_layer(layers.Layer):
    def __init__(self,units = 3):
        super(Block_layer,self).__init__()

        self.units = units
        self.compute = []
        for i in range(self.units):
            self.compute.append(Dense_layer(2))
        self.Output = layers.Flatten()

    def call(self,inputs):
        buffer = []
        x = inputs

        for i in range(self.units):
            buffer.append(self.compute[i](x))

        return self.Output(tf.concat(buffer,1))

class Multi_perceptron_machine(keras.Model):

    def __init__(self):
        super(Multi_perceptron_machine,self).__init__()

        self.Input = layers.Flatten()
        self.block_1 = Block_layer(4)
        self.block_2 = Block_layer(2)
        self.Output = Dense_layer(1)

    def call(self,inputs):

        x = self.Input(inputs)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.Output(x)

        return x

if __name__ == "__main__":
    
    my_mpm = Multi_perceptron_machine()
    my_mpm.compile(loss = keras.losses.CategoricalCrossentropy(),
                   optimizer = keras.optimizers.Adam(0.01),
                   metrics = ['accuracy'])
                   
    dataset = get_dataset()
    callbacks = [keras.callbacks.TensorBoard(log_dir = './logs')]
    my_mpm.fit(dataset,epochs=3,callbacks=callbacks)
    my_mpm.summary()
    

    