import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

class My_layer(layers.Layer):
    
    def __init__(self,output_dim,**kwargs):

        self.output_dim = output_dim
        super(My_layer,self).__init__(name = "my_layer")

    def build(self,input_shape):
        #override the build in layer
        # create a trainable weight variable for this layer

        self.kernel = self.add_weight(name = 'kernel',
                                      shape = (input_shape[1],self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

    def call(self,inputs):
        return tf.matmul(inputs,self.kernel)

    def get_config(self):
        # call the get_config in the parament class
        base_config = super(My_layer,self).get_config()
        base_config['output_dim'] = self.output_dim

        return base_config

    @classmethod
    def from_config(cls,config):
        # custom config of My_layer and initialize it
        return cls(**config)

def tfset():
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    val_data = np.random.random((100, 32))
    val_labels = np.random.random((100, 10))

    dataset = tf.data.Dataset.from_tensor_slices((data,labels))
    dataset = dataset.batch(32)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_data,val_labels))
    val_dataset = val_dataset.batch(32)

    model = tf.keras.Sequential()
    model.add(My_layer(10))
    model.add(layers.Activation('softmax'))
    model.compile(optimizer = tf.keras.optimizers.RMSprop(0.001),
                  loss='categorical_crossentropy',
                  metrics = ['accuracy'])
                    
    model.fit(dataset,epochs=5,validation_data=val_dataset)


class MyDense(layers.Layer):
    def __init__(self,units = 32):
        super(MyDense,self).__init__()
        self.units = units
    
    def build(self,input_shape):
        self.w = self.add_weight(shape=(input_shape[-1],self.units),
                                        initializer = 'random_normal',
                                        trainable = True)
        self.b = self.add_weight(shape = (self.units,),
                                 initializer = 'random_normal',
                                 trainable = True)

    def call(self,inputs):
        return tf.matmul(inputs,self.w) + self.b

if __name__ == "__main__":
    Model = MyDense(32)

    inputs = keras.Input((4,))
    outputs = Model(inputs)
    model = keras.Model(inputs,outputs)
    model.summary()