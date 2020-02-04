import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

class MyModel(tf.keras.Model):

    def __init__(self,num_classes=10):
        super(MyModel,self).__init__(name=="my_model")

        self.num_classes = num_classes
        self.dense_1 = layers.Dense(32,activation='relu')
        self.dense_2 = layers.Dense(self.num_classes,activation='sigmoid')

    def call(self,inputs):
        x1 = self.dense_1(inputs)
        return self.dense_2(x1)