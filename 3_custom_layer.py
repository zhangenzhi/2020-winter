import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

class My_layer(layers.Layer):
    
    def __init__(self):
        super(My_layer,self).__init__(name = "my_layer")

    def build(self):
        pass

    def call(self,inputs):
        pass
    
    def get_config(self):
        pass