import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# [batch, timestep, {"video": [height, width, channel], "audio": [frequency]}]
# [batch, timestep, {"location": [x, y], "pressure": [force]}]

class RNNCell(layers.Layer):

    def __init__(self,units,**kwargs):
        super(RNNCell,self).__init__(**kwargs)

        self.units = units
        self.state_size = self.units[-1]
        self.output_size = self.units[-1]

        self.w = []
        self.b = []

    def build(self,input_shapes):

        for i in range(len(self.units)):

            if i == 0:
                weight = self.add_weight(shape = (input_shapes[-1],self.units[i]),
                                         initializer = 'uniform')
            else:
                weight = self.add_weight(shape = (self.units[i-1],self.units[i]),
                                         initializer = 'uniform')
            
            bias = self.add_weight(shape = (self.units[i],),
                                   initializer = 'uniform',)
            self.w.append(weight)
            self.b.append(bias)

    def call(self,inputs,states):
        
        x = inputs

        print("x:",x)
        for i in range(len(self.w)):
            x = tf.matmul(x,self.w[i]) + self.b[i]
            x = keras.activations.tanh(x)
        # print("x",x)
        # print("states",states)
        new_states =  states + x
        new_states = tf.reshape(new_states,shape = [-1,self.output_size])
        # print("new_states",new_states)

        return x, new_states


def RNNModel():
    cell = RNNCell([64,32,10])
    rnn = layers.RNN(cell)

    inputs = keras.Input((28,28))
    x = layers.Flatten()(inputs)
    # x = inputs
    print(x)
    outputs = rnn(x)

    model = keras.models.Model(inputs,outputs)
    model.summary()
    

if __name__ == "__main__":
    RNNModel()