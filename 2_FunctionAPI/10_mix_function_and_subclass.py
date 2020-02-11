import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

#function api model
units = 32
timestep = 10
input_dim = 5

inputs = keras.Input((None,units))
x =layers.GlobalAveragePooling1D()(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs,outputs)

#subclass layer
class MyRNN(layers.Layer):
    
    def __init__(self,units=32):
        super(MyRNN,self).__init__()

        self.units = units
        self.projection_1 = layers.Dense(units=units,activation='tanh')
        self.projection_2 = layers.Dense(units=units,activation='tanh')

        self.classifier = model


    def call(self,inputs):
        outputs = []
        state = tf.zeros(shape = (inputs.shape[0],self.units))
        for t in range(inputs.shape[1]):
            x = inputs[:,t,:]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        features = tf.stack(outputs,axis=1)
        print(features.shape)

        return self.classifier(features)

mymodel = MyRNN()
_ = mymodel(tf.zeros((1,timestep,input_dim)))