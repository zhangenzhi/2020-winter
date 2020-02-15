import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np


def autoencoder():

    encoder_inputs = keras.Input(shape = (784,),name = "input_encoder")
    x = layers.Dense(64,activation = "relu",name = "encoder_dense_1")(encoder_inputs)
    x = layers.Dense(64,activation="relu",name = "encoder_dense_2")(x)
    encoder_outputs = layers.Dense(2,activation="relu",name = "encoder_outputs")(x)

    encoder = keras.Model(encoder_inputs,encoder_outputs,name='encoder')
    encoder.summary()

    classifier_inputs = encoder_outputs
    x = layers.Dense(64,activation = "relu",name = "classifier_dense_1")(classifier_inputs)
    x = layers.Dense(64,activation = "relu",name = "classifier_dense_2")(x)

    classifier_rep_inputs = keras.Input(shape = (2,) , name="rep_input" )
    classifier_rep = keras.Model(classifier_rep_inputs , x , name="rep_space")

    classifier_outputs = layers.Dense(10, name='predictions')(x)

    classifier = keras.Model(classifier_inputs, classifier_outputs, name = 'classifier')
    classifier.summary()

    mapper_inputs = keras.Input(shape = (2,) , name = 'mapper_inputs')
    x = classifier_rep(mapper_inputs)
    mapper_outputs =  layers.Dense(1 , name = "mapper_outputs")(x)

    mapper = keras.Model(mapper_inputs,mapper_output)
    mapper.summary()
    return encoder,classifier,mapper

if __name__ == "__main__":
    autoencoder()