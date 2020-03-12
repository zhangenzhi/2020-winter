import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

#tf.keras..layers.SimpleRNN, layers.GRU, layers.LSTM

def RNN_model():

    model = keras.Sequential()
    model.add(layers.Embedding(input_dim = 1000, output_dim = 64))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(10))
    model.summary()

if __name__ == "__main__":
    RNN_model()