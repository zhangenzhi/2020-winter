import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

import numpy as np

# mnist data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are Numpy arrays)
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

#one-hot y_train
one_hot_y_train = tf.one_hot(y_train.astype(np.int32),depth=10)

def get_uncompiled_model():
  inputs = keras.Input(shape=(784,), name='digits')
  x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
  x = layers.Dense(64, activation='relu', name='dense_2')(x)
  outputs = layers.Dense(10, name='predictions')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model

def get_compiled_model():
  model = get_uncompiled_model()
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])
  return model

def get_compiled_model_2():
  model = get_uncompiled_model()
  optimizer = Optimizer_lr_schedule()
  model.compile(optimizer=optimizer,
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])
  return model

class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self,logs):
        self.losses = []
    
    def on_batch_end(self,batch,logs):
        self.losses.append(logs.get('loss'))

def Train_callback():
    #EarlyStopping
    model = get_compiled_model_2()

    callbacks = [ keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1e-2,
        patience = 2,
        verbose = 1
    )] 

    model.fit(x_train,y_train,
              epochs =20,
              batch_size = 64,
              callbacks = callbacks,
              validation_split = 0.2)

def Train_Checkpoint():
  model = get_compiled_model()

  callbacks = [

          keras.callbacks.ModelCheckpoint(
            filepath = 'mymodel_{epoch}',
            save_best_only = True,
            monitor = 'val_loss',
            verbose=1
          )
  ]

  model.fit(
      x_train,y_train,
      epochs =3,
      batch_size = 64,
      callbacks = callbacks,
      validation_split = 0.2
  )

def Optimizer_lr_schedule():

  initial_learning_rate = 0.1
  lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True
  )
  optimizer = keras.optimizers.RMSprop(learning_rate = lr_schedule)
  return optimizer

if __name__ == "__main__":
    Train_callback()