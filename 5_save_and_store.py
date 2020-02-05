import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import pprint
import json

model = tf.keras.Sequential()

model.add(layers.Dense(64,activation=keras.activations.relu,input_shape =(32,)))
model.add(layers.Dense(10,activation=keras.activations.softmax))


model.compile(optimizer = keras.optimizers.Adam(0.001),
              loss = keras.losses.categorical_crossentropy,
              metrics = [keras.metrics.CategoricalAccuracy()])

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

dataset = tf.data.Dataset.from_tensor_slices((data,labels))
dataset = dataset.batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((val_data,val_labels))
val_dataset = val_dataset.batch(32)

callbacks = [ tf.keras.callbacks.EarlyStopping(patience=2,monitor = 'val_loss'),
              tf.keras.callbacks.TensorBoard(log_dir = './logs')]

model.fit(dataset,epochs = 10,validation_data = val_dataset,callbacks=callbacks)

json_string = model.to_json()

pprint.pprint(json.loads(json_string))
fresh_model = tf.keras.models.model_from_json(json_string)
fresh_model.load_weights('./weights/my_model')
model.save_weights('./weights/my_model')


# # Save weights to a HDF5 file
# model.save_weights('my_model.h5', save_format='h5')

# # Restore the model's state
# model.load_weights('my_model.h5')