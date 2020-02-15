import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,),name = "inputs")
x = layers.Dense(64,activation="sigmoid",name = "d1")(inputs)
x = layers.Dense(32,activation="sigmoid",name = "d2")(x)
outputs = layers.Dense(10,activation = "softmax",name="output")(x)

model = keras.Model(inputs,outputs)
model.summary()

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

#tf.data
dataset_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
dataset_train = dataset_train.batch(32)

dataset_val = tf.data.Dataset.from_tensor_slices((x_test,y_test))
dataset_val = dataset_val.batch(32)

model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = keras.optimizers.RMSprop(learning_rate=1e-3),
              metrics = ['sparse_categorical_accuracy'])

model.fit(dataset_train,epochs=5,validation_data = dataset_val)