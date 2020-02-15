
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np


# model
inputs = keras.Input(shape=(784,),name = 'digits')
x =layers.Dense(64,activation='relu',name = "dense_1")(inputs)
x = layers.Dense(64,activation='relu',name = "dense_2")(x)
outputs = layers.Dense(10,name="pedictions")(x)

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

model.compile(optimizer = keras.optimizers.RMSprop(),
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['sparse_categorical_accuracy'])

print('# Fit model on training data')
history = model.fit(x_train,y_train,batch_size = 64,epochs=3,validation_data=(x_val,y_val))
print('\nhistory dict:', history.history)

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print('\n# Generate predictions for 3 samples')
predictions = model.predict(x_test[:3])
print('predictions shape:', predictions.shape)