import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

def Multi_perceptron():

    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(28,28)))
    model.add(layers.Dense(64,activation=tf.keras.activations.sigmoid))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(0.01),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy','mae','mse'])
    
    return model

def Multi_perceptron_from_np():
    model = tf.keras.Sequential()
    model.add(layers.Dense(64,activation=tf.keras.activations.sigmoid,input_shape = (32,)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(0.01),
                loss='mse',
                metrics=['mse','mae'])
    return model

def ministset():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = Multi_perceptron()

    model.fit(x_train,y_train,epochs=5)
    model.evaluate(x_test,y_test,verbose=2)

def numpyset():
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    val_data = np.random.random((100, 32))
    val_labels = np.random.random((100, 10))

    model = Multi_perceptron_from_np()

    model.fit(data, labels, epochs=10, batch_size=32,
            validation_data=(val_data, val_labels))

def tfset():
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    val_data = np.random.random((100, 32))
    val_labels = np.random.random((100, 10))

    dataset = tf.data.Dataset.from_tensor_slices((data,labels))
    dataset = dataset.batch(32)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_data,val_labels))
    val_dataset = val_dataset.batch(32)

    model = Multi_perceptron_from_np()

    model.fit(dataset,epochs=10,validation_data=val_dataset)
    return model

def eval_predict():
    model = tfset()

    #directly
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    model.evaluate(data,labels,batch_size=32)
    result = model.predict(data)
    print("res:",result)

    #tf.data
    dataset = tf.data.Dataset.from_tensor_slices((data,labels))
    dataset = dataset.batch(32)

    model.evaluate(dataset)
    result = model.predict(dataset)
    print("res:",result)


if __name__ == "__main__":
    eval_predict()