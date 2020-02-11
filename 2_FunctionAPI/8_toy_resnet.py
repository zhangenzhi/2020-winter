import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np


inputs = layers.Input(shape=(32,32,3),name = 'img')
x = layers.Conv2D(32,3,activation = 'relu')(inputs)
x = layers.Conv2D(64,3,activation = 'relu')(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64,3,activation = 'relu',padding = 'same')(block_1_output)
x = layers.Conv2D(64,3, activation='relu',padding='same')(x)
block_2_output = layers.add([x,block_1_output])

x = layers.Conv2D(64,3,activation = 'relu',padding = 'same')(block_2_output)
x = layers.Conv2D(64,3,activation = 'relu',padding = 'same')(x)
block_3_output = layers.add([x,block_2_output])

x = layers.Conv2D(64,3,activation = 'relu')(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256,activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs,outputs,name = 'toy_resnet')
model.summary()
keras.utils.plot_model(model,'resnet.png',show_shapes =True)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['acc'])
model.fit(x_train, y_train,
          batch_size=64,
          epochs=1,
          validation_split=0.2)
          