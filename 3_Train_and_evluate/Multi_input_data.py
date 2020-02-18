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

def Multi_model():
    img_input = keras.Input(shape = (32,32,3), name = 'img_input')
    time_input = keras.Input(shape = (None,10),name = 'time_input')

    x1 = layers.Conv2D(3,3)(img_input)
    x1 = layers.GlobalMaxPooling2D()(x1)

    x2 = layers.Conv1D(3,3)(time_input)
    x2 = layers.GlobalMaxPooling1D()(x2)

    x = layers.concatenate([x1,x2])

    score_output = layers.Dense(1,name = 'score_output')(x)
    class_output = layers.Dense(5,name = 'class_output')(x)

    model = keras.Model(inputs = [img_input,time_input],
                        outputs = [score_output,class_output] )
    
    return model

def Train_multi_1():
    model = Multi_model()
    model.compile(optimizer= keras.optimizers.RMSprop(1e-3),
                  loss = [keras.losses.MeanSquaredError(),
                          keras.losses.CategoricalCrossentropy(from_logits = True)])
    # Generate dummy Numpy data
    img_data = np.random.random_sample(size=(100, 32, 32, 3))
    ts_data = np.random.random_sample(size=(100, 20, 10))
    score_targets = np.random.random_sample(size=(100, 1))
    class_targets = np.random.random_sample(size=(100, 5))

    # Fit on lists
    model.fit([img_data, ts_data], [score_targets, class_targets],
            batch_size=32,
            epochs=3)

    # Alternatively, fit on dicts
    model.fit({'img_input': img_data, 'time_input': ts_data},
            {'score_output': score_targets, 'class_output': class_targets},
            batch_size=32,
            epochs=3)
def Train_multi_2():
    model = Multi_model()
    model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.MeanSquaredError(),
          keras.losses.CategoricalCrossentropy(from_logits=True)],
    metrics=[[keras.metrics.MeanAbsolutePercentageError(),
              keras.metrics.MeanAbsoluteError()],
             [keras.metrics.CategoricalAccuracy()]])
def Train_multi_3():
    model = Multi_model()
    model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={'score_output': keras.losses.MeanSquaredError(),
          'class_output': keras.losses.CategoricalCrossentropy(from_logits=True)},
    metrics={'score_output': [keras.metrics.MeanAbsolutePercentageError(),
                              keras.metrics.MeanAbsoluteError()],
             'class_output': [keras.metrics.CategoricalAccuracy()]})
if __name__ == "__main__":
    Train_multi_1()


