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

def basic_loss_function(y_true, y_pred):
    return tf.math.reduce_mean(tf.abs(y_true - y_pred))

class WeightedBinaryCrossEntropy(keras.losses.Loss):
    
    def __init__(self,pos_weight,weight,from_logits=False,
                 reduction = keras.losses.Reduction.AUTO,
                 name = 'weighted_binary_crossentropy'):

        super(WeightedBinaryCrossEntropy,self).__init__(reduction=reduction,name =name)
        self.pos_weight = pos_weight
        self.weight = weight
        self.from_logits = from_logits
    
    def call(self,y_true,y_pred):
        ce = tf.losses.binary_crossentropy(y_true,y_pred,from_logits = self.from_logits)[:,None]
        ce = self.weight * (ce*(1-y_true) + self.pos_weight*ce*(y_true))

        return ce
    
class CategoricalTruePositives(keras.metrics.Metric):

    def __init__(self, ame = 'categotical_true_positives', **kwargs):
        super(CategoricalTruePositives,self).__init__(name=name,**kwargs)
        self.true_positives = self.add_weight(name = 'tp',initializer='zeros')
    
    def update_state(self,y_true,y_pred,sample_weight = None):
        y_pred = tf.reshape(tf.argmax(y_pred,axis =1),shape = (-1,1))
        values = tf.cast(y_true,'int32') == tf.cast(y_pred,'int32')
        values = tf.cast(values,'float32')

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight,'float32')
            values = tf.multiply(values,sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))
        

    def result(self):
        return self.true_positives

    def reset_states(self):
        self.true_positives.assign(0.)

class ActivityRegularizationLayer(layers.Layer):
    def call(self,inputs):

        self.add_loss(tf.reduce_sum(inputs)*0.1)
        return inputs
class MetricLoggingLayer(layers.Layer):
    def call(self,inputs):

        self.add_metric(keras.backend.std(inputs),
                        name = 'std_of_activation',
                        aggregation = 'mean')
        return inputs

def train():
    model = get_uncompiled_model()
    model.compile(optimizer=keras.optimizers.Adam(),
                loss=basic_loss_function)
    model.fit(x_train, y_train, batch_size=64, epochs=3,validation_split = 0.2)

def train_custom_loss():
    model = get_uncompiled_model()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=WeightedBinaryCrossEntropy(
            pos_weight=0.5, weight = 2, from_logits=True))
    model.fit(x_train, one_hot_y_train, batch_size=64, epochs=5)

def train_custom_metrics():
    model = get_uncompiled_model()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=WeightedBinaryCrossEntropy(
            pos_weight=0.5, weight = 2, from_logits=True),
        metrics = [CategoricalTruePositives()])

    model.fit(x_train, one_hot_y_train, batch_size=64, epochs=5)

def train_custom_metrics_middle_metrics():

    inputs = keras.Input(shape=(784,),name = 'digits')
    x = layers.Dense(64,activation = 'relu',name = 'dense_1')(inputs)
    x = ActivityRegularizationLayer()(x)
    x = layers.Dense(64,activation = 'relu',name = 'dense_2')(x)
    outputs = layers.Dense(10,name = 'predictions')(x)

    model = keras.Model(inputs,outputs)
    model.compile(optimizer = keras.optimizers.RMSprop(learning_rate = 1e-3),
                  loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True))
    model.fit(x_train,y_train,batch_size = 64,epochs =1)

def train_custom_metrics_middle_logging_metrics():
    inputs = keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)

    # Insert std logging as a layer.
    x = MetricLoggingLayer()(x)

    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, name='predictions')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(x_train, y_train,
            batch_size=64,
            epochs=1)


def train_tf_data():
    model = get_compiled_model()

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(64)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    test_dataset = test_dataset.shuffle(buffer_size = 1024).batch(64)

    model.fit(train_dataset,epochs=3)

    print('Evaluate')
    result = model.evaluate(test_dataset)
    print(dict(zip(model.metrics_names,result)))
if __name__ == "__main__":
    train_tf_data()