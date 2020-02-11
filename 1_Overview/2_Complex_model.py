import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

class MyModel(tf.keras.Model):

    def __init__(self,num_classes=10):

        # initialize the parent class of "Mymodel" which is tf.keras.Model 
        super(MyModel,self).__init__(name="my_model") 

        self.num_classes = num_classes
        self.dense_1 = layers.Dense(32,activation='relu')
        self.dense_2 = layers.Dense(self.num_classes,activation='sigmoid')

    def call(self,inputs):
        x1 = self.dense_1(inputs)
        return self.dense_2(x1)

def tfset():
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    val_data = np.random.random((100, 32))
    val_labels = np.random.random((100, 10))

    dataset = tf.data.Dataset.from_tensor_slices((data,labels))
    dataset = dataset.batch(32)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_data,val_labels))
    val_dataset = val_dataset.batch(32)

    model = MyModel(num_classes=10)
    model.compile(optimizer = tf.keras.optimizers.RMSprop(0.001),
                  loss='categorical_crossentropy',
                  metrics = ['accuracy'])
                    
    model.fit(dataset,epochs=10,validation_data=val_dataset)
    return model

if __name__ == "__main__":
    tfset()