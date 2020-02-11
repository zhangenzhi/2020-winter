from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
inputs = keras.Input(shape=(784,))
# print(inputs.shape)

dense = layers.Dense(64,activation='relu')
x = dense(inputs)
x = layers.Dense(64,activation='relu')(x)
outputs = layers.Dense(10,activation='softmax')(x)

#specifying a model from input and output
model = keras.Model(inputs=inputs, outputs=outputs,name='mnist_model')
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = keras.optimizers.Adam(0.001),
              metrics = ['accuracy'])
model.summary()
# keras.utils.plot_model(model,'keras_model.png',show_shapes =True)


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
dataset = dataset.batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_dataset = test_dataset.batch(32)

model.fit(dataset,epochs=5,validation_data = test_dataset)

test_scores = model.evaluate(test_dataset,verbose =2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])