import tensorflow as tf
import os
import time

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

def Eager_compute():
    x = [[1,2],[3,4]]
    m = tf.matmul(x,x)
    print(m)
    print(m.numpy())

    a = tf.constant(x)
    b = [1,2]
    print(tf.add(a,b))

def Eager_Training():

    w = tf.Variable([2.0])
    with tf.GradientTape() as tape:
        f = w*w
    grad = tape.gradient(f,w)

    with tf.GradientTape() as g:
        g.watch(f)
        z = f*w
    grad_z = g.gradient(z,f)

    print(grad,grad_z)

def Eager_GradientTape():

    inputs = tf.random.uniform(shape = [10,1])
    #print(inputs)

    a = tf.keras.layers.Dense(32)
    b = tf.keras.layers.Dense(32)

    with tf.GradientTape() as tape:

        tape.watch([a.variables,b.variables])
        x = a(inputs)
        result = b(x)
        print(tape.gradient(result,[a.variables,b.variables]))

def Eager_model():

    (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

    dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
    tf.cast(mnist_labels,tf.int64)))
    dataset = dataset.shuffle(1000).batch(32)    

    # Build the model
    mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,[3,3], activation='relu',
                            input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
    ])  

    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mnist_model.compile(loss = loss_object,
                        optimizer = optimizer,
                        metrics = ['accuracy'])
    mnist_model.summary()
    mnist_model.fit(dataset,epochs = 3)

class Linear(tf.keras.Model):
  def __init__(self):
    super(Linear, self).__init__()
    self.W = tf.Variable(5., name='weight')
    self.B = tf.Variable(10., name='bias')
  def call(self, inputs):
    return inputs * self.W + self.B

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

def loss(model,inputs,target):
    error = model(inputs) - target
    return tf.reduce_mean(tf.square(error))

def grad(model,inputs,targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model,inputs,targets)
    return tape.gradient(loss_value,[model.W,model.B])

def Eager_Gradient():
    model = Linear()
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)

    print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

    steps = 300
    for i in range(steps):
        grads = grad(model,training_inputs,training_outputs)
        optimizer.apply_gradients(zip(grads,[model.W,model.B]))
        if i % 20 == 0:
            print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))
    model.summary()


if __name__ == "__main__":
    Eager_model()