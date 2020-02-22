from tensorflow.keras import layers
import tensorflow as tf

class Linear(layers.Layer):

    def __init__(self,units=32,input_dim=32):
        super(Linear,self).__init__()

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value = w_init(shape = (input_dim,units),
                                                    dtype='float32'),
                             trainable = True)
        
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value = b_init(shape = (units,),
                                                    dtype = 'float32'),
                            trainable = True)
        
    def call(self,inputs):
        return tf.matmul(inputs,self.w) + self.b

class Linear_add_weight(layers.Layer):

    def __init__(self,units,input_dim):
        super(Linear_add_weight,self).__init__()

        self.w = self.add_weight(shape = (input_dim,units),
                               initializer = 'random_normal',
                               trainable = True)
        self.b = self.add_weight(shape = (units,),
                               initializer = 'zeros',
                               trainable = True)

    def call(self,inputs):
        return tf.matmul(inputs,self.w) + self.b


class nontrainable_weights(layers.Layer):
    def __init__(self,input_dim):
        super(nontrainable_weights,self).__init__()

        self.total = tf.Variable(initial_value = tf.zeros(input_dim,),
                                 trainable = False)
    
    def call(self,inputs):
        self.total.assign_add(tf.reduce_sum(inputs,axis=0))
        return self.total

class lazy_linear(layers.Layer):
    def __init__(self,units=64):
        super(lazy_linear,self).__init__()
        self.units = units

    def build(self,input_shape):
        self.w = self.add_weight(shape =  (input_shape[-1],self.units),
                                 initializer = 'random_normal',
                                 trainable = True)

        self.b = self.add_weight(shape = (self.units,),
                                 initializer = 'zeros',
                                 trainable = True)

    def call(self,inputs):
        return tf.matmul(inputs,self.w) + self.b

class MLPBlock(layers.Layer):

    def __init__(self):

        super(MLPBlock,self).__init__()
        self.linear_1 = lazy_linear(32)
        self.linear_2 = lazy_linear(32)
        self.linear_3 = lazy_linear(1)

    def call(self,inputs):

        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        x = self.linear_3(x)

        return x

class ActivityRegularizationLayer(layers.Layer):
    def __init__(self,rate):
        super(ActivityRegularizationLayer,self).__init__()
        self.rate = rate
    
    def call(self,inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs
class OutLayer(layers.Layer):
    def __init__(self):
        super(OutLayer,self).__init__()      
        self.activaity_reg = ActivityRegularizationLayer(1e-2)

    def call(self,inputs):
        return self.activaity_reg(inputs)

# mlp = MLPBlock()
# y = mlp(tf.ones(shape=(3, 64)))  # The first call to the `mlp` will create the weights
# print('weights:', len(mlp.weights))
# print('trainable weights:', len(mlp.trainable_weights))

# x = tf.ones((2, 2))
# linear_layer = lazy_linear(32)  # At instantiation, we don't know on what inputs this is going to get called
# y = linear_layer(x)
# print(linear_layer.weights)

# x = tf.ones((2, 2))
# my_sum = nontrainable_weights(2)
# y = my_sum(x)
# print(y.numpy())
# y = my_sum(x)
# print(y.numpy())

# print('weights:', len(my_sum.weights))
# print('non-trainable weights:', len(my_sum.non_trainable_weights))

# # It's not included in the trainable weights:
# print('trainable_weights:', my_sum.trainable_weights)
# print('non_trainable_weights:', my_sum.non_trainable_weights)