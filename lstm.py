import tensorflow as tf


class LSTM(object):

    def __init__(self,unis,activate_function=tf.nn.relu):
       
       self._create_params()
       self._build_layer()

    def _create_params(self):
        self.w = [] 
        w = tf.

    def _build_layer(self):
        pass

    def _model(self,x):
        pass

    def forward(self,x):
        return self._model(x)


if __name__ == "__main__":
    
    with tf.Session as sess:
        sess.run(tf.global_initializer())