import tensorflow as tf



def line_search_step(fn,init_x,rate =1.0):
    
    with tf.GradientTape as g:
        g.watch(init_x)
        value = fn(init_x)
    
    grad = g.gradient(value,init_x)
    grad_norm = tf.reduce_sum(grad*grad)
    init_value = value
    while value > init_value - rate * grad_norm:
        x = init_x - rate * grad
        rate /= 2.0
    return x,value

@tf.custom_gradient
def clip_gradient_by_norm(x,norm):
    y = tf.identity(x)
    def grad_fn(dresult):
        return [tf.clip_by_global_norm(dresult,norm),None]
    return y,grad_fn


def log1pexp(x):
  return tf.math.log(1 + tf.exp(x))

def grad_log1pexp(x):
  with tf.GradientTape() as tape:
    tape.watch(x)
    value = log1pexp(x)
  return tape.gradient(value, x)

if __name__ == "__main__":
    x = tf.constant(1000.)
    print(grad_log1pexp(x))