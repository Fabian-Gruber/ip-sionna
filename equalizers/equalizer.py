import tensorflow as tf

class equalizer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
    def __call__(self, h_hat, y, no):
        
        norm_h_hat_squared = tf.reduce_sum(tf.square(tf.abs(h_hat)), axis=-1)
                
        no_new = tf.math.divide_no_nan(
            no,
            norm_h_hat_squared
        )
                
        inner_product_h_y = tf.reduce_sum(tf.matmul(y, tf.linalg.adjoint(h_hat)))
        
        inner_product_h_y = tf.reshape(inner_product_h_y, shape=[1, 1])
                        
        x_hat = tf.math.divide_no_nan(
            inner_product_h_y,
            tf.cast(norm_h_hat_squared, dtype=tf.complex64)
        )
                
        return x_hat, no_new
