import tensorflow as tf

class equalizer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
    def __call__(self, h_hat, y, n):
        norm_h_hat = tf.norm(h_hat, ord='euclidean', axis=1)
        
        #i want to compute the inner product of h_hat and n (both of shape (1, 32)) so that the result is a scalar. this line tf.tensordot(tf.linalg.adjoint(h_hat), n, axes=1) produces a tensor of shape (10, 1, 32)
        inner_product_h_n = tf.reduce_sum(tf.tensordot(tf.linalg.adjoint(h_hat), n, axes=1))
                
        no_new = tf.reduce_mean(
            tf.norm(
                tf.math.divide_no_nan(
                    inner_product_h_n,
                    norm_h_hat ** 2
                ), ord='euclidean', axis=1
            ) ** 2
        )
        
        #print shape of h_hat
        print('shape of h_hat: ', tf.shape(h_hat))
        
        #print y and shape of y
        print('shape of y: ', tf.shape(y))
        
        inner_product_h_y = tf.reduce_sum(tf.tensordot(tf.linalg.adjoint(h_hat), y, axes=1))
        
        x_hat = tf.math.divide_no_nan(
            inner_product_h_y,
            norm_h_hat ** 2
        )
        
        return x_hat, no_new
