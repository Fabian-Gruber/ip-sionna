import tensorflow as tf

class equalizer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
    def __call__(self, h_hat, y, no, x, estimator):
               
        norm_h_hat_squared = tf.reduce_sum(tf.square(tf.abs(h_hat)), axis=-1)
                        
        no_new = tf.math.divide_no_nan(
            no,
            norm_h_hat_squared
        )

        # print('first 10 elements of y: ', y[0, 0, :])

        # print('first 10 elements of h_hat^H: ', tf.transpose(h_hat, conjugate=True, perm=[0, 2, 1])[0, :, 0])
                        
        inner_product_h_y = tf.reduce_sum(tf.matmul(y, tf.transpose(h_hat, conjugate=True, perm=[0,2,1])), axis=-1)
                                                
        x_hat = tf.math.divide_no_nan(
            inner_product_h_y,
            tf.cast(norm_h_hat_squared, dtype=tf.complex64)
        )

        # if estimator == 'ls':
        #     print('difference between x_hat_ls and x: ', tf.reduce_sum(tf.abs(x_hat - tf.reshape(x, [-1, 1]))))
        # if estimator == 'mmse':
        #     print('difference between x_hat_mmse and x: ', tf.reduce_sum(tf.abs(x_hat - tf.reshape(x, [-1, 1]))))

                        
        return x_hat, no_new
