import tensorflow as tf

class equalizer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
    def __call__(self, h_hat, y, n):
        norm_h_hat = tf.norm(h_hat, ord='euclidean', axis=1)
        
        no_new = tf.reduce_mean(
            tf.tensordot(tf.linalg.adjoint(h_hat), n, axes=1),
            norm_h_hat ** 2
        )
        
        x_hat = tf.math.divide_no_nan(
            tf.tensordot(tf.linalg.adjoint(h_hat), y, axes=1),
            tf.norm(h_hat, ord='euclidean', axis=1) ** 2
        )
        
        return x_hat, no_new
