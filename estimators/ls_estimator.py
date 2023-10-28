import tensorflow as tf

class ls_estimator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
    def __call__(self, y, x):
        h_hat_ls = tf.math.divide_no_nan(y, x)
                
        return h_hat_ls