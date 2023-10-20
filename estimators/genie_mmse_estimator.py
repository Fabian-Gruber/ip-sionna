import tensorflow as tf

class genie_mmse_estimator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
    def __call__(self, y, no, C, pilot):
        # noise_var = no^2 * I. Be careful of the data types!
        noise_var = tf.cast(tf.square(no) * tf.eye(C.shape[0]), dtype=tf.complex64)
        
        # scaled_C = |p|^2 * C. Be careful of the data types!
        scaled_C = tf.math.abs(pilot) ** 2 * C
        
        # inverse = (scaled_C + noise_var)^-1. Be careful of the data types!
        inverse = tf.linalg.inv(tf.cast(scaled_C, dtype=tf.complex64) + noise_var)
        
        # scaled_C_2 = conj(p) * C
        scaled_C_2 = tf.math.conj(pilot) * C
        
        # matrix = scaled_C_2 * inverse with shape (1, matrix_size, matrix_size)
        matrix = tf.matmul(tf.cast(scaled_C_2, dtype=tf.complex64), inverse)
        
        matrix = tf.reshape(matrix, shape=[1, tf.shape(matrix)[0], tf.shape(matrix)[1]])
        
        y_transpose = tf.transpose(y, perm=[0, 2, 1])
                                
        # h_hat_mmse = (scaled_C_2 * inverse) * y. Be careful of the data types!
        h_hat_mmse = tf.matmul(matrix, y_transpose)
                                
        return tf.transpose(h_hat_mmse, perm=[0, 2, 1])