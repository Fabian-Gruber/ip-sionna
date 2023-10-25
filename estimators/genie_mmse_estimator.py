import tensorflow as tf

class genie_mmse_estimator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
    def __call__(self, y, no, C, pilot):
                        
        # noise_var = no^2 * I. Be careful of the data types!
        noise_var = tf.cast(no * tf.eye(C.shape[-1]), dtype=tf.complex64)
        
        # Expand noise_var to match the shape of C for each sample in the batch
        noise_var = tf.broadcast_to(noise_var, shape=tf.shape(C))
        
                
        # scaled_C = |p|^2 * C
        scaled_C = tf.cast(tf.math.abs(pilot) ** 2, dtype=tf.complex64) * C
        
        inverse = tf.linalg.inv(scaled_C + noise_var)
        
        inverse = tf.TensorArray(tf.complex64, size=C.shape[0])
        
        for i in range(C.shape[0]):
            inv_i = tf.linalg.inv(scaled_C[i] + noise_var[i])
            inverse = inverse.write(i, inv_i)
        inverse = inverse.stack()
                
        # scaled_C_2 = conj(p) * C
        scaled_C_2 = tf.math.conj(pilot) * C
                
        # matrix = scaled_C_2 * inverse
        matrix = tf.matmul(scaled_C_2, inverse)
                                                        
        # h_hat_mmse = (scaled_C_2 * inverse) * y. Be careful of the data types!
        h_hat_mmse = tf.matmul(matrix, tf.transpose(y, perm=[0, 2, 1]))
                                        
        return tf.transpose(h_hat_mmse, perm=[0, 2, 1])