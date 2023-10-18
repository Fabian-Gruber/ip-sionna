import tensorflow as tf

def genie_mmse_estimator(y, no, C, pilot):
    # noise_var = no^2 * I
    noise_var = tf.scalar_mul(tf.eye(tf.shape(C)[0]), no)
    
    # scaled_C = |p|^2 * C
    scaled_C = tf.scalar_mul(tf.square(tf.abs(pilot)), C)
    
    # inverse = (scaled_C + noise_var)^-1
    inverse = tf.linalg.inv(scaled_C + noise_var)
    
    # scaled_C_2 = conj(p) * C
    scaled_C_2 = tf.scalar_mul(tf.conj(pilot), C)
    
    # h_hat_mmse = (scaled_C_2 * inverse) * y
    h_hat_mmse = tf.matmul(tf.matmul(scaled_C_2, inverse), y)
    
    return h_hat_mmse