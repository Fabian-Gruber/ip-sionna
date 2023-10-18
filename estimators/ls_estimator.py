import tensorflow as tf

def ls_estimator(h, x):
    h_hat_ls = tf.math.divide_no_nan(h, x)
    
    return h_hat_ls