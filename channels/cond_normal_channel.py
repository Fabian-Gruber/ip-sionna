try:
    import sionna as sn
except AttributeError:
    import sionna as sn
    
import tensorflow as tf
import numpy as np

from model_3gpp.utils import crandn
from model_3gpp.channel_generation import channel_generation

def complex_noise(shape, no):
    # noise ~ CN(0, no)
    print("shape: ", shape)
    noise = crandn(shape)
    noise *= tf.sqrt(no)
    return noise

class cond_normal_channel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, x, no, batch_size, n_coherence, n_antennas, h=None, C=None):
        if h is None or C is None:
            h, C = channel_generation(batch_size, n_coherence, n_antennas)
            
        n = tf.TensorArray(tf.complex64, size=batch_size)

        for i in range(batch_size):
            noise_real_i = tf.random.normal(h[i].shape, dtype=tf.float32)
            noise_imag_i = tf.random.normal(h[i].shape, dtype=tf.float32)
            noise_i = tf.complex(noise_real_i, noise_imag_i)
            noise_i = noise_i / tf.cast(tf.sqrt(no / 2.0), dtype=tf.complex64)
            n = n.write(i, noise_i)
        n = n.stack()
                
        y = h * x + n
        
        #print('first 10 elements of y: ', y[0,0,:10])
                
        return y, h, C

    