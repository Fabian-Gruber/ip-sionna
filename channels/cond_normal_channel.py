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


        noise_real = tf.random.normal(h.shape[1:], dtype=tf.float32)
        noise_imag = tf.random.normal(h.shape[1:], dtype=tf.float32)

        # Create complex noise
        noise = tf.complex(noise_real, noise_imag)

        # Broadcast the complex noise to the batch size
        n = tf.broadcast_to(noise, shape=(batch_size,) + noise.shape)

        n = n / tf.cast(tf.sqrt(no / 2.0), dtype=tf.complex64)
        
        y = h * x + n
                
        return y, h, C

    