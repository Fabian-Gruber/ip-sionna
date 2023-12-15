import tensorflow as tf
import numpy as np

from model_3gpp.utils import crandn
from model_3gpp.channel_generation import channel_generation

class cond_normal_channel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, x, no, batch_size, n_coherence, n_antennas, h=None, C=None):
        if h is None or C is None:
            h, C = channel_generation(batch_size, n_coherence, n_antennas)

        # Noise generation
        real_noise = tf.random.normal(shape=h.shape, mean=0.0, stddev=tf.sqrt(no/2))
        imag_noise = tf.random.normal(shape=h.shape, mean=0.0, stddev=tf.sqrt(no/2))
        complex_noise = tf.complex(real_noise, imag_noise)

        x = tf.reshape(x, [-1, 1, 1])

        y = h * x + complex_noise

        # Uncomment to print first 10 elements of y
        # print('first 10 elements of y: ', y[0,0,:10])

        return y, h, C

