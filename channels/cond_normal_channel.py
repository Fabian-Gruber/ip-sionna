try:
    import sionna as sn
except AttributeError:
    import sionna as sn
    
import tensorflow as tf

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
        
    def __call__(self, x, no):
        h, C = channel_generation()
        
        n = complex_noise(tf.shape(h)[0], no)
        
        n = tf.reshape(n, shape=(20, 1, 1))
        
        n = tf.broadcast_to(n, shape=h.shape)
        
        #set data type of n to data type of h
        n = tf.cast(n, dtype='float32')
        
                
        y = h * x + n
        
        return y, h, C, n
    