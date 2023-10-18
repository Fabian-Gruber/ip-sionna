try:
    import sionna as sn
except AttributeError:
    import sionna as sn
    
import tensorflow as tf

from ...snippets_3gpp.utils import crandn
from ...snippets_3gpp.example_3gpp import example1

def complex_noise(shape, no):
    # noise ~ CN(0, no)
    noise = crandn(shape, dtype=tf.complex64)
    noise *= tf.sqrt(no)
    return noise
    

def cond_normal_channel(x, no):
    h, C = example1()
    
    n = complex_noise(tf.shape(h), no)
    
    y = tf.scalar_mul(h, x) + n
    
    return y, h, C, n