from end2endModel import end2endModel as e2e

try:
    import sionna as sn
except AttributeError:
    import sionna as sn
    
import numpy as np

import tensorflow as tf

def __main__():
    num_bits_per_symbol = 2
    block_length = 256
    ebno_db_min = -10.0 # Minimum value of Eb/N0 [dB] for simulations
    ebno_db_max = 10.0 # Maximum value of Eb/N0 [dB] for simulations
    batch_size = 256 # How many examples are processed by Sionna in parallel
    n_coherence = 1
    n_antennas = 32
    genie_estimator = True
    
    uncoded_e2e_model = e2e(num_bits_per_symbol=num_bits_per_symbol, block_length=block_length, n_coherence=n_coherence, n_antennas=n_antennas, genie_estimator=genie_estimator)

    #ber_plots = sn.utils.PlotBER("Uncoded BER")
    #ber_plots.simulate(
    #    uncoded_e2e_model,
    #    ebno_dbs=np.linspace(ebno_db_min, ebno_db_max, 20),
    #    batch_size=batch_size,
    #    num_target_block_errors=100, # simulate until 100 block errors occured
    #    legend="Uncoded",
    #    soft_estimates=True,
    #    max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
    #    show_fig=True   
    #)
    
    iterations = 5
    
    vertically_stacked_bits_list = []
    vertically_stacked_llrs_list = []

    for j in range(iterations):
        vertically_stacked_bits_j, vertically_stacked_llrs_j = uncoded_e2e_model(batch_size=batch_size, ebno_db=-10.0)
        vertically_stacked_bits_list.append(vertically_stacked_bits_j)
        vertically_stacked_llrs_list.append(vertically_stacked_llrs_j)

    vertically_stacked_bits = tf.concat(vertically_stacked_bits_list, axis=1)
    vertically_stacked_llrs = tf.concat(vertically_stacked_llrs_list, axis=1)

    # Modify this part to use an appropriate threshold
    threshold = 0.0  # Adjust the threshold based on your modulation scheme
    bits_hat = tf.where(vertically_stacked_llrs > threshold, tf.ones_like(vertically_stacked_bits), tf.zeros_like(vertically_stacked_bits))
    
    # print('shape of bits_hat: ', bits_hat.shape)
    # print('shape of vertically_stacked_bits: ', vertically_stacked_bits.shape)

    # Calculate BER
    bit_errors = tf.reduce_sum(tf.cast(tf.not_equal(vertically_stacked_bits, bits_hat), dtype=tf.float32))
    total_bits = iterations * batch_size * (block_length - num_bits_per_symbol)
    ber = bit_errors / total_bits
    
    mse = tf.reduce_mean(tf.square(vertically_stacked_bits - bits_hat))
    
    #print number of 1s in vertically_stacked_bits
    # print('number of 1s in vertically_stacked_bits: ', tf.reduce_sum(vertically_stacked_bits))
    
    # #print number of 1s in vertically_stacked_llrs with threshold 0.0
    # print('number of 1s in vertically_stacked_llrs with threshold 0.0: ', tf.reduce_sum(tf.where(vertically_stacked_llrs > 0.0, tf.ones_like(vertically_stacked_bits), tf.zeros_like(vertically_stacked_bits))))
        
    # print('number of bit errors: ', bit_errors)
    
    print('bit error rate: ', ber)
    
    print('mean squared error: ', mse)
    
if __name__ == "__main__":
    __main__()
