from end2endModel import end2endModel as e2e

try:
    import sionna as sn
except AttributeError:
    import sionna as sn
    
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

def __main__():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    num_bits_per_symbol = 2
    block_length = 256
    ebno_db_min = -15.0 # Minimum value of Eb/N0 [dB] for simulations
    ebno_db_max = 35.0 # Maximum value of Eb/N0 [dB] for simulations
    batch_size = 1024 # How many examples are processed by Sionna in parallel
    n_coherence = 1
    n_antennas = 32
    training_batch_size = 10000
    covariance_type = 'circulant'
    n_gmm_components = 128
    estimator = 'gmm'
    output_quantity = 'nmse'
    iterations = 10

    if output_quantity == 'ber':
    
        uncoded_e2e_model_ber = e2e(
            num_bits_per_symbol=num_bits_per_symbol, 
            block_length=block_length, 
            n_coherence=n_coherence, 
            n_antennas=n_antennas, 
            training_batch_size=training_batch_size,
            covariance_type=covariance_type,
            n_gmm_components=n_gmm_components,
            estimator=estimator,
            output_quantity='ber'
        )
        
        vertically_stacked_bits_list = []
        vertically_stacked_llrs_list = []

        for j in range(iterations):
            vertically_stacked_bits_j, vertically_stacked_llrs_j = uncoded_e2e_model_ber(batch_size=batch_size, ebno_db=10.0)
            vertically_stacked_bits_list.append(vertically_stacked_bits_j)
            vertically_stacked_llrs_list.append(vertically_stacked_llrs_j)

        vertically_stacked_bits = tf.concat(vertically_stacked_bits_list, axis=1)
        vertically_stacked_llrs = tf.concat(vertically_stacked_llrs_list, axis=1)

        # Modify this part to use an appropriate threshold
        threshold = 0.0  # Adjust the threshold based on your modulation scheme
        bits_hat = tf.where(vertically_stacked_llrs > threshold, tf.ones_like(vertically_stacked_bits), tf.zeros_like(vertically_stacked_bits))
        
        # Calculate BER
        bit_errors = tf.reduce_sum(tf.cast(tf.not_equal(vertically_stacked_bits, bits_hat), dtype=tf.float32))
        total_bits = iterations * batch_size * (block_length - num_bits_per_symbol)
        ber = bit_errors / total_bits    
        
        print('bit error rate: ', ber)

    elif output_quantity == 'nmse':
        uncoded_e2e_model_nmse_ls = e2e(
            num_bits_per_symbol=num_bits_per_symbol, 
            block_length=block_length, 
            n_coherence=n_coherence, 
            n_antennas=n_antennas, 
            training_batch_size=training_batch_size,
            covariance_type=covariance_type,
            n_gmm_components=n_gmm_components,
            estimator='ls',
            output_quantity='nmse'
        )

        uncoded_e2e_model_nmse_mmse = e2e(
            num_bits_per_symbol=num_bits_per_symbol, 
            block_length=block_length, 
            n_coherence=n_coherence, 
            n_antennas=n_antennas, 
            training_batch_size=training_batch_size,
            covariance_type=covariance_type,
            n_gmm_components=n_gmm_components,
            estimator='mmse',
            output_quantity='nmse'
        )

        uncoded_e2e_model_nmse_gmm = e2e(
            num_bits_per_symbol=num_bits_per_symbol, 
            block_length=block_length, 
            n_coherence=n_coherence, 
            n_antennas=n_antennas, 
            training_batch_size=training_batch_size,
            covariance_type=covariance_type,
            n_gmm_components=n_gmm_components,
            estimator='gmm',
            output_quantity='nmse'
        )

        vertically_stacked_nmse_list_ls = []
        vertically_stacked_nmse_list_mmse = []
        vertically_stacked_nmse_list_gmm = []

        for j in range(iterations):
            vertically_stacked_h_j, vertically_stacked_h_hat_j = uncoded_e2e_model_nmse_ls(batch_size=batch_size, ebno_db=(-15.0 + 5*j))
            vertically_stacked_nmse_list_ls.append(tf.reduce_sum(tf.square(tf.abs(vertically_stacked_h_j - vertically_stacked_h_hat_j))) / (batch_size * n_antennas))

            vertically_stacked_h_j, vertically_stacked_h_hat_j = uncoded_e2e_model_nmse_mmse(batch_size=batch_size, ebno_db=(-15.0 + 5*j))
            vertically_stacked_nmse_list_mmse.append(tf.reduce_sum(tf.square(tf.abs(vertically_stacked_h_j - vertically_stacked_h_hat_j))) / (batch_size * n_antennas))

            vertically_stacked_h_j, vertically_stacked_h_hat_j = uncoded_e2e_model_nmse_gmm(batch_size=batch_size, ebno_db=(-15.0 + 5*j))
            vertically_stacked_nmse_list_gmm.append(tf.reduce_sum(tf.square(tf.abs(vertically_stacked_h_j - vertically_stacked_h_hat_j))) / (batch_size * n_antennas))

        nmse_ls = tf.stack(vertically_stacked_nmse_list_ls, axis=0)

        nmse_mmse = tf.stack(vertically_stacked_nmse_list_mmse, axis=0)

        nmse_gmm = tf.stack(vertically_stacked_nmse_list_gmm, axis=0)

        # plot all three nmse curves over ebno_db
        plt.figure()
        plt.plot(np.linspace(ebno_db_min, ebno_db_max, iterations), nmse_ls.numpy(), label='LS')
        plt.plot(np.linspace(ebno_db_min, ebno_db_max, iterations), nmse_mmse.numpy(), label='MMSE')
        plt.plot(np.linspace(ebno_db_min, ebno_db_max, iterations), nmse_gmm.numpy(), label='GMM')
        plt.xlabel('Eb/N0 [dB]')
        plt.ylabel('NMSE')
        plt.yscale('log')
        plt.legend()
        plt.show()
        return
    
if __name__ == "__main__":
    __main__()
