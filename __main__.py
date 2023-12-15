from end2endModel import end2endModel as e2e
    
import numpy as np
import pandas as pd
import os

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def __main__():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    num_bits_per_symbol = 2
    block_length = 256
    ebno_db_min = -15.0 # Minimum value of Eb/N0 [dB] for simulations
    ebno_db_max = 35.0 # Maximum value of Eb/N0 [dB] for simulations
    batch_size = 1024 # How many examples are processed by Sionna in parallel
    n_coherence = 1
    n_antennas = 64
    estimator = 'gmm'
    output_quantity = 'nmse'
    n_gmm_components = 128
    iterations = 10
    ebno_dbs = np.linspace(ebno_db_min, ebno_db_max, iterations)

    if output_quantity == 'ber':
    
        uncoded_e2e_model_ber = e2e(
            num_bits_per_symbol=num_bits_per_symbol, 
            block_length=block_length, 
            n_coherence=n_coherence, 
            n_antennas=n_antennas, 
            training_batch_size=100000,
            covariance_type='full',
            n_gmm_components=128,
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
            estimator='ls',
            output_quantity='nmse'
        )

        uncoded_e2e_model_nmse_mmse = e2e(
            num_bits_per_symbol=num_bits_per_symbol, 
            block_length=block_length, 
            n_coherence=n_coherence, 
            n_antennas=n_antennas, 
            estimator='mmse',
            output_quantity='nmse'
        )

        uncoded_e2e_model_nmse_gmm_circulant = e2e(
            num_bits_per_symbol=num_bits_per_symbol, 
            block_length=block_length, 
            n_coherence=n_coherence, 
            n_antennas=n_antennas, 
            training_batch_size=30000,
            covariance_type='circulant',
            n_gmm_components=n_gmm_components,
            estimator='gmm',
            output_quantity='nmse'
        )

        uncoded_e2e_model_nmse_gmm_full = e2e(
            num_bits_per_symbol=num_bits_per_symbol, 
            block_length=block_length, 
            n_coherence=n_coherence, 
            n_antennas=n_antennas, 
            training_batch_size=100000,
            covariance_type='full',
            n_gmm_components=n_gmm_components,
            estimator='gmm',
            output_quantity='nmse',
            gmm_max_iterations=1000
        )

        uncoded_e2e_model_nmse_sample_cov = e2e(
            num_bits_per_symbol=num_bits_per_symbol, 
            block_length=block_length, 
            n_coherence=n_coherence, 
            n_antennas=n_antennas,
            training_batch_size=100000,
            covariance_type='full',
            n_gmm_components=1,
            estimator='gmm',
            output_quantity='nmse'
        )

        vertically_stacked_nmse_list_ls = []
        vertically_stacked_nmse_list_mmse = []
        vertically_stacked_nmse_list_gmm_circulant = []
        vertically_stacked_nmse_list_gmm_full = []
        vertically_stacked_nmse_list_sample_cov = []

        for j in range(iterations):
            vertically_stacked_h_j, vertically_stacked_h_hat_j = uncoded_e2e_model_nmse_ls(batch_size=batch_size, ebno_db=(-15 + 5*j))
            vertically_stacked_nmse_list_ls.append(tf.reduce_sum(tf.square(tf.abs(vertically_stacked_h_j - vertically_stacked_h_hat_j))) / (batch_size * n_antennas))

            vertically_stacked_h_j, vertically_stacked_h_hat_j = uncoded_e2e_model_nmse_mmse(batch_size=batch_size, ebno_db=(-15 + 5*j))
            vertically_stacked_nmse_list_mmse.append(tf.reduce_sum(tf.square(tf.abs(vertically_stacked_h_j - vertically_stacked_h_hat_j))) / (batch_size * n_antennas))

            vertically_stacked_h_j, vertically_stacked_h_hat_j = uncoded_e2e_model_nmse_gmm_circulant(batch_size=batch_size, ebno_db=(-15 + 5*j))
            vertically_stacked_nmse_list_gmm_circulant.append(tf.reduce_sum(tf.square(tf.abs(vertically_stacked_h_j - vertically_stacked_h_hat_j))) / (batch_size * n_antennas))
           
            vertically_stacked_h_j, vertically_stacked_h_hat_j = uncoded_e2e_model_nmse_gmm_full(batch_size=batch_size, ebno_db=(-15 + 5*j))
            vertically_stacked_nmse_list_gmm_full.append(tf.reduce_sum(tf.square(tf.abs(vertically_stacked_h_j - vertically_stacked_h_hat_j))) / (batch_size * n_antennas))

            vertically_stacked_h_j, vertically_stacked_h_hat_j = uncoded_e2e_model_nmse_sample_cov(batch_size=batch_size, ebno_db=(-15 + 5*j))
            vertically_stacked_nmse_list_sample_cov.append(tf.reduce_sum(tf.square(tf.abs(vertically_stacked_h_j - vertically_stacked_h_hat_j))) / (batch_size * n_antennas))

        nmse_ls = tf.stack(vertically_stacked_nmse_list_ls, axis=0)

        nmse_mmse = tf.stack(vertically_stacked_nmse_list_mmse, axis=0)

        nmse_gmm_circulant = tf.stack(vertically_stacked_nmse_list_gmm_circulant, axis=0)

        nmse_gmm_full = tf.stack(vertically_stacked_nmse_list_gmm_full, axis=0)

        nmse_sample_cov = tf.stack(vertically_stacked_nmse_list_sample_cov, axis=0)

        nmse_data = {
            'Eb/N0 (dB)': ebno_dbs,
            'NMSE LS': nmse_ls.numpy(),
            'NMSE MMSE': nmse_mmse.numpy(),
            'NMSE GMM Circulant': nmse_gmm_circulant.numpy(),
            'NMSE GMM Full': nmse_gmm_full.numpy(),
            'NMSE Sample Covariance': nmse_sample_cov.numpy()
        }

        df = pd.DataFrame(nmse_data)

        # Save to CSV
        csv_file_path = f'/simulation_results/nmse'  # Change to your desired path
        os.makedirs(csv_file_path, exist_ok=True)

        sim_results_csv = os.path.join(csv_file_path, 'csv', f'NMSE_{n_antennas}x{n_coherence}x{batch_size}x{n_gmm_components}.csv')
        df.to_csv(sim_results_csv, index=False)

        sim_results_plot = os.path.join(csv_file_path, 'plots', f'NMSE_{n_antennas}x{n_coherence}x{batch_size}x{n_gmm_components}.png')


        # plot all three nmse curves over ebno_db
        plt.figure()
        plt.plot(ebno_dbs, nmse_ls.numpy(), label='LS')
        plt.plot(ebno_dbs, nmse_mmse.numpy(), label='MMSE')
        plt.plot(ebno_dbs, nmse_gmm_circulant.numpy(), label='GMM Circulant')
        plt.plot(ebno_dbs, nmse_gmm_full.numpy(), label='GMM Full')
        plt.plot(ebno_dbs, nmse_sample_cov.numpy(), label='Sample Covariance')
        plt.xlabel('Eb/N0 [dB]')
        plt.ylabel('NMSE')
        plt.yscale('log')
        plt.legend()
        plt.savefig(sim_results_plot)
        plt.show()
        return
    
if __name__ == "__main__":
    __main__()
