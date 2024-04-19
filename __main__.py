from end2endModel import end2endModel as e2e
    
import numpy as np
import pandas as pd
import os
import sionna as sn
import time
import threading

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def __main__():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    num_bits_per_symbol = 2
    block_length = 256
    ebno_db_min = -5.0 # Minimum value of Eb/N0 [dB] for simulations
    ebno_db_max = 15.0 # Maximum value of Eb/N0 [dB] for simulations
    batch_size = 200 # How many examples are processed by Sionna in parallel
    n_coherence = 1
    n_antennas = 64
    output_quantity = 'ber'
    n_gmm_components = 128
    iterations = 10
    ebno_dbs = np.linspace(ebno_db_min, ebno_db_max, iterations)
    code_rate = 0.9
    code = 'ldpc'
    monte_carlo_iterations = 100
    step_size = 2
    num_target_block_errors = 1000

    if output_quantity == 'ber':

        uncoded_e2e_model_ber_ls = e2e(
            num_bits_per_symbol=num_bits_per_symbol, 
            block_length=block_length, 
            n_coherence=n_coherence, 
            n_antennas=n_antennas, 
            estimator='ls',
            output_quantity='ber',
            code_rate=code_rate,
            code=code
        )

        uncoded_e2e_model_ber_mmse = e2e(
            num_bits_per_symbol=num_bits_per_symbol, 
            block_length=block_length, 
            n_coherence=n_coherence, 
            n_antennas=n_antennas, 
            estimator='mmse',
            output_quantity='ber',
            code_rate=code_rate,
            code=code
        )

        uncoded_e2e_model_ber_gmm_circulant = e2e(
            num_bits_per_symbol=num_bits_per_symbol, 
            block_length=block_length, 
            n_coherence=n_coherence, 
            n_antennas=n_antennas, 
            training_batch_size=30000,
            covariance_type='circulant',
            n_gmm_components=n_gmm_components,
            estimator='gmm',
            output_quantity='ber',
            code_rate=code_rate,
            code=code
        )

        uncoded_e2e_model_ber_gmm_full = e2e(
            num_bits_per_symbol=num_bits_per_symbol, 
            block_length=block_length, 
            n_coherence=n_coherence, 
            n_antennas=n_antennas, 
            training_batch_size=100000,
            covariance_type='full',
            n_gmm_components=n_gmm_components,
            estimator='gmm',
            output_quantity='ber',
            gmm_max_iterations=1000,
            code_rate=code_rate,
            code=code
        )

        uncoded_e2e_model_ber_sample_cov = e2e(
            num_bits_per_symbol=num_bits_per_symbol, 
            block_length=block_length, 
            n_coherence=n_coherence, 
            n_antennas=n_antennas,
            training_batch_size=100000,
            covariance_type='full',
            n_gmm_components=1,
            estimator='gmm',
            output_quantity='ber',
            code_rate=code_rate,
            code=code
        )

        uncoded_e2e_model_ber_real = e2e(
            num_bits_per_symbol=num_bits_per_symbol,
            block_length=block_length,
            n_coherence=n_coherence,
            n_antennas=n_antennas,
            estimator='real',
            output_quantity='ber',
            code_rate=code_rate,
            code=code
        )

        accumulated_ls_ber = [0] * iterations
        accumulated_mmse_ber = [0] * iterations
        accumulated_gmm_circulant_ber = [0] * iterations
        accumulated_gmm_full_ber = [0] * iterations
        accumulated_sample_cov_ber = [0] * iterations
        accumulated_real_ber = [0] * iterations

        total_block_errors_ls = [0] * iterations
        total_block_errors_mmse = [0] * iterations
        total_block_errors_gmm_circulant = [0] * iterations
        total_block_errors_gmm_full = [0] * iterations
        total_block_errors_sample_cov = [0] * iterations
        total_block_errors_real = [0] * iterations

        contributing_mc_iterations_ls = [0] * iterations
        contributing_mc_iterations_mmse = [0] * iterations
        contributing_mc_iterations_gmm_circulant = [0] * iterations
        contributing_mc_iterations_gmm_full = [0] * iterations
        contributing_mc_iterations_sample_cov = [0] * iterations
        contributing_mc_iterations_real = [0] * iterations

        for j in range(iterations):
            for mc in range(monte_carlo_iterations):
                if code == 'ldpc':
                    vertically_stacked_bits_j, vertically_stacked_llrs_j = uncoded_e2e_model_ber_ls(batch_size=batch_size, ebno_db=(-5 + 2*j))
                    bits_hat_ls_j = tf.where(vertically_stacked_llrs_j > 0, tf.ones_like(vertically_stacked_bits_j), tf.zeros_like(vertically_stacked_bits_j))
                    # vertically_stacked_ls_ber_list.append(tf.reduce_sum(tf.cast(tf.not_equal(vertically_stacked_bits_j, bits_hat_ls_j), dtype=tf.float32)) / (batch_size * (block_length - num_bits_per_symbol)))
                    bit_errors_bool = tf.not_equal(vertically_stacked_bits_j, bits_hat_ls_j)

                    # Sum bit errors per block to find blocks with any errors (result is a boolean tensor)
                    block_errors_bool = tf.reduce_any(bit_errors_bool, axis=1)

                    # Count the number of True values in block_errors_bool to get the total number of block errors
                    total_block_errors_ls[j] += tf.reduce_sum(tf.cast(block_errors_bool, dtype=tf.int32))

                    # Calculate bit error rate
                    bit_errors = tf.cast(bit_errors_bool, dtype=tf.float32)  # Cast bit errors to float for summing
                    accumulated_ls_ber[j] += tf.reduce_sum(bit_errors) / (batch_size * (block_length - num_bits_per_symbol))

                elif code == 'polar':
                    vertically_stacked_bits_j, vertically_stacked_bits_hat_j = uncoded_e2e_model_ber_ls(batch_size=batch_size, ebno_db=(-5 + 2*j))
                    bit_errors_bool = tf.not_equal(vertically_stacked_bits_j, vertically_stacked_bits_hat_j)

                    # Sum bit errors per block to find blocks with any errors (result is a boolean tensor)
                    block_errors_bool = tf.reduce_any(bit_errors_bool, axis=1)

                    # Count the number of True values in block_errors_bool to get the total number of block errors
                    total_block_errors_ls[j] += tf.reduce_sum(tf.cast(block_errors_bool, dtype=tf.int32))

                    # Calculate bit error rate
                    bit_errors = tf.cast(bit_errors_bool, dtype=tf.float32)  # Cast bit errors to float for summing
                    accumulated_ls_ber[j] += tf.reduce_sum(bit_errors) / (batch_size * (block_length - num_bits_per_symbol))


                if total_block_errors_ls[j] > num_target_block_errors and mc > 0:
                    print(f'Breaking at SNR point {-5 + 2*j}dB due to exceeding max number of block errors.')
                    break
                contributing_mc_iterations_ls[j] += 1

            print(f'LS {-5 + 2*j}dB: BER: {accumulated_ls_ber[j] / contributing_mc_iterations_ls[j]} ||| Total Block Errors: {total_block_errors_ls[j]} ||| MC Iterations: {contributing_mc_iterations_ls[j]}')
            if accumulated_ls_ber[j] == 0.0:
                for k in range(j, iterations):
                    accumulated_ls_ber[k] = tf.constant(0.0)
                print(f'Breaking at SNR point {-5 + 2*j}dB due to achieving 0 BER.')
                break
            

        for j in range(iterations):
            for mc in range(monte_carlo_iterations):
                if code == 'ldpc':
                    vertically_stacked_bits_j, vertically_stacked_llrs_j = uncoded_e2e_model_ber_mmse(batch_size=batch_size, ebno_db=(-5 + 2*j))
                    bits_hat_mmse_j = tf.where(vertically_stacked_llrs_j > 0, tf.ones_like(vertically_stacked_bits_j), tf.zeros_like(vertically_stacked_bits_j))
                    # vertically_stacked_mmse_ber_list.append(tf.reduce_sum(tf.cast(tf.not_equal(vertically_stacked_bits_j, bits_hat_mmse_j), dtype=tf.float32)) / (batch_size * (block_length - num_bits_per_symbol)))
                    bit_errors_bool = tf.not_equal(vertically_stacked_bits_j, bits_hat_mmse_j)

                    # Sum bit errors per block to find blocks with any errors (result is a boolean tensor)
                    block_errors_bool = tf.reduce_any(bit_errors_bool, axis=1)

                    # Count the number of True values in block_errors_bool to get the total number of block errors
                    total_block_errors_mmse[j] += tf.reduce_sum(tf.cast(block_errors_bool, dtype=tf.int32))

                    # Calculate bit error rate
                    bit_errors = tf.cast(bit_errors_bool, dtype=tf.float32)
                    accumulated_mmse_ber[j] += tf.reduce_sum(bit_errors) / (batch_size * (block_length - num_bits_per_symbol))

                elif code == 'polar':
                    vertically_stacked_bits_j, vertically_stacked_bits_hat_j = uncoded_e2e_model_ber_mmse(batch_size=batch_size, ebno_db=(-5 + 2*j))
                    bit_errors_bool = tf.not_equal(vertically_stacked_bits_j, vertically_stacked_bits_hat_j)

                    # Sum bit errors per block to find blocks with any errors (result is a boolean tensor)
                    block_errors_bool = tf.reduce_any(bit_errors_bool, axis=1)

                    # Count the number of True values in block_errors_bool to get the total number of block errors
                    total_block_errors_mmse[j] += tf.reduce_sum(tf.cast(block_errors_bool, dtype=tf.int32))

                    # Calculate bit error rate
                    bit_errors = tf.cast(bit_errors_bool, dtype=tf.float32)
                    accumulated_mmse_ber[j] += tf.reduce_sum(bit_errors) / (batch_size * (block_length - num_bits_per_symbol))
                    
                if total_block_errors_mmse[j] > num_target_block_errors and mc > 0:
                    print(f'Breaking at SNR point {-5 + 2*j}dB due to exceeding max number of block errors.')
                    break
                contributing_mc_iterations_mmse[j] += 1
            print(f'MMSE {-5 + 2*j}dB: BER: {accumulated_mmse_ber[j] / contributing_mc_iterations_mmse[j]} ||| Total Block Errors: {total_block_errors_mmse[j]} ||| MC Iterations: {contributing_mc_iterations_mmse[j]}')
            if accumulated_mmse_ber[j] == 0.0:
                for k in range(j, iterations):
                    accumulated_mmse_ber[k] = tf.constant(0.0)
                print(f'Breaking at SNR point {-5 + 2*j}dB due to achieving 0 BER.')
                break

        for j in range(iterations):
            for mc in range(monte_carlo_iterations):
                if code == 'ldpc':
                    vertically_stacked_bits_j, vertically_stacked_llrs_j = uncoded_e2e_model_ber_gmm_circulant(batch_size=batch_size, ebno_db=(-5 + 2*j))
                    bits_hat_gmm_circulant_j = tf.where(vertically_stacked_llrs_j > 0, tf.ones_like(vertically_stacked_bits_j), tf.zeros_like(vertically_stacked_bits_j))
                    # vertically_stacked_gmm_circulant_ber_list.append(tf.reduce_sum(tf.cast(tf.not_equal(vertically_stacked_bits_j, bits_hat_gmm_circulant_j), dtype=tf.float32)) / (batch_size * (block_length - num_bits_per_symbol)))
                    bit_errors_bool = tf.not_equal(vertically_stacked_bits_j, bits_hat_gmm_circulant_j)

                    # Sum bit errors per block to find blocks with any errors (result is a boolean tensor)
                    block_errors_bool = tf.reduce_any(bit_errors_bool, axis=1)

                    # Count the number of True values in block_errors_bool to get the total number of block errors
                    total_block_errors_gmm_circulant[j] += tf.reduce_sum(tf.cast(block_errors_bool, dtype=tf.int32))

                    # Calculate bit error rate
                    bit_errors = tf.cast(bit_errors_bool, dtype=tf.float32)
                    accumulated_gmm_circulant_ber[j] += tf.reduce_sum(bit_errors) / (batch_size * (block_length - num_bits_per_symbol))

                elif code == 'polar':
                    vertically_stacked_bits_j, vertically_stacked_bits_hat_j = uncoded_e2e_model_ber_gmm_circulant(batch_size=batch_size, ebno_db=(-5 + 2*j))
                    bit_errors_bool = tf.not_equal(vertically_stacked_bits_j, vertically_stacked_bits_hat_j)

                    # Sum bit errors per block to find blocks with any errors (result is a boolean tensor)
                    block_errors_bool = tf.reduce_any(bit_errors_bool, axis=1)

                    # Count the number of True values in block_errors_bool to get the total number of block errors
                    total_block_errors_gmm_circulant[j] += tf.reduce_sum(tf.cast(block_errors_bool, dtype=tf.int32))

                    # Calculate bit error rate
                    bit_errors = tf.cast(bit_errors_bool, dtype=tf.float32)
                    accumulated_gmm_circulant_ber[j] += tf.reduce_sum(bit_errors) / (batch_size * (block_length - num_bits_per_symbol))
                    
                if total_block_errors_gmm_circulant[j] > num_target_block_errors and mc > 0:
                    print(f'Breaking at SNR point {-5 + 2*j}dB due to exceeding max number of block errors.')
                    break
                contributing_mc_iterations_gmm_circulant[j] += 1
            print(f'GMM Circulant {-5 + 2*j}dB: BER: {accumulated_gmm_circulant_ber[j] / contributing_mc_iterations_gmm_circulant[j]} ||| Total Block Errors: {total_block_errors_gmm_circulant[j]} ||| MC Iterations: {contributing_mc_iterations_gmm_circulant[j]}')
            if accumulated_gmm_circulant_ber[j] == 0.0:
                for k in range(j, iterations):
                    accumulated_gmm_circulant_ber[k] = tf.constant(0.0)
                print(f'Breaking at SNR point {-5 + 2*j}dB due to achieving 0 BER.')
                break

        for j in range(iterations):
            for mc in range(monte_carlo_iterations):
                if code == 'ldpc':
                    vertically_stacked_bits_j, vertically_stacked_llrs_j = uncoded_e2e_model_ber_gmm_full(batch_size=batch_size, ebno_db=(-5 + 2*j))
                    bits_hat_gmm_full_j = tf.where(vertically_stacked_llrs_j > 0, tf.ones_like(vertically_stacked_bits_j), tf.zeros_like(vertically_stacked_bits_j))
                    # vertically_stacked_gmm_full_ber_list.append(tf.reduce_sum(tf.cast(tf.not_equal(vertically_stacked_bits_j, bits_hat_gmm_full_j), dtype=tf.float32)) / (batch_size * (block_length - num_bits_per_symbol)))
                    bit_errors_bool = tf.not_equal(vertically_stacked_bits_j, bits_hat_gmm_full_j)

                    # Sum bit errors per block to find blocks with any errors (result is a boolean tensor)
                    block_errors_bool = tf.reduce_any(bit_errors_bool, axis=1)

                    # Count the number of True values in block_errors_bool to get the total number of block errors
                    total_block_errors_gmm_full[j] += tf.reduce_sum(tf.cast(block_errors_bool, dtype=tf.int32))

                    # Calculate bit error rate
                    bit_errors = tf.cast(bit_errors_bool, dtype=tf.float32)
                    accumulated_gmm_full_ber[j] += tf.reduce_sum(bit_errors) / (batch_size * (block_length - num_bits_per_symbol))

                elif code == 'polar':
                    vertically_stacked_bits_j, vertically_stacked_bits_hat_j = uncoded_e2e_model_ber_gmm_full(batch_size=batch_size, ebno_db=(-5 + 2*j))
                    bit_errors_bool = tf.not_equal(vertically_stacked_bits_j, vertically_stacked_bits_hat_j)

                    # Sum bit errors per block to find blocks with any errors (result is a boolean tensor)
                    block_errors_bool = tf.reduce_any(bit_errors_bool, axis=1)

                    # Count the number of True values in block_errors_bool to get the total number of block errors
                    total_block_errors_gmm_full[j] += tf.reduce_sum(tf.cast(block_errors_bool, dtype=tf.int32))

                    # Calculate bit error rate
                    bit_errors = tf.cast(bit_errors_bool, dtype=tf.float32)
                    accumulated_gmm_full_ber[j] += tf.reduce_sum(bit_errors) / (batch_size * (block_length - num_bits_per_symbol))

                if total_block_errors_gmm_full[j] > num_target_block_errors and mc > 0:
                    print(f'Breaking at SNR point {-5 + 2*j}dB due to exceeding max number of block errors.')
                    break
                contributing_mc_iterations_gmm_full[j] += 1
            print(f'GMM Full {-5 + 2*j}dB: BER: {accumulated_gmm_full_ber[j] / contributing_mc_iterations_gmm_full[j]} ||| Total Block Errors: {total_block_errors_gmm_full[j]} ||| MC Iterations: {contributing_mc_iterations_gmm_full[j]}')
            if accumulated_gmm_full_ber[j] == 0.0:
                for k in range(j, iterations):
                    accumulated_gmm_full_ber[k] = tf.constant(0.0)
                print(f'Breaking at SNR point {-5 + 2*j}dB due to achieving 0 BER.')
                break

        for j in range(iterations):
            for mc in range(monte_carlo_iterations):
                if code == 'ldpc':
                    vertically_stacked_bits_j, vertically_stacked_llrs_j = uncoded_e2e_model_ber_sample_cov(batch_size=batch_size, ebno_db=(-5 + 2*j))
                    bits_hat_sample_cov_j = tf.where(vertically_stacked_llrs_j > 0, tf.ones_like(vertically_stacked_bits_j), tf.zeros_like(vertically_stacked_bits_j))
                    # vertically_stacked_sample_cov_ber_list.append(tf.reduce_sum(tf.cast(tf.not_equal(vertically_stacked_bits_j, bits_hat_sample_cov_j), dtype=tf.float32)) / (batch_size * (block_length - num_bits_per_symbol)))
                    bit_errors_bool = tf.not_equal(vertically_stacked_bits_j, bits_hat_sample_cov_j)

                    # Sum bit errors per block to find blocks with any errors (result is a boolean tensor)
                    block_errors_bool = tf.reduce_any(bit_errors_bool, axis=1)

                    # Count the number of True values in block_errors_bool to get the total number of block errors
                    total_block_errors_sample_cov[j] += tf.reduce_sum(tf.cast(block_errors_bool, dtype=tf.int32))

                    # Calculate bit error rate
                    bit_errors = tf.cast(bit_errors_bool, dtype=tf.float32)
                    accumulated_sample_cov_ber[j] += tf.reduce_sum(bit_errors) / (batch_size * (block_length - num_bits_per_symbol))

                elif code == 'polar':
                    vertically_stacked_bits_j, vertically_stacked_bits_hat_j = uncoded_e2e_model_ber_sample_cov(batch_size=batch_size, ebno_db=(-5 + 2*j))
                    bit_errors_bool = tf.not_equal(vertically_stacked_bits_j, vertically_stacked_bits_hat_j)

                    # Sum bit errors per block to find blocks with any errors (result is a boolean tensor)
                    block_errors_bool = tf.reduce_any(bit_errors_bool, axis=1)

                    # Count the number of True values in block_errors_bool to get the total number of block errors
                    total_block_errors_sample_cov[j] += tf.reduce_sum(tf.cast(block_errors_bool, dtype=tf.int32))

                    # Calculate bit error rate
                    bit_errors = tf.cast(bit_errors_bool, dtype=tf.float32)
                    accumulated_sample_cov_ber[j] += tf.reduce_sum(bit_errors) / (batch_size * (block_length - num_bits_per_symbol))

                if total_block_errors_sample_cov[j] > num_target_block_errors and mc > 0:
                    print(f'Breaking at SNR point {-5 + 2*j}dB due to exceeding max number of block errors.')
                    break
                contributing_mc_iterations_sample_cov[j] += 1
            print(f'Sample Covariance {-5 + 2*j}dB: BER: {accumulated_sample_cov_ber[j] / contributing_mc_iterations_sample_cov[j]} ||| Total Block Errors: {total_block_errors_sample_cov[j]} ||| MC Iterations: {contributing_mc_iterations_sample_cov[j]}')
            if accumulated_sample_cov_ber[j] == 0.0:
                for k in range(j, iterations):
                    accumulated_sample_cov_ber[k] = tf.constant(0.0)
                print(f'Breaking at SNR point {-5 + 2*j}dB due to achieving 0 BER.')
                break

        for j in range(iterations):
            for mc in range(monte_carlo_iterations):
                if code == 'ldpc':
                    vertically_stacked_bits_j, vertically_stacked_llrs_j = uncoded_e2e_model_ber_real(batch_size=batch_size, ebno_db=(-5 + 2*j))
                    bits_hat_real_j = tf.where(vertically_stacked_llrs_j > 0, tf.ones_like(vertically_stacked_bits_j), tf.zeros_like(vertically_stacked_bits_j))
                    # vertically_stacked_real_ber_list.append(tf.reduce_sum(tf.cast(tf.not_equal(vertically_stacked_bits_j, bits_hat_real_j), dtype=tf.float32)) / (batch_size * (block_length - num_bits_per_symbol)))
                    bit_errors_bool = tf.not_equal(vertically_stacked_bits_j, bits_hat_real_j)

                    # Sum bit errors per block to find blocks with any errors (result is a boolean tensor)
                    block_errors_bool = tf.reduce_any(bit_errors_bool, axis=1)

                    # Count the number of True values in block_errors_bool to get the total number of block errors
                    total_block_errors_real[j] += tf.reduce_sum(tf.cast(block_errors_bool, dtype=tf.int32))

                    # Calculate bit error rate
                    bit_errors = tf.cast(bit_errors_bool, dtype=tf.float32)
                    accumulated_real_ber[j] += tf.reduce_sum(bit_errors) / (batch_size * (block_length - num_bits_per_symbol))

                elif code == 'polar':
                    vertically_stacked_bits_j, vertically_stacked_bits_hat_j = uncoded_e2e_model_ber_real(batch_size=batch_size, ebno_db=(-5 + 2*j))
                    bit_errors_bool = tf.not_equal(vertically_stacked_bits_j, vertically_stacked_bits_hat_j)

                    # Sum bit errors per block to find blocks with any errors (result is a boolean tensor)
                    block_errors_bool = tf.reduce_any(bit_errors_bool, axis=1)

                    # Count the number of True values in block_errors_bool to get the total number of block errors
                    total_block_errors_real[j] += tf.reduce_sum(tf.cast(block_errors_bool, dtype=tf.int32))

                    # Calculate bit error rate
                    bit_errors = tf.cast(bit_errors_bool, dtype=tf.float32)
                    accumulated_real_ber[j] += tf.reduce_sum(bit_errors) / (batch_size * (block_length - num_bits_per_symbol))

                if total_block_errors_real[j] > num_target_block_errors and mc > 0:
                    print(f'Breaking at SNR point {-5 + 2*j}dB due to exceeding max number of block errors.')
                    break
                contributing_mc_iterations_real[j] += 1
            print(f'Real {-5 + 2*j}dB: BER: {accumulated_real_ber[j] / contributing_mc_iterations_real[j]} ||| Total Block Errors: {total_block_errors_real[j]} ||| MC Iterations: {contributing_mc_iterations_real[j]}')
            if accumulated_real_ber[j] == 0.0 and mc > 0:
                for k in range(j, iterations):
                    accumulated_real_ber[k] = tf.constant(0.0)
                print(f'Breaking at SNR point {-5 + 2*j}dB due to achieving 0 BER.')
                break
                
        averaged_ls_ber = [accumulated_ls_ber[j] / contributing_mc_iterations_ls[j] if contributing_mc_iterations_ls[j] > 0 else 0 for j in range(iterations)]
        averaged_mmse_ber = [accumulated_mmse_ber[j] / contributing_mc_iterations_mmse[j] if contributing_mc_iterations_mmse[j] > 0 else 0 for j in range(iterations)]
        averaged_gmm_circulant_ber = [accumulated_gmm_circulant_ber[j] / contributing_mc_iterations_gmm_circulant[j] if contributing_mc_iterations_gmm_circulant[j] > 0 else 0 for j in range(iterations)]
        averaged_gmm_full_ber = [accumulated_gmm_full_ber[j] / contributing_mc_iterations_gmm_full[j] if contributing_mc_iterations_gmm_full[j] > 0 else 0 for j in range(iterations)]
        averaged_sample_cov_ber = [accumulated_sample_cov_ber[j] / contributing_mc_iterations_sample_cov[j] if contributing_mc_iterations_sample_cov[j] > 0 else 0 for j in range(iterations)]
        averaged_real_ber = [accumulated_real_ber[j] / contributing_mc_iterations_real[j] if contributing_mc_iterations_real[j] > 0 else 0 for j in range(iterations)]

        ls_ber = tf.stack(averaged_ls_ber, axis=0)

        mmse_ber = tf.stack(averaged_mmse_ber, axis=0)

        gmm_circulant_ber = tf.stack(averaged_gmm_circulant_ber, axis=0)

        gmm_full_ber = tf.stack(averaged_gmm_full_ber, axis=0)

        sample_cov_ber = tf.stack(averaged_sample_cov_ber, axis=0)

        real_ber = tf.stack(averaged_real_ber, axis=0)

        ber_data = {
            'Eb/N0 (dB)': ebno_dbs,
            'BER LS': ls_ber.numpy(),
            'BER MMSE': mmse_ber.numpy(),
            'BER GMM Circulant': gmm_circulant_ber.numpy(),
            'BER GMM Full': gmm_full_ber.numpy(),
            'BER Sample Covariance': sample_cov_ber.numpy(),
            'BER Real': real_ber.numpy()
        }

        df = pd.DataFrame(ber_data)


        base_dir = './simulation_results/ber'

        # Subdirectories for CSV and plots
        csv_dir = os.path.join(base_dir, 'csv')
        plots_dir = os.path.join(base_dir, 'plots')

        # Create these directories if they don't exist
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # Define full paths for the CSV file and plot image
        sim_results_csv = os.path.join(csv_dir, f"BER_{n_antennas}x{n_coherence}x{batch_size}x{n_gmm_components}x{monte_carlo_iterations}x{code}x{code_rate}{'xlist' if (code == 'polar') else ''}x{step_size}.csv")
        sim_results_plot = os.path.join(plots_dir, f"BER_{n_antennas}x{n_coherence}x{batch_size}x{n_gmm_components}x{monte_carlo_iterations}x{code}x{code_rate}{'xlist' if (code == 'polar') else ''}x{step_size}.png")

        # Save the DataFrame and the plot
        df.to_csv(sim_results_csv, index=False)

        # plot all three ber curves over ebno_db
        plt.figure()
        plt.plot(ebno_dbs, ls_ber.numpy(), label='LS')
        plt.plot(ebno_dbs, mmse_ber.numpy(), label='MMSE')
        plt.plot(ebno_dbs, gmm_circulant_ber.numpy(), label='GMM Circulant')
        plt.plot(ebno_dbs, gmm_full_ber.numpy(), label='GMM Full')
        plt.plot(ebno_dbs, sample_cov_ber.numpy(), label='Sample Covariance')
        plt.plot(ebno_dbs, real_ber.numpy(), label='Real')
        plt.xlabel('Eb/N0 [dB]')
        plt.ylabel('BER')
        plt.yscale('log')
        plt.legend()
        plt.savefig(sim_results_plot)
        plt.show()
        return
        
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
            vertically_stacked_h_j, vertically_stacked_h_hat_j = uncoded_e2e_model_nmse_ls(batch_size=batch_size, ebno_db=(-5 + 2*j))
            vertically_stacked_nmse_list_ls.append(tf.reduce_sum(tf.square(tf.abs(vertically_stacked_h_j - vertically_stacked_h_hat_j))) / (batch_size * n_antennas))

            vertically_stacked_h_j, vertically_stacked_h_hat_j = uncoded_e2e_model_nmse_mmse(batch_size=batch_size, ebno_db=(-5 + 2*j))
            vertically_stacked_nmse_list_mmse.append(tf.reduce_sum(tf.square(tf.abs(vertically_stacked_h_j - vertically_stacked_h_hat_j))) / (batch_size * n_antennas))

            vertically_stacked_h_j, vertically_stacked_h_hat_j = uncoded_e2e_model_nmse_gmm_circulant(batch_size=batch_size, ebno_db=(-5 + 2*j))
            vertically_stacked_nmse_list_gmm_circulant.append(tf.reduce_sum(tf.square(tf.abs(vertically_stacked_h_j - vertically_stacked_h_hat_j))) / (batch_size * n_antennas))
           
            vertically_stacked_h_j, vertically_stacked_h_hat_j = uncoded_e2e_model_nmse_gmm_full(batch_size=batch_size, ebno_db=(-5 + 2*j))
            vertically_stacked_nmse_list_gmm_full.append(tf.reduce_sum(tf.square(tf.abs(vertically_stacked_h_j - vertically_stacked_h_hat_j))) / (batch_size * n_antennas))

            vertically_stacked_h_j, vertically_stacked_h_hat_j = uncoded_e2e_model_nmse_sample_cov(batch_size=batch_size, ebno_db=(-5 + 2*j))
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


        base_dir = './simulation_results/nmse'

        # Subdirectories for CSV and plots
        csv_dir = os.path.join(base_dir, 'csv')
        plots_dir = os.path.join(base_dir, 'plots')

        # Create these directories if they don't exist
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # Define full paths for the CSV file and plot image
        sim_results_csv = os.path.join(csv_dir, f'NMSE_{n_antennas}x{n_coherence}x{batch_size}x{n_gmm_components}.csv')
        sim_results_plot = os.path.join(plots_dir, f'NMSE_{n_antennas}x{n_coherence}x{batch_size}x{n_gmm_components}.png')

        # Save the DataFrame and the plot
        df.to_csv(sim_results_csv, index=False)

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
