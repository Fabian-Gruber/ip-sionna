try:
    import sionna as sn
except AttributeError:
    import sionna as sn
import tensorflow as tf
import numpy as np
import time
import pickle
import os

from channels import cond_normal_channel as cnc

from estimators import ls_estimator as lse
from estimators import genie_mmse_estimator as gme
from gmm_model.gmm_cplx import Gmm as gmm_cplx

from training_samples_gmm import training_samples as ts

from equalizers import equalizer as eq


class end2endModel(tf.keras.Model):

    def __init__(self, num_bits_per_symbol, block_length, n_coherence, n_antennas, estimator, output_quantity, training_batch_size=None, covariance_type=None, n_gmm_components=None, gmm_max_iterations=500, code_rate=0, code='ldpc', n_blocks=1):
        super().__init__()
        
        self.estimator = estimator
        self.output_quantity = output_quantity
        self.code_rate = code_rate
        
        self.n_coherence = n_coherence
        self.n_antennas = n_antennas
        self.n_blocks = n_blocks
        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = block_length
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        if self.code_rate != 0:
            if code == 'ldpc':
                self.encoder = sn.fec.ldpc.LDPC5GEncoder(self.block_length - self.num_bits_per_symbol, int((self.block_length - self.num_bits_per_symbol) / self.code_rate))
                self.decoder = sn.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=False)
            elif code == 'polar':
                self.encoder = sn.fec.polar.encoding.Polar5GEncoder(self.block_length - self.num_bits_per_symbol, int((self.block_length - self.num_bits_per_symbol) / self.code_rate))
                self.decoder = sn.fec.polar.decoding.Polar5GDecoder(self.encoder, dec_type='SCL', list_size=2)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.channel = cnc.cond_normal_channel()
        
        self.ls_estimator = lse.ls_estimator()
        self.mmse_estimator = gme.genie_mmse_estimator()

        if self.estimator == 'gmm':
            model_dir = './training_samples_gmm/training_models'
            os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

            # Define the full path for the model file
            model_filename = os.path.join(model_dir, f'gmm_model_{n_gmm_components}x{training_batch_size}x{covariance_type}x{n_antennas}x{n_coherence}.pkl')
            print('filename:', model_filename)
            if os.path.exists(model_filename):
                print(f"Loading GMM model from {model_filename}")
                with open(model_filename, 'rb') as file:
                    self.gmm_estimator = pickle.load(file)
            else:
                self.gmm_estimator = gmm_cplx(
                    n_components = n_gmm_components,
                    random_state = 2,
                    max_iter = gmm_max_iterations,
                    verbose = 2,
                    n_init = 1,
                    covariance_type = covariance_type,
                )

                self.training_gmm = ts.training_samples()

                tic = time.time()

                h_training = self.training_gmm(training_batch_size, n_coherence, n_antennas)

                self.gmm_estimator.fit(h_training)

                toc = time.time()

                print(f"training done. ({toc - tic:.3f} s)")

                with open(model_filename, 'wb') as file:
                    pickle.dump(self.gmm_estimator, file)

        
        self.equalizer = eq.equalizer()
        
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation, num_bits_per_symbol=self.num_bits_per_symbol)
    
    def __call__(self, batch_size, ebno_db):
        
        #pilot phase
        
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=self.code_rate)
                                                
        pilot = tf.ones([batch_size,1,1], dtype=tf.complex64)
                        
        y_p, h, C = self.channel(pilot, no, batch_size, self.n_coherence, self.n_antennas)

        if self.estimator == 'ls':

            h_hat = self.ls_estimator(y_p, pilot)
            all_bits, all_llr_ls = self.transmit_data_blocks(h_hat, no, C, h, batch_size)
            if self.output_quantity == 'nmse':
                return h, h_hat
            else:
                return all_bits, all_llr_ls

        elif self.estimator == 'mmse':

            h_hat = self.mmse_estimator(y_p, no, C, pilot)
            all_bits, all_llr_mmse = self.transmit_data_blocks(h_hat, no, C, h, batch_size)
            if self.output_quantity == 'nmse':
                return h, h_hat
            else:
                return all_bits, all_llr_mmse

        elif self.estimator == 'gmm':

            y_p_np = y_p.numpy().astype(np.complex64)
            y_p_np = np.squeeze(y_p_np, axis=1)
            h_hat = tf.cast(self.gmm_estimator.estimate_from_y(y_p_np, ebno_db, self.n_antennas, n_summands_or_proba='all'), dtype=tf.complex64)
            h_hat = tf.reshape(h_hat, (batch_size, 1, self.n_antennas))
            all_bits, all_llr_gmm = self.transmit_data_blocks(h_hat, no, C, h, batch_size)
            if self.output_quantity == 'nmse':
                return h, h_hat
            else:
                return all_bits, all_llr_gmm
        else:
            all_bits, all_llr_gmm = self.transmit_data_blocks(h, no, C, h, batch_size)
            return all_bits, all_llr_gmm
    
    def transmit_data_blocks(self, h_hat, no, C, h, batch_size):
        all_bits = []
        all_llr = []
        for block_idx in range(self.n_blocks):  # Loop to reuse channel 'self.num_blocks' times
            bits = self.binary_source([batch_size, self.block_length])
            bits = bits[:, self.num_bits_per_symbol:]

            if self.code_rate != 0:
                coded_bits = self.encoder(bits)
                x = self.mapper(coded_bits)
            else:
                x = self.mapper(bits)

            y = []
            x_hat = []
            no_new = []

            llr = tf.TensorArray(dtype=tf.float32, size=tf.cast((self.block_length - self.num_bits_per_symbol) / (self.num_bits_per_symbol * (self.code_rate if self.code_rate != 0 else 1)), dtype=tf.int32))
            for i in range(tf.shape(x)[1]):
                y_i = self.channel(x[:, i], no, batch_size, self.n_coherence, self.n_antennas, h, C)[0]
                y.append(y_i)

                x_hat_i, no_new_i = self.equalizer(h_hat, y_i, no)
                x_hat.append(x_hat_i)
                no_new.append(no_new_i)

                llr_i = self.demapper([x_hat_i, no_new_i])
                llr = llr.write(i, llr_i)

            llr = llr.stack()
            llr = tf.transpose(llr, perm=[1, 0, 2])
            llr = tf.split(llr, num_or_size_splits=2, axis=2)
            if self.code_rate != 0:
                llr = tf.reshape(tf.stack(llr, axis=2), coded_bits.shape)
                llr = self.decoder(llr)
            else:
                llr = tf.reshape(tf.stack(llr, axis=2), bits.shape)

            all_bits.append(bits)
            all_llr.append(llr)

        # Stack the results to create a single tensor for each
        all_bits_tensor = tf.concat(all_bits, axis=0)
        all_llr_tensor = tf.concat(all_llr, axis=0)

        return all_bits_tensor, all_llr_tensor

