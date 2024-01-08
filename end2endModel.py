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

    def __init__(self, num_bits_per_symbol, block_length, n_coherence, n_antennas, estimator, output_quantity, training_batch_size=None, covariance_type=None, n_gmm_components=None, gmm_max_iterations=500, code_rate=0):
        super().__init__()
        
        self.estimator = estimator
        self.output_quantity = output_quantity
        self.code_rate = code_rate
        
        self.n_coherence = n_coherence
        self.n_antennas = n_antennas
        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = block_length
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        if self.code_rate != 0:
            self.encoder = sn.fec.ldpc.LDPC5GEncoder(self.block_length - self.num_bits_per_symbol, int((self.block_length - self.num_bits_per_symbol) / self.code_rate))
            self.decoder = sn.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=False)
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
                                coderate=1.0)
        
        
        # #print('value of no: ', no)

        bits = self.binary_source([batch_size, self.block_length]) # Blocklength set to 1024 bits
        bits = bits[:, self.num_bits_per_symbol:]

        if self.code_rate != 0:
            coded_bits = self.encoder(bits)
            x = self.mapper(coded_bits)
        else:
            x = self.mapper(bits)
                                        
        pilot = tf.ones([batch_size,1,1], dtype=tf.complex64)
                        
        y_p, h, C = self.channel(pilot, no, batch_size, self.n_coherence, self.n_antennas)

        if self.estimator == 'ls':

            h_hat_ls = self.ls_estimator(y_p, pilot)
            # print(f'difference between h_hat_ls and h at {ebno_db}DB: ', tf.reduce_sum(tf.square(tf.abs(h - h_hat_ls))) / tf.cast(batch_size * self.n_antennas, dtype=tf.float32))

            if self.output_quantity == 'nmse':
                return h, h_hat_ls

            #uplink phase
            else:
                y = []
                x_hat_ls = []
                no_ls_new = []

                if self.code_rate != 0:
                    llr_ls = tf.TensorArray(dtype=tf.float32, size=tf.cast(self.block_length - self.num_bits_per_symbol, dtype=tf.int32))

                    for i in range(tf.shape(x)[1]):
                        y_i = self.channel(x[:, i], no, batch_size, self.n_coherence, self.n_antennas, h, C)[0]
                        y.append(y_i)
                    
                        x_hat_ls_i, no_ls_new_i = self.equalizer(h_hat_ls, y_i, no)
                        x_hat_ls.append(x_hat_ls_i)
                        no_ls_new.append(no_ls_new_i)
                        

                        llr_ls_i = self.demapper([x_hat_ls_i, no_ls_new_i])
                        llr_ls = llr_ls.write(i, llr_ls_i)
                    
                    llr_ls = llr_ls.stack()
                    llr_ls = tf.transpose(llr_ls, perm=[1, 0, 2])
                    llr_ls = tf.split(llr_ls, num_or_size_splits=2, axis=2)
                    llr_ls = tf.reshape(tf.stack(llr_ls, axis=2), coded_bits.shape)
                    llr_ls = self.decoder(llr_ls)

                else: 
                    llr_ls = tf.TensorArray(dtype=tf.float32, size=tf.cast(self.block_length / self.num_bits_per_symbol - 1, dtype=tf.int32))

                    for i in range(tf.shape(x)[1]):
                        #y = h * x + n for all x except first one
                        y_i = self.channel(x[:, i], no, batch_size, self.n_coherence, self.n_antennas, h, C)[0]
                        y.append(y_i)

                        x_hat_ls_i, no_ls_new_i = self.equalizer(h_hat_ls, y_i, no)
                        x_hat_ls.append(x_hat_ls_i)
                        no_ls_new.append(no_ls_new_i)

                        llr_ls_i = self.demapper([x_hat_ls_i, no_ls_new_i])
                        llr_ls = llr_ls.write(i, llr_ls_i)
                    
                    llr_ls = llr_ls.stack()
                    llr_ls = tf.transpose(llr_ls, perm=[1, 0, 2])
                    llr_ls = tf.split(llr_ls, num_or_size_splits=2, axis=2)
                    llr_ls = tf.reshape(tf.stack(llr_ls, axis=2), bits.shape)
                
                bits_hat = tf.where(llr_ls > 0, tf.ones_like(llr_ls), tf.zeros_like(llr_ls))

                print(f'bit error rate ls at {ebno_db}DB: ', tf.reduce_sum(tf.cast(tf.not_equal(bits, bits_hat), dtype=tf.float32)) / tf.cast(batch_size * self.block_length, dtype=tf.float32))

                return bits, llr_ls
            

        elif self.estimator == 'mmse':

            h_hat_mmse = self.mmse_estimator(y_p, no, C, pilot)
            # print(f'difference between h_hat_mmse and h at {ebno_db}DB: ', tf.reduce_sum(tf.square(tf.abs(h - h_hat_mmse))) / tf.cast(batch_size * self.n_antennas, dtype=tf.float32))

            if self.output_quantity == 'nmse':
                return h, h_hat_mmse
            
            #uplink phase
            else:
                y = []
                x_hat_mmse = []
                no_mmse_new = []

                if self.code_rate != 0:
                    llr_mmse = tf.TensorArray(dtype=tf.float32, size=tf.cast(self.block_length - self.num_bits_per_symbol, dtype=tf.int32))

                    for i in range(tf.shape(x)[1]):
                        y_i = self.channel(x[:, i], no, batch_size, self.n_coherence, self.n_antennas, h, C)[0]
                        y.append(y_i)
                    
                        x_hat_mmse_i, no_mmse_new_i = self.equalizer(h_hat_mmse, y_i, no)
                        x_hat_mmse.append(x_hat_mmse_i)
                        no_mmse_new.append(no_mmse_new_i)
                        

                        llr_mmse_i = self.demapper([x_hat_mmse_i, no_mmse_new_i])
                        llr_mmse = llr_mmse.write(i, llr_mmse_i)
                    
                    llr_mmse = llr_mmse.stack()
                    llr_mmse = tf.transpose(llr_mmse, perm=[1, 0, 2])
                    llr_mmse = tf.split(llr_mmse, num_or_size_splits=2, axis=2)
                    llr_mmse = tf.reshape(tf.stack(llr_mmse, axis=2), coded_bits.shape)
                    llr_mmse = self.decoder(llr_mmse)

                else: 
                    llr_mmse = tf.TensorArray(dtype=tf.float32, size=tf.cast(self.block_length / self.num_bits_per_symbol - 1, dtype=tf.int32))

                    for i in range(tf.shape(x)[1]):
                        #y = h * x + n for all x except first one
                        y_i = self.channel(x[:, i], no, batch_size, self.n_coherence, self.n_antennas, h, C)[0]
                        y.append(y_i)

                        x_hat_mmse_i, no_mmse_new_i = self.equalizer(h_hat_mmse, y_i, no)
                        x_hat_mmse.append(x_hat_mmse_i)
                        no_mmse_new.append(no_mmse_new_i)

                        llr_mmse_i = self.demapper([x_hat_mmse_i, no_mmse_new_i])
                        llr_mmse = llr_mmse.write(i, llr_mmse_i)
                    
                    llr_mmse = llr_mmse.stack()
                    llr_mmse = tf.transpose(llr_mmse, perm=[1, 0, 2])
                    llr_mmse = tf.split(llr_mmse, num_or_size_splits=2, axis=2)
                    llr_mmse = tf.reshape(tf.stack(llr_mmse, axis=2), bits.shape)

                bits_hat = tf.where(llr_mmse > 0, tf.ones_like(llr_mmse), tf.zeros_like(llr_mmse))

                print(f'bit error rate mmse at {ebno_db}DB: ', tf.reduce_sum(tf.cast(tf.not_equal(bits, bits_hat), dtype=tf.float32)) / tf.cast(batch_size * self.block_length, dtype=tf.float32))

                return bits, llr_mmse

        elif self.estimator == 'gmm':

            #convert y_p from data type tf.complex64 to np.complex64
            y_p_np = y_p.numpy().astype(np.complex64)

            y_p_np = np.squeeze(y_p_np, axis=1)

            h_hat_gmm = self.gmm_estimator.estimate_from_y(y_p_np, ebno_db, self.n_antennas, n_summands_or_proba='all')

            h_hat_gmm = tf.cast(h_hat_gmm, dtype=tf.complex64)

            h_hat_gmm = tf.reshape(h_hat_gmm, (batch_size, 1, self.n_antennas))
            # print(f'difference between h_hat_gmm and h at {ebno_db}DB: ', tf.reduce_sum(tf.square(tf.abs(h - h_hat_gmm))) / tf.cast(batch_size * self.n_antennas, dtype=tf.float32))

            if self.output_quantity == 'nmse':
                return h, h_hat_gmm

            #uplink phase
            else: 
                y = []
                x_hat_gmm = []
                no_gmm_new = []

                if self.code_rate != 0:
                    llr_gmm = tf.TensorArray(dtype=tf.float32, size=tf.cast(self.block_length - self.num_bits_per_symbol, dtype=tf.int32))

                    for i in range(tf.shape(x)[1]):
                        y_i = self.channel(x[:, i], no, batch_size, self.n_coherence, self.n_antennas, h, C)[0]
                        y.append(y_i)
                    
                        x_hat_gmm_i, no_gmm_new_i = self.equalizer(h_hat_gmm, y_i, no)
                        x_hat_gmm.append(x_hat_gmm_i)
                        no_gmm_new.append(no_gmm_new_i)
                        

                        llr_gmm_i = self.demapper([x_hat_gmm_i, no_gmm_new_i])
                        llr_gmm = llr_gmm.write(i, llr_gmm_i)
                    
                    llr_gmm = llr_gmm.stack()
                    llr_gmm = tf.transpose(llr_gmm, perm=[1, 0, 2])
                    llr_gmm = tf.split(llr_gmm, num_or_size_splits=2, axis=2)
                    llr_gmm = tf.reshape(tf.stack(llr_gmm, axis=2), coded_bits.shape)
                    llr_gmm = self.decoder(llr_gmm)

                else:
                    llr_gmm = tf.TensorArray(dtype=tf.float32, size=tf.cast(self.block_length / self.num_bits_per_symbol - 1, dtype=tf.int32))

                    for i in range(tf.shape(x)[1]):
                        #y = h * x + n for all x except first one
                        y_i = self.channel(x[:, i], no, batch_size, self.n_coherence, self.n_antennas, h, C)[0]
                        y.append(y_i)

                        x_hat_gmm_i, no_gmm_new_i = self.equalizer(h_hat_gmm, y_i, no)
                        x_hat_gmm.append(x_hat_gmm_i)
                        no_gmm_new.append(no_gmm_new_i)

                        llr_gmm_i = self.demapper([x_hat_gmm_i, no_gmm_new_i])
                        llr_gmm = llr_gmm.write(i, llr_gmm_i)
                    
                    llr_gmm = llr_gmm.stack()
                    llr_gmm = tf.transpose(llr_gmm, perm=[1, 0, 2])
                    llr_gmm = tf.split(llr_gmm, num_or_size_splits=2, axis=2)
                    llr_gmm = tf.reshape(tf.stack(llr_gmm, axis=2), bits.shape)

                bits_hat = tf.where(llr_gmm > 0, tf.ones_like(llr_gmm), tf.zeros_like(llr_gmm))

                print(f'bit error rate gmm at {ebno_db}DB: ', tf.reduce_sum(tf.cast(tf.not_equal(bits, bits_hat), dtype=tf.float32)) / tf.cast(batch_size * self.block_length, dtype=tf.float32))

                return bits, llr_gmm
        else:
            y = []
            x_hat_real = []
            no_real_new = []

            if self.code_rate != 0:
                llr_real = tf.TensorArray(dtype=tf.float32, size=tf.cast(self.block_length - self.num_bits_per_symbol, dtype=tf.int32))

                for i in range(tf.shape(x)[1]):
                    y_i = self.channel(x[:, i], no, batch_size, self.n_coherence, self.n_antennas, h, C)[0]
                    y.append(y_i)
                
                    x_hat_real_i, no_real_new_i = self.equalizer(h, y_i, no)
                    x_hat_real.append(x_hat_real_i)
                    no_real_new.append(no_real_new_i)
                    

                    llr_real_i = self.demapper([x_hat_real_i, no_real_new_i])
                    llr_real = llr_real.write(i, llr_real_i)
                
                llr_real = llr_real.stack()
                llr_real = tf.transpose(llr_real, perm=[1, 0, 2])
                llr_real = tf.split(llr_real, num_or_size_splits=2, axis=2)
                llr_real = tf.reshape(tf.stack(llr_real, axis=2), coded_bits.shape)
                llr_real = self.decoder(llr_real)

            else:
                llr_real = tf.TensorArray(dtype=tf.float32, size=tf.cast(self.block_length / self.num_bits_per_symbol - 1, dtype=tf.int32))

                for i in range(tf.shape(x)[1]):
                    y_i = self.channel(x[:, i], no, batch_size, self.n_coherence, self.n_antennas, h, C)[0]
                    y.append(y_i)
                
                    x_hat_real_i, no_real_new_i = self.equalizer(h, y_i, no)
                    x_hat_real.append(x_hat_real_i)
                    no_real_new.append(no_real_new_i)
                    

                    llr_real_i = self.demapper([x_hat_real_i, no_real_new_i])
                    llr_real = llr_real.write(i, llr_real_i)

                llr_real = llr_real.stack()
                llr_real = tf.transpose(llr_real, perm=[1, 0, 2])
                llr_real = tf.split(llr_real, num_or_size_splits=2, axis=2)
                llr_real = tf.reshape(tf.stack(llr_real, axis=2), bits.shape)

            bits_hat = tf.where(llr_real > 0, tf.ones_like(llr_real), tf.zeros_like(llr_real))

            print(f'bit error rate real at {ebno_db}DB: ', tf.reduce_sum(tf.cast(tf.not_equal(bits, bits_hat), dtype=tf.float32)) / tf.cast(batch_size * self.block_length, dtype=tf.float32))

            return bits, llr_real