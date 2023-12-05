try:
    import sionna as sn
except AttributeError:
    import sionna as sn
import tensorflow as tf
import numpy as np
import time

from channels import cond_normal_channel as cnc

from estimators import ls_estimator as lse
from estimators import genie_mmse_estimator as gme
from estimators import gmm_estimator as gmm

from training_samples_gmm import training_samples as ts

from equalizers import equalizer as eq


class end2endModel(tf.keras.Model):

    def __init__(self, num_bits_per_symbol, block_length, n_coherence, n_antennas, training_batch_size, covariance_type, n_gmm_components, estimator, output_quantity):
        super().__init__()
        
        self.estimator = estimator
        self.output_quantity = output_quantity
        
        self.n_coherence = n_coherence
        self.n_antennas = n_antennas
        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = block_length
        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.channel = cnc.cond_normal_channel()
        
        self.ls_estimator = lse.ls_estimator()
        self.mmse_estimator = gme.genie_mmse_estimator()
        self.gmm_estimator = gmm.gmm_estimator(
            n_components = n_gmm_components,
            random_state = 2,
            max_iter = 500,
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
        
        
        self.equalizer = eq.equalizer()
        
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation, num_bits_per_symbol=self.num_bits_per_symbol)
    
    def __call__(self, batch_size, ebno_db):
        
        #pilot phase
        
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        
        
        #print('value of no: ', no)

        bits = self.binary_source([batch_size, self.block_length]) # Blocklength set to 1024 bits
        bits = bits[:, self.num_bits_per_symbol:]

        x = self.mapper(bits)
                                        
        pilot = tf.ones([batch_size,1,1], dtype=tf.complex64)
                        
        y_p, h, C = self.channel(pilot, no, batch_size, self.n_coherence, self.n_antennas)

        h_hat_ls = self.ls_estimator(y_p, pilot)

        h_hat_mmse = self.mmse_estimator(y_p, no, C, pilot)

        noise_covariance = tf.eye(C.shape[-1]) * no

        #convert y_p from data type tf.complex64 to np.complex64
        y_p_np = y_p.numpy().astype(np.complex64)

        y_p_np = np.squeeze(y_p_np, axis=1)

        h_hat_gmm = self.gmm_estimator.estimate(y_p_np, noise_covariance, self.n_antennas, n_components_or_probability=1.0)

        h_hat_gmm = tf.cast(h_hat_gmm, dtype=tf.complex64)

        h_hat_gmm = tf.reshape(h_hat_gmm, (batch_size, 1, self.n_antennas))
                        
        # print('difference between h_hat_ls and h: ', tf.reduce_sum(tf.square(tf.abs(h_hat_ls - tf.cast(h, dtype=tf.complex64)))) / tf.reduce_sum(tf.square(tf.abs(h))))
        # print('difference between h_hat_mmse and h: ', tf.reduce_sum(tf.square(tf.abs(h_hat_mmse - tf.cast(h, dtype=tf.complex64)))) / tf.reduce_sum(tf.square(tf.abs(h))))
        # print('difference between h_hat_gmm and h: ', tf.reduce_sum(tf.square(tf.abs(h_hat_gmm - tf.cast(h, dtype=tf.complex64)))) / tf.reduce_sum(tf.square(tf.abs(h))))
                                
        if self.output_quantity == 'nmse' and self.estimator == 'ls':
            return h, h_hat_ls
        elif self.output_quantity == 'nmse' and self.estimator == 'mmse':
            return h, h_hat_mmse
        elif self.output_quantity == 'nmse' and self.estimator == 'gmm':
            return h, h_hat_gmm

        #uplink phase
                
        #x_data = all x except x[0][0] (x has shape (batch_size, block_length / num_bits_per_symbol))
        
        # print('shape of x_data: ', x_data.shape)
                                        
        y = []
        x_hat_ls = []
        x_hat_mmse = []
        x_hat_gmm = []
        no_ls_new = []
        no_mmse_new = []
        no_gmm_new = []
        llr_ls = tf.TensorArray(dtype=tf.float32, size=tf.cast(self.block_length / self.num_bits_per_symbol - 1, dtype=tf.int32))
        llr_mmse = tf.TensorArray(dtype=tf.float32, size=tf.cast(self.block_length / self.num_bits_per_symbol - 1, dtype=tf.int32))
        llr_gmm = tf.TensorArray(dtype=tf.float32, size=tf.cast(self.block_length / self.num_bits_per_symbol - 1, dtype=tf.int32))
                
        for i in range(tf.shape(x)[1]):
            #y = h * x + n for all x except first one
            # print('x_data[0][i].shape: ', x_data[0][i].shape)
            y_i = self.channel(x[:, i], no, batch_size, self.n_coherence, self.n_antennas, h, C)[0]
            # print('y_i.shape: ', y_i.shape)
            y.append(y_i)
        
            x_hat_ls_i, no_ls_new_i = self.equalizer(h_hat_ls, y_i, no)
            # print('x_hat_ls_i.shape: ', x_hat_ls_i.shape)
            # print('no_ls_new_i.shape: ', no_ls_new_i.shape)
            x_hat_ls.append(x_hat_ls_i)
            no_ls_new.append(no_ls_new_i)
            
            # print('difference between x_hat_ls_i and x_data[:, i]: ', tf.reduce_sum(tf.abs(x_hat_ls_i - tf.reshape(x_data[:, i], [-1, 1]))))

            x_hat_mmse_i, no_mmse_new_i = self.equalizer(h_hat_mmse, y_i, no)
            x_hat_mmse.append(x_hat_mmse_i)
            no_mmse_new.append(no_mmse_new_i)
                        
            # print('difference between x_hat_mmse_i and x_data[:, i]: ', tf.reduce_sum(tf.abs(x_hat_mmse_i - tf.reshape(x_data[:, i], [-1, 1]))))
                        
            x_hat_gmm_i, no_gmm_new_i = self.equalizer(h_hat_gmm, y_i, no,)
            x_hat_gmm.append(x_hat_gmm_i)
            no_gmm_new.append(no_gmm_new_i)
            
            llr_ls_i = self.demapper([x_hat_ls_i, no_ls_new_i])
            #llr_ls = tf.concat([llr_ls, tf.reshape(llr_ls_i, (batch_size, 2, 1))], axis=2)
            llr_ls = llr_ls.write(i, llr_ls_i)
            
            llr_mmse_i = self.demapper([x_hat_mmse_i, no_mmse_new_i])
            # print('llr_mmse_i.shape: ', llr_mmse_i.shape)
            llr_mmse = llr_mmse.write(i, llr_mmse_i)
            #llr_mmse = tf.concat([llr_mmse, tf.reshape(llr_mmse_i, (batch_size, 2, 1))], axis=2)

            llr_gmm_i = self.demapper([x_hat_gmm_i, no_gmm_new_i])
            llr_gmm = llr_gmm.write(i, llr_gmm_i)
        
        
        llr_ls = llr_ls.stack()
        #print('llr_ls first two elements: ', llr_ls[0][0][0], llr_ls[0][0][1])
        # print('shape of bits: ', bits.shape)
        llr_ls = tf.transpose(llr_ls, perm=[1, 0, 2])
        #print('llrs_ls.shape after transpose: ', llr_ls.shape)
        llr_mmse = llr_mmse.stack()
        # print('llr_mmse.shape: ', llr_mmse.shape)
        llr_mmse = tf.transpose(llr_mmse, perm=[1, 0, 2])
        #print('llrs_mmse.shape after transpose: ', llr_mmse.shape)
        llr_gmm = llr_gmm.stack()
        llr_gmm = tf.transpose(llr_gmm, perm=[1, 0, 2])
        
        
        llr_ls = tf.split(llr_ls, num_or_size_splits=2, axis=2)
        llr_mmse = tf.split(llr_mmse, num_or_size_splits=2, axis=2)
        llr_gmm = tf.split(llr_gmm, num_or_size_splits=2, axis=2)
            
        
        llr_ls = tf.reshape(tf.stack(llr_ls, axis=2), bits.shape)
        llr_mmse = tf.reshape(tf.stack(llr_mmse, axis=2), bits.shape)
        llr_gmm = tf.reshape(tf.stack(llr_gmm, axis=2), bits.shape)
                        
        
        if self.estimator == 'mmse':
            return bits, llr_mmse
        elif self.estimator == 'ls':
            return bits, llr_ls
        else:
            return bits, llr_gmm