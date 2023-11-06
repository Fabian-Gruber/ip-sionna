try:
    import sionna as sn
except AttributeError:
    import sionna as sn
import tensorflow as tf

from channels import cond_normal_channel as cnc

from estimators import ls_estimator as lse
from estimators import genie_mmse_estimator as gme

from equalizers import equalizer as eq


class end2endModel(tf.keras.Model):

    def __init__(self, num_bits_per_symbol, block_length, n_coherence, n_antennas, genie_estimator):
        super().__init__()
        
        self.genie_estimator = genie_estimator
        
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
        
        self.equalizer = eq.equalizer()
        
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation, num_bits_per_symbol=self.num_bits_per_symbol)
        
    def __call__(self, batch_size, ebno_db):
        
        #pilot phase
        
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        
        #print('value of no: ', no)

        bits = self.binary_source([batch_size, self.block_length]) # Blocklength set to 1024 bits
        x = self.mapper(bits)
        
        # print('shape of x: ', x.shape)
                                
        pilot = tf.ones((batch_size, 1, 1), dtype=tf.complex64)
                        
        y_p, h, C = self.channel(pilot, no, batch_size, self.n_coherence, self.n_antennas)
              
        h_hat_ls = self.ls_estimator(y_p, pilot)

        h_hat_mmse = self.mmse_estimator(y_p, no, C, pilot)
        
        
                
        print('difference between h_hat_ls and h: ', tf.reduce_sum(tf.abs(h_hat_ls - h)))
        print('difference between h_hat_mmse and h: ', tf.reduce_sum(tf.abs(h_hat_mmse - tf.cast(h, dtype=tf.complex64))))
                                
        #uplink phase
                
        #x_data = all x except x[0][0] (x has shape (batch_size, block_length / num_bits_per_symbol))
        x_data = x[:, 1:]
        
        # print('shape of x_data: ', x_data.shape)
                                        
        y = []
        x_hat_ls = []
        x_hat_mmse = []
        no_ls_new = []
        no_mmse_new = []
        llr_ls = tf.TensorArray(dtype=tf.float32, size=tf.cast(self.block_length / self.num_bits_per_symbol - 1, dtype=tf.int32))
        llr_mmse = tf.TensorArray(dtype=tf.float32, size=tf.cast(self.block_length / self.num_bits_per_symbol - 1, dtype=tf.int32))
                
        for i in range(tf.shape(x_data)[1]):
            #y = h * x + n for all x except first one
            # print('x_data[0][i].shape: ', x_data[0][i].shape)
            y_i = self.channel(x_data[:, i], no, batch_size, self.n_coherence, self.n_antennas, h, C)[0]
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
                        
            #print('value of x_hat_mmse_i: ', x_hat_mmse_i)
            #print('value of no_mmse_new_i: ', no_mmse_new_i[0])
            
            llr_ls_i = self.demapper([x_hat_ls_i, no_ls_new_i])
            #llr_ls = tf.concat([llr_ls, tf.reshape(llr_ls_i, (batch_size, 2, 1))], axis=2)
            llr_ls = llr_ls.write(i, llr_ls_i)
            
            llr_mmse_i = self.demapper([x_hat_mmse_i, no_mmse_new_i])
            # print('llr_mmse_i.shape: ', llr_mmse_i.shape)
            llr_mmse = llr_mmse.write(i, llr_mmse_i)
            #llr_mmse = tf.concat([llr_mmse, tf.reshape(llr_mmse_i, (batch_size, 2, 1))], axis=2)
        
         
        bits = bits[:, self.num_bits_per_symbol:]
        
        llr_ls = llr_ls.stack()
        #print('llr_ls first two elements: ', llr_ls[0][0][0], llr_ls[0][0][1])
        # print('shape of bits: ', bits.shape)
        llr_ls = tf.transpose(llr_ls, perm=[1, 0, 2])
        #print('llrs_ls.shape after transpose: ', llr_ls.shape)
        llr_mmse = llr_mmse.stack()
        # print('llr_mmse.shape: ', llr_mmse.shape)
        llr_mmse = tf.transpose(llr_mmse, perm=[1, 0, 2])
        #print('llrs_mmse.shape after transpose: ', llr_mmse.shape)
        
        
        llr_ls = tf.split(llr_ls, num_or_size_splits=2, axis=2)
        llr_mmse = tf.split(llr_mmse, num_or_size_splits=2, axis=2)
            
        
        llr_ls = tf.reshape(tf.stack(llr_ls, axis=2), bits.shape)
        llr_mmse = tf.reshape(tf.stack(llr_mmse, axis=2), bits.shape)
        
        # bits_hat = tf.where(llr_ls > 0.0, tf.ones_like(bits), tf.zeros_like(bits))
        
        # print("bits_hat before reshaping:\n", bits_hat[:3, :6])
        # #print("llr_mmse before reshaping:\n", llr_mmse[:3, :3, :])
        # print("bits:\n", bits[:3, :6])  # 6 columns to account for interleaving
        #print('shape of llr_ls after reshape: ', llr_ls.shape)
        #print('llr_ls first two elements after reshape: ', llr_ls[0][0], llr_ls[0][1])
        # print('shape of llr_mmse after reshape: ', llr_mmse.shape)
        
        # bits_hat = tf.where(llr_ls > 0.0, tf.ones_like(bits), tf.zeros_like(bits))
        
        # print('first 30 bits in bits_hat: ', bits_hat[1][:30])
        # print('first 30 bits in bits: ', bits[1][:30])
        
        # mismatched_indices = tf.where(tf.not_equal(bits, bits_hat))
        # print(mismatched_indices[100: 150])

                        
        
        if self.genie_estimator:
            return bits, llr_mmse
        
        return bits, llr_ls
        
