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

    def __init__(self, num_bits_per_symbol, block_length, n_coherence, n_antennas):
        super().__init__()
        
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

        bits = self.binary_source([batch_size, self.block_length]) # Blocklength set to 1024 bits
        x = self.mapper(bits)
                                
        pilot = tf.ones((batch_size, 1, 1), dtype=tf.complex64)
                        
        y_p, h, C = self.channel(pilot, no, batch_size, self.n_coherence, self.n_antennas)
        
        h_hat_ls = self.ls_estimator(h, pilot)
        
        h_hat_mmse = self.mmse_estimator(y_p, no, C, pilot)
        
        #uplink phase
        
        #x_data = all x except x[0][0] (x has shape (1, 512))
        x_data = x[:, 1:]
        
                        
        y = []
        x_hat_ls = []
        x_hat_mmse = []
        no_ls_new = []
        no_mmse_new = []
        llr_ls = tf.zeros((batch_size, 2, 0), dtype=tf.float32)
        llr_mmse = tf.zeros((batch_size, 2, 0), dtype=tf.float32)
                
        for i in range(tf.shape(x_data)[1]):
            #y = h * x + n for all x except first one
            y_i = self.channel(x_data[0][i], no, batch_size, self.n_coherence, self.n_antennas, h, C)[0]
            y.append(y_i)
        
            x_hat_ls_i, no_ls_new_i = self.equalizer(h_hat_ls, y_i, no)
            x_hat_ls.append(x_hat_ls_i)
            no_ls_new.append(no_ls_new_i)

            x_hat_mmse_i, no_mmse_new_i = self.equalizer(h_hat_mmse, y_i, no)
            x_hat_mmse.append(x_hat_mmse_i)
            no_mmse_new.append(no_mmse_new_i)
            
            llr_ls_i = self.demapper([x_hat_ls_i, no_ls_new_i])
            llr_ls = tf.concat([llr_ls, tf.reshape(llr_ls_i, (batch_size, 2, 1))], axis=2)
            
            llr_mmse_i = self.demapper([x_hat_mmse_i, no_mmse_new_i])
            llr_mmse = tf.concat([llr_mmse, tf.reshape(llr_mmse_i, (batch_size, 2, 1))], axis=2)
            
        bits = bits[:, self.num_bits_per_symbol:]
        
        llr_ls = tf.reshape(llr_ls, bits.shape)
        llr_mmse = tf.reshape(llr_mmse, bits.shape)
        
        return bits, llr_ls, llr_mmse
        
