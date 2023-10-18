try:
    import sionna as sn
except AttributeError:
    import sionna as sn
import tensorflow as tf

from .channels import cond_normal_channel as cnc

from .estimators import ls_estimator as lse
from .estimators import genie_mmse_estimator as gme

from .equalizers import equalizer as eq


class end2endModel(tf.keras.Model):

    def __init__(self, num_bits_per_symbol, block_length):
        super().__init__()
        
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
        x[0] = tf.cast(1.0, x[0].dtype)
                
        y_p, h, C, n = self.channel(x[0], no)
        
        h_hat_ls = self.ls_estimator(h, x[0])
        
        h_hat_mmse = self.mmse_estimator(y_p, no, C, x[0])
        
        #uplink phase
        
        #x_data = all x except first one
        x_data = x[1:]
        
        for i in range(len(x_data)):
            #y = h * x + n for all x except first one
            y, _, _ = self.channel(x_data[i], no)
        
        x_hat_ls, no_ls_new = self.equalizer(h_hat_ls, y, n)
        
        x_hat_mmse, no_mmse_new = self.equalizer(h_hat_mmse, y, n)
        
        llr_ls = self.demapper([x_hat_ls, no_ls_new])
        
        llr_mmse = self.demapper([x_hat_mmse, no_mmse_new])
        
        return bits, llr_ls, llr_mmse
        
                
        
        
