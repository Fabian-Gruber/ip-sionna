from .scm_multi import SCMMulti
from scipy.linalg import toeplitz
import numpy as np

def channel_generation(batch_size, n_coherences, n_antennas):
    """
    SIMO version.
    """
    path_sigma = 2.0
    n_path = 1
    channel = SCMMulti(path_sigma=path_sigma, n_path=n_path)

    # generate channel samples with a certain batch size
    rng = np.random.default_rng(1235428719812346)

    h, t = channel.generate_channel(batch_size, n_coherences, n_antennas, rng)
        
    # Initialize an empty array to store the covariance matrices
    C = np.empty((batch_size, n_antennas, n_antennas), dtype=np.complex64)
    
    for i in range(batch_size):
        # Calculate the covariance matrix for each sample
        C[i, :, :] = np.transpose(toeplitz(t[i, :]))
    
    # print('Generated ' + str(batch_size) + ' SIMO channel samples of size ' + str(n_antennas) + 'x1.')
    
    return h, C
