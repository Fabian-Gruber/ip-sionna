from .scm_multi import SCMMulti
from scipy.linalg import toeplitz
import numpy as np

def channel_generation():
    """
    SIMO version.
    """
    path_sigma = 2.0
    n_path = 1
    channel = SCMMulti(path_sigma=path_sigma, n_path=n_path)

    # generate channel samples with certain batch size
    batch_size = 20
    n_coherences = 1
    n_antennas = 32
    rng = np.random.default_rng(1235428719812346)

    h, t = channel.generate_channel(batch_size, n_coherences, n_antennas, rng)

    # full covariance matrix of first sample
    C = toeplitz(t[0, :])

    print('Generated ' + str(batch_size) + ' SIMO channel samples of size ' + str(n_antennas) + 'x1.')
    
    return h, C