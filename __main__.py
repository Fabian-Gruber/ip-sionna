from end2endModel import end2endModel as e2e

try:
    import sionna as sn
except AttributeError:
    import sionna as sn
    
import numpy as np

def __main__():
    num_bits_per_symbol = 2
    block_length = 1024
    ebno_db_min = -10.0 # Minimum value of Eb/N0 [dB] for simulations
    ebno_db_max = 10.0 # Maximum value of Eb/N0 [dB] for simulations
    batch_size = 1024 # How many examples are processed by Sionna in parallel
    n_coherence = 1
    n_antennas = 32
    genie_estimator = False

    uncoded_e2e_model = e2e(num_bits_per_symbol=num_bits_per_symbol, block_length=block_length, n_coherence=n_coherence, n_antennas=n_antennas, genie_estimator=genie_estimator)

    ber_plots = sn.utils.PlotBER("Uncoded BER")
    ber_plots.simulate(
        uncoded_e2e_model,
        ebno_dbs=np.linspace(ebno_db_min, ebno_db_max, 20),
        batch_size=batch_size,
        num_target_block_errors=100, # simulate until 100 block errors occured
        legend="Uncoded",
        soft_estimates=True,
        max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
        show_fig=True   
    )
    
if __name__ == "__main__":
    __main__()
