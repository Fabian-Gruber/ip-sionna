from end2endModel import end2endModel as e2e
    
import numpy as np
import pandas as pd
import os
import sys
import sionna as sn

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class BER_NMSE_Simulation:
    def __init__(self, config):
        self.config = config
        self.init_models()
        self.results = {estimator: np.zeros(config['iterations']) for estimator in self.estimators}
        self.total_block_errors = {estimator: np.zeros(config['iterations']) for estimator in self.estimators}
        self.mc_iterations = {estimator: np.zeros(config['iterations']) for estimator in self.estimators}

    def init_models(self):
        """Initialize all required models based on configuration."""
        self.models = {}
        self.estimators = ['ls', 'mmse', 'gmm_circulant', 'gmm_full', 'sample_cov', 'real']
        batch_size = self.config['batch_size']
        block_length = self.config['block_length']
        num_bits_per_symbol = self.config['num_bits_per_symbol']
        n_coherence = self.config['n_coherence']
        n_antennas = self.config['n_antennas']
        n_gmm_components = self.config['n_gmm_components']
        code_rate = self.config['code_rate']
        code = self.config['code']
        n_blocks = self.config['n_blocks']

        for estimator in self.estimators:
            model = e2e(
                num_bits_per_symbol=num_bits_per_symbol,
                block_length=block_length,
                n_coherence=n_coherence,
                n_antennas=n_antennas,
                training_batch_size=(100000 if (estimator == 'gmm_full' or estimator == 'sample_cov') else 30000 if estimator == 'gmm_circulant' else None),
                covariance_type=('full' if (estimator == 'gmm_full' or estimator == 'sample_cov') else 'circulant' if estimator == 'gmm_circulant' else None),
                n_gmm_components=(n_gmm_components if (estimator == 'gmm_full' or estimator == 'gmm_circulant') else 1 if estimator == 'sample_cov' else None),
                estimator=(estimator if (estimator != 'gmm_circulant' and estimator != 'gmm_full' and estimator != 'sample_cov') else 'gmm'),
                output_quantity=self.config['output_quantity'],
                code_rate=code_rate,
                code=code,
                n_blocks=n_blocks
            )
            print(f"Initialized {estimator.upper()} model.")
            self.models[estimator] = model

    # The following methods are used to run BER and NMSE simulation and collect the results
    def run_simulation(self):
        """Run the BER or NMSE simulation across all estimators and conditions."""
        for estimator in self.estimators:
            print(f"Running {estimator.upper()} simulation...")
            for j in range(self.config['iterations']):
                ebno_db = self.config['ebno_dbs'][j]
                if self.config['output_quantity'] == 'ber':
                    self.simulate_ber_estimator(estimator, ebno_db, j)
                    if self.results[estimator][j] == 0 and self.mc_iterations[estimator][j] > 0:
                        # for k in range(j + 1, self.config['iterations']):
                        #     self.results[estimator][k] = 0
                        #     self.total_block_errors[estimator][k] = 0
                        #     self.mc_iterations[estimator][k] = 0
                        print(f"Stopping early at Eb/N0 {ebno_db}dB due to zero BER.")
                        break
                elif self.config['output_quantity'] == 'nmse':
                    self.simulate_nmse_estimator(estimator, ebno_db, j)

    def simulate_ber_estimator(self, estimator, ebno_db, iteration):
        model = self.models[estimator]
        ber_accumulated = 0
        block_errors = 0
        mc_count = 0
        last_line_length = 0  # Track the length of the last printed line

        for mc in range(self.config['monte_carlo_iterations']):
            bits, llrs = model(batch_size=self.config['batch_size'], ebno_db=ebno_db)
            bits_hat = tf.where(llrs > 0, tf.ones_like(bits), tf.zeros_like(bits))
            bit_errors_bool = tf.not_equal(bits, bits_hat)
            block_errors_bool = tf.reduce_any(bit_errors_bool, axis=1)
            block_errors += tf.reduce_sum(tf.cast(block_errors_bool, dtype=tf.int32))

            bit_errors = tf.cast(bit_errors_bool, dtype=tf.float32)
            ber_accumulated += tf.reduce_sum(bit_errors) / (self.config['batch_size'] * self.config['n_blocks'] * (self.config['block_length'] - self.config['num_bits_per_symbol']))

            current_line = f"\r{estimator.upper()} {ebno_db}dB: Current BER: {ber_accumulated / (mc + 1)} ||| Blocks with Errors: {block_errors} ||| MC Run {mc + 1}/{self.config['monte_carlo_iterations']}"
            sys.stdout.write(current_line)
            sys.stdout.flush()
            last_line_length = len(current_line)

            if block_errors > self.config['n_target_block_errors'] and mc > 0:
                # Clear the current line before printing the stop message
                sys.stdout.write('\r' + ' ' * last_line_length + '\r')
                sys.stdout.flush()
                print(f"Stopping early at Eb/N0 {ebno_db}dB due to exceeding block error target.")
                break
            mc_count += 1

        # Clear the line one more time before printing final results
        sys.stdout.write('\r' + ' ' * last_line_length + '\r')
        sys.stdout.flush()
        # Print the final results
        self.results[estimator][iteration] = ber_accumulated / mc_count
        self.total_block_errors[estimator][iteration] = block_errors
        self.mc_iterations[estimator][iteration] = mc_count
        print(f"{estimator.upper()} {ebno_db}dB: Final BER: {self.results[estimator][iteration]} ||| Total Blocks with Errors: {block_errors} ||| Total MC Iterations: {mc_count}")

    def simulate_nmse_estimator(self, estimator, ebno_db, iteration):
        model = self.models[estimator]
        h, h_hat = model(batch_size=self.config['batch_size'], ebno_db=ebno_db)
        nmse = tf.reduce_mean(tf.square(tf.abs(h - h_hat))).numpy()
        self.results[estimator][iteration] = nmse
        print(f"{estimator.upper()} Eb/N0 {ebno_db}dB: NMSE: {nmse}")        

    def plot_results(self, plots_dir):
        """Plot the simulation results."""
        path = os.path.join(plots_dir, f"BER_{self.config['n_antennas']}x{self.config['n_coherence']}x{self.config['batch_size']}x{self.config['n_gmm_components']}x{self.config['monte_carlo_iterations']}x{self.config['code']}x{self.config['code_rate']}{'xlist' if (self.config['code'] == 'polar') else ''}x{self.config['n_blocks']}.png")
        plt.figure()
        for est in self.estimators:
            plt.plot(self.config['ebno_dbs'], self.results[est], label=f'{est.upper()} BER')
        plt.xlabel('Eb/N0 [dB]')
        plt.ylabel(self.config['output_quantity'].upper())
        plt.yscale('log')
        plt.legend()
        plt.savefig(path)

    def save_results(self, csv_dir):
        """Save the simulation results to CSV."""
        path = os.path.join(csv_dir, f"BER_{self.config['n_antennas']}x{self.config['n_coherence']}x{self.config['batch_size']}x{self.config['n_gmm_components']}x{self.config['monte_carlo_iterations']}x{self.config['code']}x{self.config['code_rate']}{'xlist' if (self.config['code'] == 'polar') else ''}x{self.config['n_blocks']}.csv")
        data = {'Eb/N0 (dB)': self.config['ebno_dbs']}
        for est in self.estimators:
            print('results: ', self.results[est])
            print('results length: ', len(self.results[est]))
            data[f"{est.upper()} {self.config['output_quantity'].upper()}"] = self.results[est]
            print('data: ', data)
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    # The following methods are used to run BER simulation for different numbers of GMM Components
    def run_gmm_full_comparison(self, gmm_components):
        """Run simulations for 'gmm_full' with different numbers of GMM components."""
        self.gmm_results = {}
        base_params = {
            'num_bits_per_symbol': self.config['num_bits_per_symbol'],
            'block_length': self.config['block_length'],
            'n_coherence': self.config['n_coherence'],
            'n_antennas': self.config['n_antennas'],
            'code_rate': self.config['code_rate'],
            'code': self.config['code'],
            'n_blocks': self.config['n_blocks'],
            'training_batch_size': 100000,
            'covariance_type': 'full',
            'output_quantity': self.config['output_quantity']
        }

        # Iterate over each GMM component setting
        for comp in gmm_components:
            print(f"Running GMM Full with {comp} components")
            params = base_params.copy()
            params['n_gmm_components'] = comp
            self.models['gmm_full'] = e2e(estimator='gmm', **params)

            # Initialize results storage for this component count
            self.gmm_results[comp] = []

            for j, ebno_db in enumerate(self.config['ebno_dbs']):
                print(f"Simulating Eb/N0 = {ebno_db} dB for {comp} components")
                self.simulate_ber_estimator('gmm_full', ebno_db, j)
                # Collect results specifically for this component count and Eb/N0
                self.gmm_results[comp].append(self.results['gmm_full'][j])
        
        self.plot_gmm_full_results(gmm_components)
        self.save_gmm_full_results(gmm_components)

    def plot_gmm_full_results(self, gmm_components):
        """Plot the BER results for the GMM Full estimator across different component counts."""
        plt.figure()
        for comp in gmm_components:
            plt.plot(self.config['ebno_dbs'], self.gmm_results[comp], label=f'GMM Full {comp} Components')
        plt.xlabel('Eb/N0 [dB]')
        plt.ylabel('BER' if self.config['output_quantity'] == 'ber' else 'NMSE')
        plt.yscale('log' if self.config['output_quantity'] == 'ber' else 'linear')
        plt.title('GMM Full Performance Comparison')
        plt.legend()
        plt.savefig(f"{self.config['base_dir']}/gmm_full_comparison.png")

    def save_gmm_full_results(self, gmm_components):
        """Save the GMM Full simulation results to CSV."""
        data = {'Eb/N0 (dB)': self.config['ebno_dbs']}
        for comp in gmm_components:
            data[f'GMM Full {comp} {self.config["output_quantity"].upper()}'] = self.gmm_results[comp]
        df = pd.DataFrame(data)
        df.to_csv(f"{self.config['base_dir']}/gmm_full_comparison.csv", index=False)

def __main__():
    config = {
        'num_bits_per_symbol': 2,
        'block_length': 256,
        'ebno_db_min': -5.0,
        'ebno_db_max': 15.0,
        'batch_size': 500,
        'n_coherence': 1,
        'n_antennas': 64,
        'n_gmm_components': 32,
        'iterations': 11,
        'ebno_dbs': np.linspace(-5.0, 15.0, 11),
        'code_rate': 0.5,
        'code': 'ldpc',
        'monte_carlo_iterations': 100,
        'n_target_block_errors': 250,
        'n_blocks': 5,
        'output_quantity': 'ber',  # Change this to 'nmse' for NMSE simulations, to 'ber' for BER Simulation
        'datatype': '3gpp', # Change this to '3gpp' for using 3gpp channels, 'measurement' for measured channels
        'base_dir': './simulation_results',
    }
    simulation_type = 'all'  # Change this to 'gmm_full' to run GMM Full comparison or to 'all' to compare all estimators
    
    csv_dir = os.path.join(config['base_dir'], config['output_quantity'], 'csv')
    plots_dir = os.path.join(config['base_dir'], config['output_quantity'], 'plots')
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    simulation = BER_NMSE_Simulation(config)

    if simulation_type == 'gmm_full':
        simulation.run_gmm_full_comparison([1, 2, 4, 8, 16, 32, 64])
    elif simulation_type == 'all':
        simulation.run_simulation()
        simulation.save_results(csv_dir)
        simulation.plot_results(plots_dir)

if __name__ == "__main__":
    __main__()
