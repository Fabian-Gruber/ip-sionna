import os
import pandas as pd
import numpy as np
from model_3gpp.channel_generation import channel_generation

class training_samples:
    def __init__(self):
        super().__init__()
    
    def __call__(self, training_batch_size, n_coherence, n_antennas):
        # Filename based on parameters
        filename = f"training_samples_{training_batch_size}x{n_coherence}x{n_antennas}.csv"
        filepath = os.path.join("training_samples_gmm", "training_csv_files", filename)

        # Check if file exists
        if os.path.exists(filepath):
            # Read from CSV
            df = pd.read_csv(filepath, header=None)
            h = df.to_numpy()

            # Convert string representation of complex numbers to actual complex numbers
            h = np.array([[self.convert_to_complex(num) for num in row] for row in h], dtype=np.complex64)

            print(f"Data read from {filepath}")
        else:
            # If file doesn't exist, generate, save, and return h
            h, _ = channel_generation(training_batch_size, n_coherence, n_antennas)

            print('shape of h: ', h.shape)

            self.save_h_to_csv(h, filename)
            h = np.squeeze(h, axis=1)

        return h
    
    def save_h_to_csv(self, h, filename):
        # Create directory if it doesn't exist
        directory = os.path.join("training_samples_gmm", "training_csv_files")
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Path for the CSV file
        filepath = os.path.join(directory, filename)

        # Formatting and saving
        formatted_h = np.array([["{0.real}+{0.imag}j".format(num) for num in row.flatten()] for row in h])
        df = pd.DataFrame(formatted_h)
        df.to_csv(filepath, index=False, header=False)
        print(f"Data saved to {filepath}")
    
    def convert_to_complex(self, num_str):
        """
        Convert a string representation of a complex number to a complex number.
        Handles both 'real+imagj' and 'real-imagj' formats.
        """
        if 'j' not in num_str:
            return complex(float(num_str), 0.0)
        real_part, imag_part = num_str.split('j')[0].split('+') if '+' in num_str else num_str.split('j')[0].split('-')
        imag_part = imag_part + '1' if imag_part == '' else imag_part
        return complex(float(real_part), float(imag_part) * (1 if '+' in num_str else -1))

