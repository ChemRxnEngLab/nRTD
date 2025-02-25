#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:18:54 2024

@author: tuanaoyuncu
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Base directory for the data
base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Disc_variation/CNN_1st'
discs = [100, 200, 300, 400, 500, 600]  
num_plots = 6  # Number of subplots to display

# Set up the plot with 3x2 subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 10))

for i, disc in enumerate(discs[:num_plots], start=1):
    # Set up directory and file paths for each disc value
    disc_dir = os.path.join(base_dir, f'Disc_{disc}')
    t_expected_path = os.path.join(disc_dir, f't_E_expected_first_layer_{disc}.npy')
    t_predicted_path = os.path.join(disc_dir, f't_E_predicted_first_layer_{disc}.npy')
    E_expected_path = os.path.join(disc_dir, f'E_expected_first_layer_{disc}.npy')
    E_predicted_path = os.path.join(disc_dir, f'E_predicted_first_layer_{disc}.npy')

    # Load data if files exist
    if all(os.path.exists(path) for path in [t_expected_path, t_predicted_path, E_expected_path, E_predicted_path]):
        t_expected = np.load(t_expected_path)
        t_predicted = np.load(t_predicted_path)
        E_expected = np.load(E_expected_path)
        E_predicted = np.load(E_predicted_path)
        
        # Check and handle if data shapes mismatch
        if t_expected.shape != E_expected.shape:
            print(f"Expected data shape mismatch for Disc {disc}: {t_expected.shape} vs {E_expected.shape}. Skipping...")
            continue
        if t_predicted.shape != E_predicted.shape:
            print(f"Predicted data shape mismatch for Disc {disc}: {t_predicted.shape} vs {E_predicted.shape}. Skipping...")
            continue
        
        # Determine subplot row and column based on index
        row, col = divmod(i - 1, 2)
        
        # Plot expected and predicted E data
        axs[row, col].plot(t_expected, E_expected, label=f'Expected (Disc {disc})', color='blue')
        axs[row, col].plot(t_predicted, E_predicted, label=f'Predicted (Disc {disc})', color='purple', linestyle='--')
        
        # Set labels and legend
        axs[row, col].set_title(f'Disc {disc}')
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('E')
        axs[row, col].legend()
    else:
        print(f"One or more data files not found for Disc {disc} in {disc_dir}. Skipping...")

# Adjust layout and display the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
