#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:46:20 2024

@author: tuanaoyuncu
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

WD = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD")
DATA = WD / "Data"
PWD = DATA / "C_001" / "H_085_C1" / "S_009_C1" 
c_out_files_path = PWD / "TOA_MGA_20231020_009_000001_t_processed.npy"
t_conv_files_path = PWD / "TOA_MGA_20231020_009_000001_x_processed.npy"
c_out = np.load(c_out_files_path)
t_conv = np.load(t_conv_files_path)
plt.step(c_out, t_conv, where='post')
plt.xlabel('t/s')
plt.ylabel('x/1')
plt.show()


WD = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD")
DATA = WD / "Data"
PWD = DATA / "C_001" / "H_085_C1" / "S_009_C1" 

# Define the file pattern
file_pattern = "TOA_MGA_20231020_009_*"


c_out_files_path = sorted(glob.glob(str(PWD / f"{file_pattern}x_processed.npy")))
t_conv_files_path = sorted(glob.glob(str(PWD / f"{file_pattern}t_processed.npy")))

# Load data
c_out_list = [np.load(file_path) for file_path in c_out_files_path]
t_conv_list = [np.load(file_path) for file_path in t_conv_files_path]
for file
  #for file_num in [19]:
      if file_num in even:
         continue
      process_data(file_num)

# Plot the step functions
for c_out, t_conv in zip(c_out_list, t_conv_list):
    plt.step(t_conv, c_out, where='post')

plt.xlabel('t/s')
plt.ylabel('x/1')
plt.show()





