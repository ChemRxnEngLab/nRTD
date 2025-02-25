#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:15:06 2024

@author: tuanaoyuncu
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

WD = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD")
DATA = WD / "Data"
#PWD = DATA / "0001"/"C_001" / "H_085_C1" / "S_009_C1" 
PWD = DATA / "0001"/"C_002" / "H_085_C2" / "S_012_C2" 

def process_data(file_num):
    x_file_path = PWD / f"TOA_MGA_20231020_012_{file_num:06d}_x_processed.npy"
    t_file_path = PWD / f"TOA_MGA_20231020_012_{file_num:06d}_t_processed.npy"

    # Load _x.npy and _t.npy
    x = np.load(x_file_path)
    t = np.load(t_file_path)
    plt.plot(t, x, label=file_num)

def main():
    start_file_num = 1
    end_file_num = 20

    # Iterate over the file numbers and process each data file
    for file_num in range(start_file_num, end_file_num + 1):
        if file_num % 2 != 1:
            continue
        process_data(file_num)

    # Show the legend and plot
    plt.legend()
    plt.xlabel("t/s")
    plt.ylabel("x/1")
    plt.show()

if __name__ == "__main__":
    main()
