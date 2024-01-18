#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:58:51 2024

@author: tuanaoyuncu
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

WD = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD")
DATA = WD / "Data"
PWD = DATA / "C_001" / "H_185_C1" / "S_007_C1" 

def process_data(file_num, ax):
    x_file_path = PWD / f"TOA_MGA_20231013_007_{file_num:06d}_x_processed.npy"
    t_file_path = PWD / f"TOA_MGA_20231013_007_{file_num:06d}_t_processed.npy"
    x = np.load(x_file_path)
    t = np.load(t_file_path)
    ax.plot(t, x, label=f"File {file_num}")

def main():
    file_numbers = range(1, 20, 2)
    fig, axes = plt.subplots(10, 1, figsize=(10, 20))
    for i, file_num in enumerate(file_numbers):
        ax = axes[i]
        process_data(file_num, ax)
        process_data(file_num + 1, ax)
        ax.legend()
        ax.set_xlabel("t/s")
        ax.set_ylabel("x/1")
    plt.show()

if __name__ == "__main__":
    main()
