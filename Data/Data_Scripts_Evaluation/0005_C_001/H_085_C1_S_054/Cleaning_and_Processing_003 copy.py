#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 21:25:12 2023

@author: tuanaoyuncu
"""

import numpy as np
import scipy.interpolate as sc
import matplotlib.pyplot as plt
from pathlib import Path

WD = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD")
DATA = WD / "Data"

def process_data(file_num):
    # Construct file paths for _x.npy and _t.npy
    x_file_path = DATA / "0005" / "C_001" / "H_085_C1" / "S_054_C1" / f"TOA_MGA_20240701_0054_{file_num:06d}_x.npy"
    t_file_path = DATA / "0005" / "C_001" / "H_085_C1" / "S_054_C1" / f"TOA_MGA_20240701_0054_{file_num:06d}_t.npy"

    # Load _x.npy and _t.npy
    x = np.load(x_file_path)
    t = np.load(t_file_path)
    
    print("Dimensions of x:", x.shape)
    print("Dimensions of t:", t.shape)

    # Interpolate data
    f = sc.interpolate.interp1d(t, x[0, :])
    t_n = t[-1]
    t_b = t_n - 46
    t_evel = np.linspace(t_b, t_n, 500)
    t_pretty = t_evel - t_b
    x_evel = f(t_evel)

    # Save processed data
    np.save(DATA / "0005" / "C_001" / "H_085_C1" / "S_054_C1" / f"TOA_MGA_20240701_0054__{file_num:06d}_t_processed.npy", t_pretty)
    np.save(DATA / "0005" / "C_001" / "H_085_C1" / "S_054_C1" / f"TOA_MGA_20240701_0054_{file_num:06d}_x_processed.npy", x_evel)

    plt.plot(t_pretty, x_evel, label=f"File {file_num}")

def main():
    # Specify the range of file numbers you want to process
    start_file_num = 1
    end_file_num = 20

    # Iterate over the file numbers and process each data file
    for file_num in range(start_file_num, end_file_num + 1):
        process_data(file_num)

    plt.legend()
    # plt.xlabel()
    # plt.ylabel()
    plt.savefig("Figure_001_S_054", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
