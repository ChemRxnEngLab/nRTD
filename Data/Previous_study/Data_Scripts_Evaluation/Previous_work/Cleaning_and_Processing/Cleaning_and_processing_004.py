#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 21:25:12 2023

@author: tuanaoyuncu
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from pathlib import Path

WD = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD")
DATA = WD / "Data"
PWD=DATA / "0003" / "C_001_0003" / "H_080_C1_0003" / "S_035"

def process_data(file_num):
    # Construct file paths for _x.npy and _t.npy
    x_file_path = PWD / f"TOA_MGA_20240228_0035_{file_num:03d}_x.npy"
    t_file_path = PWD / f"TOA_MGA_20240228_0035_{file_num:03d}_t.npy"

    # Load _x.npy and _t.npy
    x = np.load(x_file_path)
    t = np.load(t_file_path)[:,0]
    
    print("Dimensions of x:", x.shape)
    print("Dimensions of t:", t.shape)

    # Interpolate data
    f = sc.interpolate.interp1d(t, x[0, :])
    t_n = t[-1]
    t_b = t_n - 40
    t_evel = np.linspace(t_b, t_n, 500)
    t_pretty = t_evel - t_b
    x_evel = f(t_evel)

    # Save processed data
    np.save(PWD / f"TOA_MGA_20240228_0035_{file_num:03d}_t_processed.npy", t_pretty)
    np.save(PWD / f"TOA_MGA_20240228_0035_{file_num:03d}_x_processed.npy", x_evel)

    plt.plot(t_pretty, x_evel)#, label=f"File {file_num}")
    # plt.vlines(1.1, 0,0.05)
    # plt.hlines(0.0505, 0,55)

def main():
    # Specify the range of file numbers you want to process
    start_file_num = 1
    end_file_num = 20

    # Iterate over the file numbers and process each data file
    for file_num in range(start_file_num, end_file_num + 1):
         #if file_num in [2,4,6,8,10,12,14,16,18,20]:
            # continue
    #for file_num in [7,17]:
        process_data(file_num)

    # Show the legend and plot
    plt.legend()
    plt.xlabel("t/s")
    plt.ylabel("x/1")
    # #plt.savefig("S_009_processed.png")
    # #plt.savefig("S_009_processed.png")
    # #plt.savefig("S_009_max.png")
    # plt.show()

if __name__ == "__main__":
    main()
