#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from pathlib import Path

WD = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD")
DATA = WD / "Experiments"
PWD = DATA /"Preliminary"/"Kernel_size_variation"/"Kernel_size_10"/"H_085_C1"/"S_009_C1"

def process_data(file_num):
    
    x_file_path = PWD / f"TOA_MGA_20231020_009_{file_num:06d}_x.npy"
    t_file_path = PWD / f"TOA_MGA_20231020_009_{file_num:06d}_t.npy"

    x = np.load(x_file_path)
    t = np.load(t_file_path)#[:, 0]
    
    print("Dimensions of x:", x.shape)
    print("Dimensions of t:", t.shape)

    f = sc.interpolate.interp1d(t, x[0, :])
    t_n = t[-1]
    t_b = t_n - 41
    t_evel = np.linspace(t_b, t_n, 42)
    t_pretty = t_evel - t_b
    x_evel = f(t_evel)

    # Save processed data
    np.save(PWD / f"TOA_MGA_20231020_009_{file_num:06d}_t_processed.npy", t_pretty)
    np.save(PWD / f"TOA_MGA_20231020_009_{file_num:06d}_x_processed.npy", x_evel)

    plt.plot(t_pretty, x_evel, label=f"File {file_num}") 

def main():

    start_file_num = 1
    end_file_num = 21

    #for file_num in range(start_file_num, end_file_num+1):
    for file_num in range(start_file_num, end_file_num):
         #if file_num in [9,5]:
         #   continue
         process_data(file_num)


    plt.legend()
    plt.xlabel("t/s")
    plt.ylabel("x/1")
    plt.savefig("S_009_10.png")
    plt.show()

if __name__ == "__main__":
    main()
