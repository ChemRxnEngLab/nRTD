#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from pathlib import Path

WD = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD")
DATA = WD / "Data"
PWD = DATA / "0003" / "C_004_0003" / "H_180_C4_0003" / "S_040" 

def process_data(file_num):
    
    x_file_path = PWD / f"TOA_MGA_20240228_0040_{file_num:03d}_x.npy"
    t_file_path = PWD / f"TOA_MGA_20240228_0040_{file_num:03d}_t.npy"

    x = np.load(x_file_path)
    t = np.load(t_file_path)[:, 0]
    
    print("Dimensions of x:", x.shape)
    print("Dimensions of t:", t.shape)

    f = sc.interpolate.interp1d(t, x[0, :])
    t_n = t[-1]
    t_b = t_n - 55
    t_evel = np.linspace(t_b, t_n, 500)
    t_pretty = t_evel - t_b
    x_evel = f(t_evel)

    # Save processed data
    np.save(PWD / f"TOA_MGA_20240228_0039_{file_num:03d}_t_processed.npy", t_pretty)
    np.save(PWD / f"TOA_MGA_20240228_0039_{file_num:03d}_x_processed.npy", x_evel)

    plt.plot(t_pretty, x_evel, label=f"File {file_num}") 

def main():

    start_file_num = 1
    end_file_num = 20

    for file_num in range(start_file_num, end_file_num+1):
    #for file_num in range(start_file_num, end_file_num):
     #     if file_num in [12]:
      #       continue
         process_data(file_num)


    plt.legend()
    plt.xlabel("t/s")
    plt.ylabel("x/1")
    #plt.savefig("S_0040_processed_eliminated.png")
    plt.show()

if __name__ == "__main__":
    main()
