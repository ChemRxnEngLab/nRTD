#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:49:20 2023

@author: tuanaoyuncu
"""
from pathlib import Path
import numpy as np
import scipy as sc
#from MS_Calculation import cal_compostion, calc_calibration
WD = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD")
DATA = WD / "Data"

# def main():
#    folder = DATA / "C_002" / "H_135_C2" / "S_013_C2"
#    for data_file in folder.glob("TOA_MGA_*.npy"):
#        data_filename = data_file.stem
#        x=np.load("TOA_MGA_*_x.npy")
#        t=np.load("TOA_MGA_*_t.npy")
#        f=sc.interpolate.interp1d(t,x[0,:])
#        t_n=t[-1]
#        t_b=t_n-41
#        t_evel=np.linspace(t_b,t_n,500)
#        t_pretty=t_evel-t_b
#        x_evel=f(t_evel)
#        np.save(folder / (data_filename + "_t.npy"), t_pretty)
#        np.save(folder / (data_filename + "_x.npy"), x_evel)
#        print(x_evel)
#        print(t_evel)
#        if __name__ == "__main__":
#            main()
           
import numpy as np
import scipy.interpolate as sc

def main():
    folder = DATA / "C_001" / "H_085_C1" / "S_009_C1"
    for data_file in folder.glob("TOA_MGA_*_*.npy"):
        data_filename = data_file.stem
        x = np.load(folder / (data_filename + "x.npy"))
        t = np.load(folder / (data_filename + "t.npy"))
        f = sc.interpolate.interp1d(t, x[0, :])
        t_n = t[-1]
        t_b = t_n - 41
        t_evel = np.linspace(t_b, t_n, 500)
        t_pretty = t_evel - t_b
        x_evel = f(t_evel)
        np.save(folder / (data_filename + "_t.npy"), t_pretty)
        np.save(folder / (data_filename + "_x.npy"), x_evel)
        print(x_evel)
        print(t_evel)

if __name__ == "__main__":
    main()
         
           
           
           
   # cleaning_files = list(folder.glob("TOA_MGA_*_x.npy))
   # cleaning_files = list(folder.glob("TOA_MGA_*_t.npy))
   # x=np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_000001_x.npy")
   # t=np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_000001_t.npy")
# f=sc.interpolate.interp1d(t,x[0,:])
# #print(f(3))
# t_n=t[-1]
# t_b=t_n-41
# t_evel=np.linspace(t_b,t_n,500)
# t_pretty=t_evel-t_b
# x_evel=f(t_evel)
# np.save(folder / (data_filename + "_x.npy"), t_pretty)
# np.save(folder / (data_filename + "_t.npy"), x_evel)
# # print(x_evel)
# # print(t_evel)