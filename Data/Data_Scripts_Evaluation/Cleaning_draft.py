#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 20:39:18 2023

@author: tuanaoyuncu
"""


import numpy as np
import scipy.interpolate as sc
from pathlib import Path

WD = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD")
DATA = WD / "Data"

def main():
    folder = DATA / "C_001" / "H_085_C1" / "S_009_C1"

    for data_file_x in folder.glob("TOA_MGA_*_x.npy"):
        data_filename = data_file_x.stem

        # Load _x.npy
        x = np.load(folder / (data_filename + "_x.npy"))

        # Load corresponding _t.npy
        t = np.load(folder / (data_filename + "_t.npy"))

        f = sc.interpolate.interp1d(t, x[0, :])
        t_n = t[-1]
        t_b = t_n - 41
        t_evel = np.linspace(t_b, t_n, 500)
        t_pretty = t_evel - t_b
        x_evel = f(t_evel)

        # Save processed data
        np.save(folder / (data_filename + "_t_processed.npy"), t_pretty)
        np.save(folder / (data_filename + "_x_processed.npy"), x_evel)

        print(x_evel)
        print(t_evel)

if __name__ == "__main__":
    main()

