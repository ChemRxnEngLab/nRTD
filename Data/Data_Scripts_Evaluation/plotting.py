#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 22:29:47 2023

@author: tuanaoyuncu
"""
import numpy as np
import scipy.interpolate as sc
import matplotlib.pyplot as plt
from pathlib import Path

WD = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD")
DATA = WD / "Data"

def data_plot():
    folder = DATA / "C_002" / "H_185_C2" / "S_014_C2"
    t_files = list(folder.glob("TOA_MGA_*_*_*_t_processed.npy"))
    for t_file in t_files:
        x_file = folder / f"{t_file.stem.replace('_t_processed', '_x_processed')}.npy"
        t_processed = np.load(t_file)
        x_processed = np.load(x_file)
        
        plt.plot(t_processed, x_processed, label=f"File {t_file.stem[-6:]}")

    # Show the legend and plot
    plt.legend()
    plt.xlabel("Processed Time")
    plt.ylabel("Processed Data")
    plt.title("Processed Data for Multiple Files")
    plt.show()
data_plot()
