#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 21:57:07 2023

@author: tuanaoyuncu
"""

import numpy as np
import matplotlib.pyplot as plt
# t = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_135_C1/S_010_C1_001/TOA_MGA_20231020_010_000001_t_processed.npy")
# x = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_135_C1/S_010_C1_001/TOA_MGA_20231020_010_000001_x_processed.npy")
# print(t)
# plt.plot(t,x)
# plt.legend()
# plt.show()


t1 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_085_C1/S_009_C1_001/TOA_MGA_20231020_009_000001_t_processed.npy")
x1 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_085_C1/S_009_C1_001/TOA_MGA_20231020_009_000001_x_processed.npy")
x1 /= 0.05
t2 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_135_C1/S_010_C1_001/TOA_MGA_20231020_010_000001_t_processed.npy")
x2 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_135_C1/S_010_C1_001/TOA_MGA_20231020_010_000001_x_processed.npy")
x2 /= 0.0333
t3 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_185_C1/S_007_C1_001/TOA_MGA_20231013_007_000001_t_processed.npy")
x3 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_185_C1/S_007_C1_001/TOA_MGA_20231013_007_000001_x_processed.npy")
x3 /= 0.025
plt.plot(t1, x1, label='100', color='red')
plt.plot(t2, x2, label='150', color='black')
plt.plot(t3, x3, label='200', color='green')
plt.xlabel('t')
plt.ylabel('x / (corresponding values)')
plt.legend()
plt.show()

t4 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_085_C2/S_012_C2_001/TOA_MGA_20231020_012_000001_t_processed.npy")
x4 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_085_C2/S_012_C2_001/TOA_MGA_20231020_012_000001_x_processed.npy")
x4 /= 0.05

t5 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_135_C2/S_013_C2_001/TOA_MGA_20231020_013_000001_t_processed.npy")
x5 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_135_C2/S_013_C2_001/TOA_MGA_20231020_013_000001_x_processed.npy")
x5 /= 0.0333

t6 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_185_C2/S_014_C2_001/TOA_MGA_20231020_014_000001_t_processed.npy")
x6 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_185_C2/S_014_C2_001/TOA_MGA_20231020_014_000001_x_processed.npy")
x6 /= 0.025


plt.plot(t4, x4, label='100', color='red')
plt.plot(t5, x5, label='150', color='black')
plt.plot(t6, x6, label='200', color='green')
plt.xlabel('t')
plt.ylabel('x / (corresponding values)')
plt.legend()
plt.show()

t7 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0003/C_004_0003/H_080_C4_0003/S_038/TOA_MGA_20240228_0038_001_t_processed.npy")
x7 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0003/C_004_0003/H_080_C4_0003/S_038/TOA_MGA_20240228_0038_001_x_processed.npy")
x7 /= 0.1

t8 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0003/C_004_0003/H_130_C4_0003/S_039/TOA_MGA_20240228_0039_001_t_processed.npy")
x8 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0003/C_004_0003/H_130_C4_0003/S_039/TOA_MGA_20240228_0039_001_x_processed.npy")
x8 /= 0.066

t9 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0003/C_004_0003/H_180_C4_0003/S_040/TOA_MGA_20240228_0040_001_t_processed.npy")
x9 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0003/C_004_0003/H_180_C4_0003/S_040/TOA_MGA_20240228_0040_001_x_processed.npy")
x9 /= 0.05

plt.plot(t4, x7, label='100', color='red')
plt.plot(t5, x8, label='150', color='black')
plt.plot(t6, x9, label='200', color='green')
plt.xlabel('t')
plt.ylabel('x / (corresponding values)')
plt.legend()
plt.show()


t10 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_085_C1/S_009_C1_001/TOA_MGA_20231020_009_000001_t_processed.npy")
x10 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_085_C1/S_009_C1_001/TOA_MGA_20231020_009_000001_x_processed.npy")
x10 /= 0.05
t11 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_085_C2/S_012_C2_001/TOA_MGA_20231020_012_000001_t_processed.npy")
x11 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_002/H_085_C2/S_012_C2_001/TOA_MGA_20231020_012_000001_x_processed.npy")
x11 /= 0.05
t12 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0003/C_004_0003/H_080_C4_0003/S_038/TOA_MGA_20240228_0038_001_t_processed.npy")
x12 = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0003/C_004_0003/H_080_C4_0003/S_038/TOA_MGA_20240228_0038_001_x_processed.npy")
x12 /= 0.1
plt.plot(t4, x10, label='1st', color='red')
plt.plot(t5, x11, label='2nd', color='black')
plt.plot(t6, x12, label='4th', color='green')
plt.xlabel('t')
plt.ylabel('x [1]')
plt.legend()
plt.show()

