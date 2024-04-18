#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 21:57:07 2023

@author: tuanaoyuncu
"""

# import numpy as np
# import matplotlib.pyplot as plt
# t=np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_185_C1/S_007_C1/TOA_MGA_20231013_007_000001_t.npy")
# x=np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_185_C1/S_007_C1/TOA_MGA_20231013_007_000001_x.npy")
# print(t)
# plt.plot(t,x[0,:])
# plt.legend()
# plt.show()

# def find_point_of_increase(file_path, target_x):
#     # Read the file and extract x and y values
#     with open(file_path, '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0001/C_001/H_185_C1/S_007_C1/TOA_MGA_20231013_007_000001_t.npy') as file:
#         lines = file.readlines()
#         x_values = []
#         y_values = []
#         for line in lines:
#             x, y = map(float, line.strip().split())  # Assuming x and y are space-separated
#             x_values.append(x)
#             y_values.append(y)

#     # Iterate through x and y values to find the point of increase
#     for i in range(1, len(x_values)):
#         if x_values[i] >= target_x and y_values[i] > y_values[i - 1]:
#             return x_values[i], y_values[i]  # Found the point of increase

#     # If no increase is found
#     return None, None

# # Example usage:
# file_path = "your_file.txt"
# target_x = 0.002  # Specify the target x value
# x_increase, y_increase = find_point_of_increase(file_path, target_x)
# if x_increase is not None:
#     print(f"The y value starts to increase at x = {x_increase}, y = {y_increase}")
# else:
#     print("No increase found for the given x value.")

import numpy as np
import matplotlib.pyplot as plt

# Load data
t = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0003/C_004_0003/H_080_C4_0003/S_038/TOA_MGA_20240228_0038_001_t.npy")
x = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/0003/C_004_0003/H_080_C4_0003/S_038/TOA_MGA_20240228_0038_001_x.npy")

# Plot data
plt.plot(t, x[0, :])
plt.xlabel('t')
plt.ylabel('x')
plt.vlines(1, 0,0.05)
print(x)
plt.legend()
plt.show()

def find_point_of_increase(t_values, x_values, target_x):
    # Iterate through x and y values to find the point of increase
    for i in range(1, len(t_values)):
        if t_values[i] >= target_x and x_values[i] > x_values[i - 1]:
            return t_values[i], x_values[i]  # Found the point of increase

    # If no increase is found
    return None, None

# Example usage:
target_x = 4
x_increase, y_increase = find_point_of_increase(t, x[0, :], target_x)
if x_increase is not None:
    print(f"The x value starts to increase at t = {x_increase}, x = {y_increase}")
else:
    print("No increase found for the given x value.")
