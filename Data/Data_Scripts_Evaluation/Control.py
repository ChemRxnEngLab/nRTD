#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 21:57:07 2023

@author: tuanaoyuncu
"""

import numpy as np
import matplotlib.pyplot as plt
t=np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_002/H_085_C2/S_012_C2/TOA_MGA_20231020_012_000001_t.npy")
x=np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_002/H_085_C2/S_012_C2/TOA_MGA_20231020_012_000001_x.npy")
print(x)
plt.plot(t,x[0,:])
plt.legend()
plt.show()

