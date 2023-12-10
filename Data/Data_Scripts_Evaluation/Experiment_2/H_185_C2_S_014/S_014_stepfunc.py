#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:43:54 2023

@author: tuanaoyuncu
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
# WD = Path("/Users/tuanaoyuncu/Documents/GitHub/nRTD")
# DATA = WD / "Data"
# PWD=DATA / "C_002" / "H_185_C2" / "S_014_C2" 
x = np.linspace(0, 3, 100)
y = np.piecewise(x, [x < 1.8, x >= 1.8], [0, 0.0255])#like a true/false
plt.step(x, y, where='post')
plt.xlabel('t/s')
plt.ylabel('x/1')
plt.xticks(np.arange(min(x), max(x) + 0.5, 0.5))#digits
plt.yticks([0, 0.0255])
# filename = "S_014_stepfunc.png"
# filepath = PWD / filename
# plt.savefig(filepath)
plt.show()