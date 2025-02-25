#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:59:17 2023

@author: tuanaoyuncu
"""
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 51, 500)
y = np.piecewise(x, [x < 1, x >= 1], [0, 0.05])#like a true/false
plt.step(x, y, where='post')
plt.xlabel('t/s')
plt.ylabel('x/1')
plt.xticks(np.arange(min(x), max(x) + 5, 5))#digits
plt.yticks([0, 0.05])
plt.show()