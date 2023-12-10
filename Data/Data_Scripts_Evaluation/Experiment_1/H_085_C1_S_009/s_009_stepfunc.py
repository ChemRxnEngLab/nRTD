#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:24:49 2023

@author: tuanaoyuncu
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 3, 100)
y = np.piecewise(x, [x < 1.1, x >= 1.1], [0, 0.0505])#like a true/false
plt.step(x, y, where='post')
plt.xlabel('t/s')
plt.ylabel('x/1')
plt.xticks(np.arange(min(x), max(x) + 0.5, 0.5))#digits
plt.yticks([0, 0.0505])
plt.show()

