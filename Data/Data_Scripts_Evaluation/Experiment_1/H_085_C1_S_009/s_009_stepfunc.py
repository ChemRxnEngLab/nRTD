#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:24:49 2023

@author: tuanaoyuncu
"""
import matplotlib.pyplot as plt
import numpy as np
t = np.linspace(0, 41, 201)
x = np.piecewise(t, [t < 1, t >= 1], [0, 0.05])  # like a true/false
plt.step(t, x, where='post')
plt.xlabel('t/s')
plt.ylabel('x/1')
plt.xticks(np.arange(min(t), max(t) + 5, 5))  # digits
plt.yticks([0, 0.05])
plt.show()
np.save('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/Data_Scripts_Evaluation/Experiment_1/H_085_C1_S_009/t_target.npy', t)
np.save('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/Data_Scripts_Evaluation/Experiment_1/H_085_C1_S_009/x_target.npy', x)
