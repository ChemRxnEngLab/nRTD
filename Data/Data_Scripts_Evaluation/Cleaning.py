#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:49:20 2023

@author: tuanaoyuncu
"""
import numpy as np
import scipy as sc
x=np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_000001_x.npy")
t=np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_000001_t.npy")
f=sc.interpolate.interp1d(t,x[0,:])
print(f(3))
t_n=t[-1]
t_b=t_n-41
t_evel=np.linspace(t_b,t_n,500)
t_pretty=t_evel-t_b
x_evel=f(t_evel)
print(x_evel)
