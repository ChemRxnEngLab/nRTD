#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:09:58 2024

@author: tuanaoyuncu
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import os

def Cholete(t: npt.NDArray[np.float64], alpha: float, beta: float, tau: float, g: float) -> npt.NDArray[np.float64]: 
    print(f"tau = {tau}")
    print(f"alpha = {alpha}")
    print(f"beta = {beta}")
    print(f"g = {g}")
    
    H = np.where(t < g, 0, 1)
    k = ((1 - alpha) / (beta * tau))
    exp_term = (1 - alpha) * np.exp(k * (g - t))
    E = alpha * H - exp_term + (1 - alpha)
    E[E < 0] = 0
    # E /= np.trapz(E, t)  
    return E

t = np.linspace(0, 100, 500)
c_0 = np.zeros_like(t)
c_0[t > 5] = 1
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

beta_values = np.array([0.1, 0.3, 0.9])
base_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Cholete_Model'

for beta in beta_values:
    E = Cholete(t, 0.2, beta, 5, 2.5)
    
    E_t_normalized = E / np.sum(E)
    print(f"beta: {beta}, Integral of E: {np.sum(E)}")
    c_out_full = np.convolve(c_0, E_t_normalized, mode="full")
    t_conv_full = np.linspace(t[0], t[-1] + t[-1], len(c_out_full))
    valid_indices = t_conv_full <= 100
    t_conv = t_conv_full[valid_indices]
    c_out = c_out_full[valid_indices]
    Bo_dir = os.path.join(base_dir, f'beta{beta}')
    os.makedirs(Bo_dir, exist_ok=True)
    np.save(os.path.join(Bo_dir, 'time.npy'), t_conv)
    np.save(os.path.join(Bo_dir, 'concentration.npy'), E)

    ax1.plot(t, E, label=f'beta {beta}')
    ax2.plot(np.linspace(0, 100, len(c_out)), c_out, label=f'beta {beta}')

ax1.set_xlabel('t')
ax1.set_ylabel('E')
ax1.legend()

ax2.plot(t, c_0, label='c_0', linestyle='--', color='black')
ax2.set_xlim(0, 120)
ax2.set_ylim(0, 1.1)
ax2.set_xlabel('t')
ax2.set_ylabel('C')
ax2.legend()

plt.tight_layout()
plt.show()


for beta in beta_values:
    beta_dir = os.path.join(base_dir, f'beta{beta}')
    t_conv = np.load(os.path.join(beta_dir, 'time.npy'))  
    c_out = np.load(os.path.join(beta_dir, 'concentration.npy')) 
    print(f"beta = {beta}:")
    print(f"t shape: {t_conv.shape}")
    print(f"c shape: {c_out.shape}")
    print(f"t values: {t_conv[:10]}...")
    print(f"c values: {E[:10000]}...\n")

