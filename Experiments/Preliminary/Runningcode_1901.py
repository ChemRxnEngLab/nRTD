#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 00:11:35 2024

@author: tuanaoyuncu
"""

import sys
sys.path.append("/Users/tuanaoyuncu/Documents/GitHub/nRTD/lib")
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from nrtd import RTDModule

n_disc = 377 
t_input = torch.linspace(0, 31, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[:, :, t_input > 1] = 0.05  
n_conv = 500
t_conv = torch.linspace(0, 41, n_conv)
c_out = torch.zeros((1, 1, n_conv))  
c_out[:, :, t_conv > 1] = 0.05 
plt.show()
model = RTDModule(
    kernel_size=122,
    learning_rate=1e-3,
)
c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 10, model.kernel_size)
plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")
plt.plot(t_conv, c_out[0, 0, :].numpy(), label="Exp", color="green")
plt.plot(t_conv, c_conv[0, 0, :].detach().numpy(), label="Predicted", color="red")
plt.plot(t_E, E, label="E", color="orange")
plt.legend()
plt.xlim((0,10))
plt.ylim((0,0.2))
plt.show()
ds = TensorDataset(c_in, c_out)
dl = DataLoader(ds, batch_size=1)
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=10000,
)
trainer.fit(model, dl)
c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 10, model.kernel_size)
plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")
plt.plot(t_conv, c_out[0, 0, :].numpy(), label="Exp", color="green")
plt.plot(t_conv, c_conv[0, 0, :].detach().numpy(), label="Predicted", color="red")
plt.plot(t_E, E, label="E", color="orange")
plt.xlim((0,10))
plt.ylim((0,0.2))
plt.legend()
plt.show()
