#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 00:56:44 2023

@author: tuanaoyuncu
"""
import sys

sys.path.append("/Users/tuanaoyuncu/Documents/GitHub/nRTD/lib")

import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from nrtd import RTDModule
import numpy as np


n_disc = 33  
t_input = torch.linspace(0, 41, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[:, :, t_input > 1] = 0.05  
n_conv = 201
t_conv = torch.linspace(0, 10, n_conv)
c_out = torch.zeros((1, 1, n_conv))  
c_out[:, :, t_conv > 1] = 0.05 

plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")
plt.plot(t_input, c_out[0, 0, :].numpy(), label="Loaded c_out", color="green")
plt.xlim((0, 10))
plt.ylim((0, 0.2))
plt.legend()
plt.show()

#Initialize the RTDModule model
model = RTDModule(
    kernel_size=122,
    learning_rate=1e-3,
)

# Visualize initial prediction and parameters
c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 10, model.kernel_size)
plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")
plt.plot(t_input, c_out[0, 0, :].numpy(), label="Loaded c_out", color="green")
plt.plot(t_input, c_conv[0, 0, :].detach().numpy(), label="Predicted", color="red")
plt.plot(t_E, E, label="E", color="orange")
plt.legend()
plt.xlim((0, 10))
plt.ylim((0, 0.2))
plt.show()

# Create TensorDataset and DataLoader for training
ds = TensorDataset(c_in, c_out)
dl = DataLoader(ds, batch_size=1)

# Initialize PyTorch Lightning Trainer
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=10000,
)

# Train the model
trainer.fit(model, dl)

# Visualize post-training results
c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 10, model.kernel_size)
plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")
plt.plot(t_input, c_out[0, 0, :].numpy(), label="Loaded c_out", color="green")
plt.plot(t_input, c_conv[0, 0, :].detach().numpy(), label="Predicted", color="red")
plt.plot(t_E, E, label="E", color="orange")
plt.xlim((0, 10))
plt.ylim((0, 0.2))
plt.legend()
plt.show()

