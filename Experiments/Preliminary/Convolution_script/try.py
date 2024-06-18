#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 23:23:32 2024

@author: tuanaoyuncu
"""
import sys

sys.path.append("/Users/tuanaoyuncu/Documents/GitHub/nRTD/lib")
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from nrtd import RTDModule
import numpy as np


n_disc = 377
t_input = torch.linspace(0, 31, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[:, :, t_input > 1] = 0.05
c_out_np = np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_000001_x_processed.npy")
t_out_np=np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_000001_t_processed.npy")
n_conv = 122
c_out = torch.tensor(np.stack([c_out_np] * 1), dtype=torch.float32)
t_out = torch.tensor(np.stack([t_out_np] * 1), dtype=torch.float32)



model = RTDModule(
    kernel_size=500,
    learning_rate=1e-3,
)

# Visualize initial prediction and parameters
c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 10, model.kernel_size)
# Plot initial signals
# plt.figure()
# plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")
# plt.plot(t_input, c_out[0, 0, :].numpy(), label="Loaded c_out - Sample 1", color="green")
# plt.xlim((0, 10))
# plt.ylim((0, 0.2))
# plt.legend()
# plt.show()

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
# Plot post-training signals
# plt.figure()
# plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")
# plt.plot(t_input, c_out[0, 0, :].numpy(), label="Loaded c_out - Sample 1", color="green")
# plt.plot(t_input, c_conv[0, 0, :].detach().numpy(), label="Predicted", color="red")
# plt.plot(t_E, E, label="E", color="orange")
# plt.xlim((0, 10))
# plt.ylim((0, 0.2))
# plt.legend()
# plt.show()
