import sys

sys.path.append("/Users/tuanaoyuncu/Documents/GitHub/nRTD/lib")

import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from nrtd import RTDModule
import numpy as np
#from RTDModule import t_conv, c_out

### dummy data


n_disc = 33.333
t_input = torch.linspace(0, 41, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[:, :, t_input > 1] = 0.05
n_conv = 201
c_out = sorted(glob.glob('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_001/H_085_C1/S_009_C1_TOA_MGA_20231020_009_000001_x_*.npy'))
t_conv = sorted(glob.glob('/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_001/H_085_C1/S_009_C1_TOA_MGA_20231020_009_000001_t_*.npy'))
plt.plot(t_input, c_in[0, 0, :].numpy())
plt.plot(t_conv, c_out[0, 0, :].numpy())
plt.show()


model = RTDModule(
    kernel_size=167,
    learning_rate=1e-3,
)

c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 10, model.kernel_size)

plt.plot(t_input, c_in[0, 0, :].numpy())
plt.plot(t_conv, c_out[0, 0, :].numpy())
plt.plot(t_conv, c_conv[0, 0, :].detach().numpy())
plt.plot(t_E, E, label="E")
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

plt.plot(t_input, c_in[0, 0, :].numpy())
plt.plot(t_conv, c_out[0, 0, :].numpy())
plt.plot(t_conv, c_conv[0, 0, :].detach().numpy())
plt.plot(t_E, E, label="E")
plt.show()
