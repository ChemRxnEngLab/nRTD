import sys

sys.path.append("lib")

import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from nRTD import RTDModule

### dummy data

n_disc = 33
t_input = torch.linspace(0, 2, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[:, :, t_input > 1] = 1

n_conv = 201
t_conv = torch.linspace(0, 12, n_conv)

c_out = torch.zeros((1, 1, n_conv))
c_out[:, :, t_conv > 4] = 1 / 3 * (t_conv[t_conv > 4] - 4)
c_out[:, :, t_conv > 7] = 1

plt.plot(t_input, c_in[0, 0, :].numpy())
plt.plot(t_conv, c_out[0, 0, :].numpy())

plt.show()


model = RTDModule(
    kernel_size=167,
)

c_conv = model(c_in)

plt.plot(t_input, c_in[0, 0, :].numpy())
plt.plot(t_conv, c_out[0, 0, :].numpy())
plt.plot(t_conv, c_conv[0, 0, :].detach().numpy())

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
