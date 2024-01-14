import sys

sys.path.append("/Users/tuanaoyuncu/Documents/GitHub/nRTD/lib")

import torch
import matplotlib.pyplot as plt
from nrtd import RTDNet

kernel_size = 100
net = RTDNet(
    kernel_size=kernel_size,
    padding_mode="replicate",
    n_compartements=1,
)

n_disc = 33,333
t_0, t_i_end = (0, 10)
t = torch.linspace(t_0, t_i_end, n_disc)
t_E = torch.linspace(0, 10, kernel_size)
c_0 = torch.zeros((2, 1, n_disc))
c_0[:, :, t > t_i_end / 2] = 1

print(net(c_0).shape)

c = net(c_0).squeeze().detach().numpy()
# squeeze removes the batch dimension, maybe it is worth writing this as a part of the net, we will see
print(c.shape)

t_c = torch.linspace(0, 10 + t_i_end, net.output_shape(c_0))
plt.plot(t, c_0[0, :, :].squeeze(), label="c_0")
plt.plot(t_E, net.E[0], label="E")
plt.plot(t_c, c[0, :], label="c")
plt.legend()
plt.grid()
plt.show()
