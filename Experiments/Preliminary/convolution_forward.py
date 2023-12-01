import sys

sys.path.append("lib")

import torch
import matplotlib.pyplot as plt
from nRTD import RTDNet

net = RTDNet(kernel_size=100, padding_mode="replicate", n_compartements=1)

n_disc = 100
t = torch.linspace(0, 10, n_disc)

c_0 = torch.zeros((1, n_disc))
c_0[:, t > 3] = 1

c = net(c_0).squeeze().detach().numpy()
# squeeze removes the batch dimension, maybe it is worth writing this as a part of the net, we will see
print(c.shape)

t_c = torch.linspace(0, 20, net.output_shape(c_0))


plt.plot(t, c_0.squeeze(), label="c_0")
plt.plot(t, net.E[0], label="E")
plt.plot(t_c, c, label="c")
plt.legend()
plt.show()
