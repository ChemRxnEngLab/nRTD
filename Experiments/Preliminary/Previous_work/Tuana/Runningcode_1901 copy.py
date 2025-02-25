# import sys
# sys.path.append("/Users/tuanaoyuncu/Documents/GitHub/nRTD/lib")
# import torch
# from torch.utils.data import TensorDataset, DataLoader
# import lightning.pytorch as pl
# import matplotlib.pyplot as plt
# from nrtd import RTDModule
# import numpy as np


# n_disc = 377
# t_input = torch.linspace(0, 31, n_disc)
# c_in = torch.zeros((500, 1, n_disc))
# c_in[:, :, t_input > 1] = 0.05
# t_conv = torch.tensor(np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_000001_t_processed.npy"), dtype=torch.float32)
# c_out = torch.tensor(np.load("/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_000001_x_processed.npy"), dtype=torch.float32)
# c_out = c_out.unsqueeze(0).expand(500, -1, -1)
# t_conv = t_conv.unsqueeze(0).expand(500, -1)

# plt.show()
# print(f"c_in size: {c_in.size()}")
# print(f"c_out size: {c_out.size()}")
# print(f"t_conv size: {t_conv.size()}")

# model = RTDModule(
#     kernel_size=122,
#     learning_rate=1e-3,
# )
# c_conv = model(c_in)
# E = model.net.E[0]
# t_E = torch.linspace(0, 10, model.kernel_size)

# plt.figure()
# plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")
# plt.plot(t_conv[0, :].numpy(), c_out[0, 0, :].numpy(), label="Exp", color="green")
# plt.plot(t_conv[0, :].numpy(), c_conv[0, 0, :].detach().numpy(), label="Predicted", color="red")
# plt.plot(t_E, E, label="E", color="orange")
# plt.legend()
# plt.xlim((0, 10))
# plt.ylim((0, 0.2))
# plt.show()

# ds = TensorDataset(c_in, c_out)
# dl = DataLoader(ds, batch_size=1)

# trainer = pl.Trainer(
#     accelerator="auto",
#     max_epochs=100,
# )
# trainer.fit(model, dl)

# c_conv = model(c_in)
# E = model.net.E[0]
# t_E = torch.linspace(0, 10, model.kernel_size)

# plt.figure()
# plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")
# plt.plot(t_conv[0, :].numpy(), c_out[0, 0, :].numpy(), label="Exp", color="green")
# plt.plot(t_conv[0, :].numpy(), c_conv[0, 0, :].detach().numpy(), label="Predicted", color="red")
# plt.plot(t_E, E, label="E", color="orange")
# plt.xlim((0, 10))
# plt.ylim((0, 0.2))
# plt.legend()
# plt.show()

##########

import sys

# sys.path.append("/Users/tuanaoyuncu/Documents/GitHub/nRTD/lib")
sys.path.append("lib")
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from nRTD import RTDModule
import numpy as np


n_disc = 377
t_input = torch.linspace(0, 31, n_disc)
c_in = torch.zeros((10, 1, n_disc))
c_in[:, :, t_input > 1] = 0.05
file_numbers = range(1, 20, 2)
c_out_list = []
t_conv_list = []

for i, file_num in enumerate(file_numbers):
    # t_conv_path = f"/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_{file_num:06d}_t_processed.npy"
    t_conv_path = f"Data/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_{file_num:06d}_t_processed.npy"
    # c_out_path = f"/Users/tuanaoyuncu/Documents/GitHub/nRTD/Data/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_{file_num:06d}_x_processed.npy"
    c_out_path = f"Data/C_001/H_085_C1/S_009_C1/TOA_MGA_20231020_009_{file_num:06d}_x_processed.npy"
    t_conv = (
        torch.tensor(np.load(t_conv_path), dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    c_out = (
        torch.tensor(np.load(c_out_path), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )
    c_out_list.append(c_out)
    t_conv_list.append(t_conv)
c_out = torch.cat(c_out_list, dim=0)
t_conv = torch.cat(t_conv_list, dim=0)
# c_out = c_out.unsqueeze(2).expand(-1, -1, 1, -1)
# t_conv = t_conv.unsqueeze(2).expand(-1, -1, 1, -1)
plt.show()
print(f"c_in size: {c_in.size()}")
print(f"c_out size: {c_out.size()}")
print(f"t_conv size: {t_conv.size()}")

model = RTDModule(
    kernel_size=122,
    learning_rate=1e-3,
)
c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 10, model.kernel_size)

plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    plt.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label=f"Exp_{file_numbers[i]}",
        color="green",
    )

plt.plot(
    t_conv[0, 0, :].numpy(),
    c_conv[0, 0, :].detach().numpy(),
    label="Predicted",
    color="red",
)
plt.plot(t_E, E, label="E", color="orange")
plt.legend()
plt.xlim((0, 10))
plt.ylim((0, 0.2))
plt.show()

print(model(c_in).size())

ds = TensorDataset(c_in, c_out)
dl = DataLoader(ds, batch_size=1)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=1000,  ####if I will change the epoch then the load is also changing ???HEEH now its different?
)
trainer.fit(model, dl)
trainer.test(model, dl)

c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 10, model.kernel_size)

plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

# Iterate over the loaded files and plot them
for i in range(c_out.size(1)):
    plt.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label=f"Exp_{file_numbers[i]}",
        color="green",
    )

plt.plot(
    t_conv[0, 0, :].numpy(),
    c_conv[0, 0, :].detach().numpy(),
    label="Predicted",
    color="red",
)
plt.plot(t_E, E, label="E", color="orange")
plt.xlim((0, 10))
plt.ylim((0, 0.2))
plt.legend()
plt.show()
