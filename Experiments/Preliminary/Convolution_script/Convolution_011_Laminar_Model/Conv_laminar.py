import sys
import os
module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import wandb
from nrtd import RTDModule

tau_5_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Laminar_Flow_Model/tau_5.0'

t_conv_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)


n_disc = 239
t_input = torch.linspace(0, 30, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[::2, :, t_input > 5] = 1
c_in[1::2, :, t_input < 5] = 1

#file_numbers = range(1, 21)
c_out_list = []
t_conv_list = []

c_out_list.append(c_out_tau)
t_conv_list.append(t_conv_tau)


# for i, file_num in enumerate(file_numbers):

#     pass


c_out = torch.cat(c_out_list, dim=0)
t_conv = torch.cat(t_conv_list, dim=0)

plt.show()
print(f"c_in size: {c_in.size()}")
print(f"c_out size: {c_out.size()}")
print(f"t_conv size: {t_conv.size()}")

model = RTDModule(
    kernel_size=60,
    learning_rate=10e-4,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)
c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 40, model.kernel_size)
E = E / E.max()
j = 0  

plt.figure()
plt.plot(t_input, c_in[j, 0, :].numpy(), label="SF", color="blue")

# for i in range(c_out.size(1)):
#     plt.plot(
#         t_conv[j, i, :].numpy(),
#         c_out[j, i, :].numpy(),
#         label=f"Exp_{file_numbers[i]}" if i < len(file_numbers) else "Tau 1",
#         color="green",
#     )
for i in range(c_out.size(1)):
    plt.plot(
        t_conv[j, i, :].numpy(),
        c_out[j, i, :].numpy(),
        label= "Tau 5.0",
        color="green")
plt.plot(
    t_conv[j, 0, :].numpy(),
    c_conv[j, 0, :].detach().numpy(),
    label="Predicted",
    color="red",
)
plt.plot(t_E, E, label="E", color="orange")
plt.legend()
plt.xlim((0, 40))
plt.ylim((0,1.1))
plt.show()

print(model(c_in).size())


ds = TensorDataset(c_in, c_out)
dl = DataLoader(ds, batch_size=20, shuffle=True)


trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=10000,
    deterministic=True
)
trainer.fit(model, dl)
trainer.test(model, dl)


c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 40, model.kernel_size)
E = E / E.max()

plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    plt.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label="Tau 5.0",
        color="green",
    )

plt.plot(
    t_conv[0, 0, :].numpy(),
    c_conv[0, 0, :].detach().numpy(),
    label="Predicted",
    color="red",
)
plt.plot(t_E, E, label="E", color="orange")
plt.xlim((0, 20))
plt.ylim((0, 1.1))
plt.legend()

plt.savefig("Figure_conv_laminar_dis300")
plt.show()
