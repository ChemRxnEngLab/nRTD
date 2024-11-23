import sys
import os
module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
from nRTD import RTDModule
from lightning.pytorch import loggers as pl_loggers

tau_5_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Unified_time_delay'

t_conv_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'time_J2_tau1_alpha0.2.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'concentration_J2_tau1_alpha0.2.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

n_disc = 111
t_input = torch.linspace(0, 20, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[:, :, t_input > 5] = 1
c_out_list = []
t_conv_list = []
c_out_list.append(c_out_tau)
t_conv_list.append(t_conv_tau)

c_out = torch.cat(c_out_list, dim=0)
t_conv = torch.cat(t_conv_list, dim=0)

plt.show()

model = RTDModule(
    kernel_size=88,
    learning_rate=10e-3,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)

c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 25, model.kernel_size)
E = E / E.max()
j = 0  

plt.figure()
plt.plot(t_input, c_in[j, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    plt.plot(
        t_conv[j, i, :].numpy(),
        c_out[j, i, :].numpy(),
        label="Tau 5.0",
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
plt.ylim((0, 1.1))
plt.show()

print(model(c_in).size())

ds = TensorDataset(c_in, c_out)
dl = DataLoader(ds, batch_size=20, shuffle=True)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=15000,
    deterministic=True
)
trainer.fit(model, dl)
trainer.test(model, dl)

c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 25, model.kernel_size)
E = E / E.max()


plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    plt.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label="",
        color="green",
    )

plt.plot(
    t_conv[0, 0, :].numpy(),
    c_conv[0, 0, :].detach().numpy(),
    label="Predicted",
    color="red",
)

t_plot = torch.linspace(0, 40, 1000).numpy()

E_expected=3.24642*np.exp(-2.34833*t_plot)*t_plot+0.0392897* np.exp(-0.851669 *t_plot) *t_plot - 0.477252 *np.exp(-2.34833* t_plot) + 0.477252* np.exp(-0.851669 *t_plot)

#E_expected[t_plot < 5] = 0

E_expected =E_expected /E_expected.max()

plt.plot(t_plot, E_expected, label="E_Expected", color="purple", linestyle='--')
plt.plot(t_E, E, label="E_predicted", color="orange")
#plt.savefig("Figure_conv_unifiedtdelay_J2_tau1")
plt.xlim((0, 50))
plt.ylim((0, 1.1))
plt.legend()
plt.show()

