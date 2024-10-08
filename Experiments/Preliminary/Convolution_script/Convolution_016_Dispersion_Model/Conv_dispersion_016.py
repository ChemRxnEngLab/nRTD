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
from lightning.pytorch import loggers as pl_loggers


# if wandb.run is not None:
#     wandb.finish()
# module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
# sys.path.append(module_path)

tau_5_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Dispersion_Model/Bo1_200_150s'
t_conv_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)


n_disc = 100
t_input = torch.linspace(0, 50, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[::2, :, t_input > 5] = 1
c_in[1::2, :, t_input < 5] = 1
c_out_list = [c_out_tau]
t_conv_list = [t_conv_tau]
c_out = torch.cat(c_out_list, dim=0)
t_conv = torch.cat(t_conv_list, dim=0)


print(f"c_in size: {c_in.size()}")
print(f"c_out size: {c_out.size()}")
print(f"t_conv size: {t_conv.size()}")

model = RTDModule(
    kernel_size=99,
    learning_rate=10e-3,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)

c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(5, 50, model.kernel_size)
E = E / E.max()
j = 0

plt.figure()
plt.plot(t_input, c_in[j, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    plt.plot(
        t_conv[j, i, :].numpy(),
        c_out[j, i, :].numpy(),
        label="Tau 5.0",
        color="green"
    )
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

# def expected_formula(t):
#     return np.where(t >= 2.5, 25 / (2 * t**3), 0)

# t_E_np = t_E.numpy()
# E_expected_np = expected_formula(t_E_np)

# # Normalize the expected E
# E_expected_np /= np.sum(E_expected_np)

print(model(c_in).size())
ds = TensorDataset(c_in, c_out)
dl = DataLoader(ds, batch_size=20, shuffle=True)
# wandb_logger = pl_loggers.WandbLogger(
#     project="nRTD",
#     log_model=True
# )

# trainer = pl.Trainer(
#     accelerator="gpu" if torch.cuda.is_available() else "cpu",
#     max_epochs=10000,
#     logger=wandb_logger,
#     deterministic=True,
# )

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=10000,
    deterministic=True,
)

trainer.fit(model, dl)
trainer.test(model, dl)
c_conv = model(c_in)
E = model.net.E[0]
# t_E = torch.linspace(5, 50, model.kernel_size)  
# E = E / torch.sum(E * (t_E[1] - t_E[0]))  
t_E = torch.linspace(5, 50, model.kernel_size)
E = E / E.max()
t_E_np = np.linspace(0, 50,500)

def expected_formula(t):
    return np.where(t >= 2.5, 1/2*(np.sqrt(1/(np.pi*(t/5))))*np.exp(-(1*((1-(t/5))**2))/(4*(t/5))), 0)
E_expected_np = expected_formula(t_E_np)
dt = t_E_np[1] - t_E_np[0]  
E_expected_np /= np.sum(E_expected_np * dt)
E_expected_np =E_expected_np /E_expected_np.max()


plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    plt.plot(
        t_conv[0, i, :].numpy(),
        c_out[0, i, :].numpy(),
        label="Bo1",
        color="green",
    )

plt.plot(
    t_conv[0, 0, :].numpy(),
    c_conv[0, 0, :].detach().numpy(),
    label="Predicted",
    color="red",
)
plt.plot(t_E_np+2.5, E_expected_np, label="E (Expected )", color="purple", linestyle="--")
plt.plot(t_E, E, label="E", color="orange")
plt.xlim((0, 100))
plt.ylim((0, 1.1))
plt.legend()
plt.savefig("Figure_conv_dispersion_Bo1_200disc_expected_epoch_10000_003")
plt.show()
