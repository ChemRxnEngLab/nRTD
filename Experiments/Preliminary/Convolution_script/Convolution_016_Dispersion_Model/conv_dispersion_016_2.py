import sys
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import wandb
from nrtd import RTDModule
from lightning.pytorch import loggers as pl_loggers

if wandb.run is not None:
    wandb.finish()

module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)

tau_5_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Dispersion_Model/Bo1_200'
t_conv_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

n_disc = 60
t_input = torch.linspace(0, 70, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[::2, :, t_input > 1] = 1
c_in[1::2, :, t_input < 1] = 1
c_out_list = [c_out_tau]
t_conv_list = [t_conv_tau]
c_out = torch.cat(c_out_list, dim=0)
t_conv = torch.cat(t_conv_list, dim=0)

model = RTDModule(
    kernel_size=139,
    learning_rate=10e-3,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)

c_conv = model(c_in)
E_predicted = model.net.E[0]

# Ensure E_predicted is a tensor before using torch.sum
if isinstance(E_predicted, np.ndarray):
    E_predicted = torch.tensor(E_predicted, dtype=torch.float32)

E_predicted = E_predicted / torch.sum(E_predicted)
t_E = torch.linspace(0, 0, model.kernel_size)
E = E_predicted / E_predicted.max()

plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    plt.plot(t_conv[0, i, :].numpy(), c_out[0, i, :].numpy(), label="Bo1", color="green")

plt.plot(t_conv[0, 0, :].numpy(), c_conv[0, 0, :].detach().numpy(), label="Predicted", color="red")
plt.plot(t_E.numpy(), E.numpy(), label="E", color="orange")
plt.xlim((0, 40))
plt.ylim((0, 1.1))
plt.legend()
plt.show()

def expected_formula(t):
    return np.where(t >= 2.5, 1/2*(np.sqrt(1/(np.pi*(t/5))))*np.exp(-(1*((1-(t/5))**2))/(4*(t/5))), 0)

t_E_np = t_E.numpy()
E_expected_np = expected_formula(t_E_np)

plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    plt.plot(t_conv[0, i, :].numpy(), c_out[0, i, :].numpy(), label="Bo1", color="green")

plt.plot(t_conv[0, 0, :].numpy(), c_conv[0, 0, :].detach().numpy(), label="Predicted", color="red")
plt.plot(t_E_np, E_expected_np, label="E (Expected)", color="purple", linestyle="--")
plt.plot(t_E.numpy(), E_predicted.numpy(), label="E (Predicted)", color="orange")
plt.xlim((0, 100))
plt.ylim((0, 1.1))
plt.legend()
plt.savefig("Figure_conv_dispersion_Bo1_300disc_expected")
plt.show()

ds = TensorDataset(c_in, c_out)
dl = DataLoader(ds, batch_size=20, shuffle=True)
wandb_logger = pl_loggers.WandbLogger(project="nRTD", log_model=True)

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=100,
    logger=wandb_logger,
    deterministic=True,
)

trainer.fit(model, dl)
trainer.test(model, dl)

c_conv = model(c_in)
E_predicted = model.net.E[0]
t_E = torch.linspace(0, 30, model.kernel_size)

plt.figure()
plt.plot(t_input, c_in[0, 0, :].numpy(), label="SF", color="blue")

for i in range(c_out.size(1)):
    plt.plot(t_conv[0, i, :].numpy(), c_out[0, i, :].numpy(), label="Bo1", color="green")

plt.plot(t_conv[0, 0, :].numpy(), c_conv[0, 0, :].detach().numpy(), label="Predicted", color="red")
plt.plot(t_E.numpy(), E_predicted.numpy(), label="E (Predicted)", color="orange")
plt.plot(t_E.numpy(), E_expected_np, label="E (Expected)", color="purple", linestyle="--")
plt.xlim((0, 100))
plt.ylim((0, 1.1))
plt.legend()
plt.savefig("Figure_conv_dispersion_Bo1_200disc_expected")
plt.show()
