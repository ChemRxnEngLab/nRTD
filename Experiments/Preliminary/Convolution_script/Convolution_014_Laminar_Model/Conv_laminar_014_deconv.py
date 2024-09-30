import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from scipy.optimize import curve_fit

module_path = os.path.expanduser("~/Documents/GitHub/nRTD/lib")
sys.path.append(module_path)
from nrtd import RTDModule

tau_5_dir = '/Users/tuanaoyuncu/Documents/GitHub/nRTD/Experiments/Preliminary/Litrature/Laminar_Flow_Model/tau_5.0'
t_conv_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'time.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
c_out_tau = torch.tensor(np.load(os.path.join(tau_5_dir, 'concentration.npy')), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

n_disc = 500
t_input = torch.linspace(0, 35, n_disc)
c_in = torch.zeros((1, 1, n_disc))
c_in[::2, :, t_input > 1] = 1
c_in[1::2, :, t_input < 1] = 1
c_out_list = [c_out_tau]
t_conv_list = [t_conv_tau]
c_out = torch.cat(c_out_list, dim=0)
t_conv = torch.cat(t_conv_list, dim=0)

model = RTDModule(
    kernel_size=291,
    learning_rate=10e-3,
    use_scheduler=True,
    scheduler_kwargs={"factor": 0.5, "patience": 80},
)

ds = TensorDataset(c_in, c_out)
dl = DataLoader(ds, batch_size=20, shuffle=True)

wandb_logger = pl_loggers.WandbLogger(project="nRTD", log_model=True)
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=100,
    logger=wandb_logger, deterministic=True
)

trainer.fit(model, dl)
trainer.test(model, dl)

c_conv = model(c_in)
E = model.net.E[0]
t_E = torch.linspace(0, 25, model.kernel_size)
E = E / E.max()

t_input_np = t_input.numpy()
c_in_np = c_in[0, 0, :].numpy()
t_conv_np = t_conv[0, :, :].numpy()
c_out_np = c_out[0, :, :].numpy()
c_conv_np = c_conv[0, 0, :].detach().numpy()
t_E_np = t_E.numpy()
E_np = E

def expected_formula(t):
    return np.where(t >= 2.5, 25 / (2 * t**3), 0)
E_expected_np = expected_formula(t_E_np)

def fit_function(t, a, b, c, d):
    return a * np.exp(b * t) + c * t + d
popt, pcov = curve_fit(fit_function, t_input_np, c_conv_np, p0=[1, 1, 1, 1])
a, b, c, d = popt
print(f"Fitted Parameters:\n a = {a}\n b = {b}\n c = {c}\n d = {d}")

c_conv_fitted = fit_function(t_input_np, *popt)

plt.figure()
plt.plot(t_input_np, c_in_np, label="Input Concentration (c_in)", color="blue")
plt.plot(t_input_np, c_conv_np, label=" Predicted (c_conv)", color="red", linestyle="--")
plt.plot(t_input_np, c_conv_fitted, label="Fitted Curve", color="green")
plt.plot(t_E_np, E_np, label="E (Model Output)", color="orange")
plt.plot(t_E_np, E_expected_np, label="E (Expected )", color="purple", linestyle="--")
plt.xlim((0, 40))
plt.ylim((0, 1.1))
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()
plt.show()
